"""Custom RL algorithms (PPO, SAC, TD3) tailored for the denoising task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


EPS = 1e-6


def atanh_clipped(x: torch.Tensor) -> torch.Tensor:
    """Stable inverse tanh used for squashed Gaussian log-probs."""
    x = torch.clamp(x, -1 + 1e-6, 1 - 1e-6)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class MLP(nn.Module):
    """Simple feed-forward network."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], activation=nn.ReLU):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionScaler:
    """Maps between tanh-space actions and actual env actions."""

    def __init__(self, low: np.ndarray, high: np.ndarray, device: torch.device):
        low = low.astype(np.float32)
        high = high.astype(np.float32)
        self.low = low
        self.high = high
        self.center = (high + low) / 2.0
        self.scale = (high - low) / 2.0
        self.center_t = torch.as_tensor(self.center, device=device)
        self.scale_t = torch.as_tensor(self.scale, device=device)

    def to_env(self, action_tanh: np.ndarray) -> np.ndarray:
        return (action_tanh * self.scale + self.center).astype(np.float32)

    def tensor_to_env(self, action_tanh: torch.Tensor) -> torch.Tensor:
        return action_tanh * self.scale_t + self.center_t

    def env_to_tanh(self, action_env: np.ndarray) -> np.ndarray:
        return np.clip((action_env - self.center) / (self.scale + EPS), -0.999999, 0.999999)


class GaussianPolicy(nn.Module):
    """State-dependent Gaussian policy with tanh squashing."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Tuple[int, ...],
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
    ):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dims)
        last_dim = hidden_dims[-1] if hidden_dims else obs_dim
        self.mu_head = nn.Linear(last_dim, act_dim)
        self.log_std_head = nn.Linear(last_dim, act_dim)
        self.log_std_bounds = log_std_bounds

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(obs)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        min_log_std, max_log_std = self.log_std_bounds
        log_std = torch.clamp(log_std, min_log_std, max_log_std)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self(obs)
        std = torch.exp(log_std)
        noise = torch.randn_like(mu)
        pre_tanh = mu + noise * std
        action = torch.tanh(pre_tanh)
        log_prob = (
            Normal(mu, std).log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)
        entropy = (0.5 + 0.5 * np.log(2 * np.pi) + log_std).sum(dim=-1, keepdim=True)
        return action, log_prob, entropy

    def log_prob(self, obs: torch.Tensor, action_tanh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self(obs)
        std = torch.exp(log_std)
        pre_tanh = atanh_clipped(action_tanh)
        log_prob = (
            Normal(mu, std).log_prob(pre_tanh) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)
        entropy = (0.5 + 0.5 * np.log(2 * np.pi) + log_std).sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self(obs)
        return torch.tanh(mu)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dims)
        last_dim = hidden_dims[-1] if hidden_dims else obs_dim
        self.head = nn.Linear(last_dim, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.backbone(obs)
        return torch.tanh(self.head(x))


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        layers = []
        last_dim = obs_dim + act_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class RolloutBuffer:
    """Stores on-policy rollouts for PPO."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(
        self,
        obs: np.ndarray,
        action_tanh: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.observations.append(obs.astype(np.float32))
        self.actions.append(action_tanh.astype(np.float32))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(float(done))

    def __len__(self) -> int:
        return len(self.rewards)

    def compute(self, last_value: float, gamma: float, gae_lambda: float) -> Dict[str, torch.Tensor]:
        values = self.values + [float(last_value)]
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        gae = 0.0
        for step in reversed(range(len(self.rewards))):
            mask = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * values[step + 1] * mask - values[step]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[step] = gae
        returns = advantages + np.array(values[:-1], dtype=np.float32)
        obs = torch.as_tensor(np.array(self.observations), device=self.device)
        actions = torch.as_tensor(np.array(self.actions), device=self.device)
        old_log_probs = torch.as_tensor(np.array(self.log_probs).reshape(-1, 1), device=self.device)
        advantages_t = torch.as_tensor(advantages.reshape(-1, 1), device=self.device)
        returns_t = torch.as_tensor(returns.reshape(-1, 1), device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        return {
            "obs": obs,
            "actions": actions,
            "log_probs": old_log_probs,
            "advantages": advantages_t,
            "returns": returns_t,
        }


class ReplayBuffer:
    """Simple replay buffer for SAC/TD3."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs, action_tanh, reward, next_obs, done) -> None:
        self.obs[self.idx] = obs
        self.actions[self.idx] = action_tanh
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, len(self), size=batch_size)
        batch = {
            "obs": torch.as_tensor(self.obs[idxs], device=device),
            "actions": torch.as_tensor(self.actions[idxs], device=device),
            "rewards": torch.as_tensor(self.rewards[idxs], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idxs], device=device),
            "dones": torch.as_tensor(self.dones[idxs], device=device),
        }
        return batch


class PPOAgent:
    """Vanilla PPO (clip objective) with shared networks."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        hidden_dims: Tuple[int, ...] = (256, 256),
        lr: float = 3e-4,
        max_grad_norm: float = 1.0,
    ):
        self.device = device
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_dims).to(device)
        self.value_net = ValueNetwork(obs_dim, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()), lr=lr
        )
        self.scaler = ActionScaler(action_low, action_high, device)
        self.max_grad_norm = max_grad_norm

    def act(self, obs: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        if deterministic:
            action_tanh = self.policy.deterministic(obs_t)
            log_prob = None
        else:
            action_tanh, log_prob, _ = self.policy.sample(obs_t)
        value = self.value_net(obs_t)
        action_np = action_tanh.squeeze(0).detach().cpu().numpy()
        env_action = self.scaler.to_env(action_np)
        return env_action, action_np, (log_prob.item() if log_prob is not None else 0.0), value.item()

    def act_eval(self, obs: np.ndarray) -> np.ndarray:
        return self.act(obs, deterministic=True)[0]

    def evaluate_value(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        return self.value_net(obs_t).item()

    def update(
        self,
        rollout: Dict[str, torch.Tensor],
        clip_range: float,
        entropy_coef: float,
        value_coef: float,
        epochs: int,
        batch_size: int,
    ) -> Dict[str, float]:
        obs = rollout["obs"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"]
        advantages = rollout["advantages"]
        returns = rollout["returns"]

        num_samples = obs.shape[0]
        losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        for _ in range(epochs):
            idx_perm = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, batch_size):
                idx = idx_perm[start : start + batch_size]
                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_adv = advantages[idx]
                batch_returns = returns[idx]
                batch_old_logp = old_log_probs[idx]

                log_probs, entropy = self.policy.log_prob(batch_obs, batch_actions)
                values = self.value_net(batch_obs)

                ratio = torch.exp(log_probs - batch_old_logp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = entropy.mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                losses["policy_loss"] += policy_loss.item()
                losses["value_loss"] += value_loss.item()
                losses["entropy"] += entropy_loss.item()
                approx_kl = (batch_old_logp - log_probs).mean().item()
                losses["approx_kl"] += approx_kl

        updates = max(1, (epochs * num_samples) // batch_size)
        for k in losses:
            losses[k] /= updates
        return losses


class SACAgent:
    """Soft Actor-Critic with automatic entropy tuning."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        hidden_dims: Tuple[int, ...] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        tau: float = 0.005,
        target_entropy_scale: float = 1.0,
        replay_size: int = 200_000,
    ):
        self.device = device
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_dims).to(device)
        self.q1 = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.target_q1 = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.target_q2 = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -act_dim * target_entropy_scale
        self.tau = tau
        self.scaler = ActionScaler(action_low, action_high, device)
        self.replay = ReplayBuffer(obs_dim, act_dim, replay_size)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        if deterministic:
            action_tanh = self.policy.deterministic(obs_t)
        else:
            action_tanh, _, _ = self.policy.sample(obs_t)
        action_np = action_tanh.squeeze(0).detach().cpu().numpy()
        env_action = self.scaler.to_env(action_np)
        return env_action, action_np

    def update(self, batch_size: int, gamma: float) -> Dict[str, float]:
        batch = self.replay.sample(batch_size, self.device)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_obs)
            target_q = torch.min(
                self.target_q1(next_obs, next_action), self.target_q2(next_obs, next_action)
            )
            target = rewards + gamma * (1 - dones) * (target_q - self.alpha * next_log_prob)

        current_q1 = self.q1(obs, actions)
        current_q2 = self.q2(obs, actions)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_action, log_prob, _ = self.policy.sample(obs)
        q_val = torch.min(self.q1(obs, new_action), self.q2(obs, new_action))
        actor_loss = (self.alpha.detach() * log_prob - q_val).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
                target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)
            for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }


class TD3Agent:
    """Twin Delayed DDPG for continuous control."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        hidden_dims: Tuple[int, ...] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        replay_size: int = 200_000,
    ):
        self.device = device
        self.actor = DeterministicPolicy(obs_dim, act_dim, hidden_dims).to(device)
        self.actor_target = DeterministicPolicy(obs_dim, act_dim, hidden_dims).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.q1 = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.scaler = ActionScaler(action_low, action_high, device)
        self.replay = ReplayBuffer(obs_dim, act_dim, replay_size)
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_updates = 0

    def act(self, obs: np.ndarray, deterministic: bool = False, noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        action_tanh = self.actor(obs_t)
        action_np = action_tanh.squeeze(0).detach().cpu().numpy()
        if (not deterministic) and noise_std > 0.0:
            action_np = np.clip(action_np + np.random.normal(0, noise_std, size=action_np.shape), -1.0, 1.0)
        env_action = self.scaler.to_env(action_np)
        return env_action, action_np

    def update(self, batch_size: int, gamma: float) -> Dict[str, float]:
        batch = self.replay.sample(batch_size, self.device)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = torch.clamp(self.actor_target(next_obs) + noise, -1.0, 1.0)
            target_q = torch.min(
                self.q1_target(next_obs, next_action), self.q2_target(next_obs, next_action)
            )
            target = rewards + gamma * (1 - dones) * target_q

        current_q1 = self.q1(obs, actions)
        current_q2 = self.q2(obs, actions)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        info = {"critic_loss": critic_loss.item()}
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:
            actor_loss = -self.q1(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            info["actor_loss"] = actor_loss.item()
            with torch.no_grad():
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)
                for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
                    target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)
                for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
                    target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)
        return info

