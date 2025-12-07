"""Train denoising agents (PPO/SAC/TD3) with custom PyTorch implementations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import wandb

from .custom_algorithms import PPOAgent, SACAgent, TD3Agent, RolloutBuffer
from .dataset import load_feature_csv, split_dataset
from .env import DenoiseEnv


def log_hparams(wandb_run, args) -> None:
    if wandb_run is None:
        return
    base = {
        "hp/algo": args.algo,
        "hp/reward_scale": args.reward_scale,
        "hp/max_steps": args.max_steps,
        "hp/stoi_weight": args.stoi_weight,
        "hp/estoi_weight": args.estoi_weight,
        "hp/diversity_weight": args.diversity_weight,
        "hp/quality_interval": args.quality_interval,
        "hp/device": args.device,
    }
    base["hp/hidden_dims"] = ",".join(str(h) for h in args.hidden_dims)
    if args.algo == "ppo":
        base.update(
            {
                "hp/ppo/clip_range": args.clip_range,
                "hp/ppo/entropy_coef": args.entropy_coef,
                "hp/ppo/value_coef": args.value_coef,
                "hp/ppo/gae_lambda": args.gae_lambda,
                "hp/ppo/rollout_steps": args.rollout_steps,
                "hp/ppo/update_epochs": args.update_epochs,
                "hp/ppo/batch_size": args.batch_size,
            }
        )
    elif args.algo == "sac":
        base.update(
            {
                "hp/sac/random_steps": args.random_steps,
                "hp/sac/warmup_steps": args.warmup_steps,
                "hp/sac/updates_per_step": args.updates_per_step,
                "hp/sac/target_entropy_scale": args.target_entropy_scale,
                "hp/sac/tau": args.tau,
            }
        )
    else:  # TD3
        base.update(
            {
                "hp/td3/exploration_noise": args.exploration_noise,
                "hp/td3/policy_noise": args.policy_noise,
                "hp/td3/noise_clip": args.noise_clip,
                "hp/td3/policy_delay": args.policy_delay,
                "hp/td3/tau": args.tau,
            }
        )
    wandb_run.log(base, step=0)


def _serialize_args(args) -> Dict[str, object]:
    data = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            data[key] = str(value)
        elif isinstance(value, (list, tuple)):
            data[key] = list(value)
        else:
            data[key] = value
    return data


def save_checkpoint(
    algo: str,
    agent,
    save_dir: Path,
    args,
    eval_stats: Dict[str, float],
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "algo": algo,
        "timestamp": time.time(),
        "eval_stats": {k: float(v) for k, v in eval_stats.items()},
        "args": _serialize_args(args),
    }
    if isinstance(agent, PPOAgent):
        payload["policy_state"] = agent.policy.state_dict()
        payload["value_state"] = agent.value_net.state_dict()
        payload["action_low"] = agent.scaler.low
        payload["action_high"] = agent.scaler.high
    elif isinstance(agent, SACAgent):
        payload.update(
            {
                "policy_state": agent.policy.state_dict(),
                "q1_state": agent.q1.state_dict(),
                "q2_state": agent.q2.state_dict(),
                "target_q1_state": agent.target_q1.state_dict(),
                "target_q2_state": agent.target_q2.state_dict(),
                "log_alpha": float(agent.log_alpha.detach().cpu().item()),
                "action_low": agent.scaler.low,
                "action_high": agent.scaler.high,
            }
        )
    elif isinstance(agent, TD3Agent):
        payload.update(
            {
                "actor_state": agent.actor.state_dict(),
                "actor_target_state": agent.actor_target.state_dict(),
                "q1_state": agent.q1.state_dict(),
                "q2_state": agent.q2.state_dict(),
                "q1_target_state": agent.q1_target.state_dict(),
                "q2_target_state": agent.q2_target.state_dict(),
                "action_low": agent.scaler.low,
                "action_high": agent.scaler.high,
            }
        )
    checkpoint_path = save_dir / "model.pt"
    torch.save(payload, checkpoint_path)
    with (save_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(payload["eval_stats"], fh, indent=2)
    with (save_dir / "args.json").open("w", encoding="utf-8") as fh:
        json.dump(payload["args"], fh, indent=2)
    return checkpoint_path


def prepare_envs(args) -> Tuple[DenoiseEnv, DenoiseEnv, int]:
    dataset, feature_cols = load_feature_csv(args.features)
    train_ds, val_ds = split_dataset(dataset, test_size=0.1, seed=args.seed)
    train_env = DenoiseEnv(
        train_ds,
        reward_scale=args.reward_scale,
        seed=args.seed,
        max_steps=args.max_steps,
        stoi_weight=args.stoi_weight,
        estoi_weight=args.estoi_weight,
        diversity_weight=args.diversity_weight,
        quality_interval=args.quality_interval,
    )
    val_env = DenoiseEnv(
        val_ds,
        reward_scale=args.reward_scale,
        seed=args.seed + 1000,
        max_steps=args.max_steps,
        stoi_weight=args.stoi_weight,
        estoi_weight=args.estoi_weight,
        diversity_weight=args.diversity_weight,
        quality_interval=args.quality_interval,
    )
    print(f"[rl] samples={dataset.num_samples}, feature_dim={len(feature_cols)}")
    return train_env, val_env, len(feature_cols)


def evaluate_policy(env: DenoiseEnv, agent, episodes: int) -> Dict[str, float]:
    stats = {"reward": [], "si_sdr": [], "si_sdr_gain": [], "stoi": [], "estoi": []}
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        last_info: Dict[str, float] = {}
        while not done:
            if isinstance(agent, PPOAgent):
                action = agent.act_eval(obs)
            elif isinstance(agent, SACAgent):
                action, _ = agent.act(obs, deterministic=True)
            else:
                action, _ = agent.act(obs, deterministic=True, noise_std=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            last_info = info
            done = bool(terminated or truncated)
        stats["reward"].append(ep_reward)
        stats["si_sdr"].append(last_info.get("si_sdr", 0.0))
        stats["si_sdr_gain"].append(last_info.get("gain", 0.0))
        stats["stoi"].append(last_info.get("stoi", 0.0))
        stats["estoi"].append(last_info.get("estoi", 0.0))
    return {k: float(np.mean(v)) for k, v in stats.items()}


def maybe_log(wandb_run, data: Dict[str, float], step: int) -> None:
    if wandb_run is not None:
        wandb_run.log(data, step=step)


def update_reward_components(
    acc: Dict[str, float], info: Dict[str, float], args
) -> None:
    scale = args.reward_scale if args.reward_scale else 1.0
    acc["gain"] += info.get("gain", 0.0) / scale
    acc["stoi"] += info.get("stoi_gain", 0.0) * args.stoi_weight
    acc["estoi"] += info.get("estoi_gain", 0.0) * args.estoi_weight
    acc["diversity"] += info.get("diversity_bonus", 0.0)


def reward_component_summary(acc: Dict[str, float], count: int) -> Dict[str, float]:
    denom = max(1, count)
    return {key: value / denom for key, value in acc.items()}


def train_with_ppo(train_env: DenoiseEnv, val_env: DenoiseEnv, args, wandb_run) -> None:
    device = torch.device(args.device)
    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    agent = PPOAgent(
        obs_dim,
        act_dim,
        train_env.action_space.low,
        train_env.action_space.high,
        device,
        hidden_dims=tuple(args.hidden_dims),
        lr=args.actor_lr,
        max_grad_norm=args.max_grad_norm,
    )
    buffer = RolloutBuffer(args.rollout_steps, obs_dim, act_dim, device)
    obs, _ = train_env.reset()
    global_step = 0
    train_reward_sum = 0.0
    train_gain_sum = 0.0
    train_count = 0
    next_eval = args.eval_interval
    reward_components = {"gain": 0.0, "stoi": 0.0, "estoi": 0.0, "diversity": 0.0}
    reward_components = {"gain": 0.0, "stoi": 0.0, "estoi": 0.0, "diversity": 0.0}
    reward_components = {"gain": 0.0, "stoi": 0.0, "estoi": 0.0, "diversity": 0.0}

    while global_step < args.total_steps:
        buffer.reset()
        while len(buffer) < args.rollout_steps and global_step < args.total_steps:
            action_env, action_tanh, log_prob, value = agent.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = train_env.step(action_env)
            done = bool(terminated or truncated)
            buffer.add(obs, action_tanh, log_prob, value, reward, done)
            obs = next_obs
            train_reward_sum += reward
            train_gain_sum += info.get("gain", 0.0)
            train_count += 1
            update_reward_components(reward_components, info, args)
            global_step += 1
            if done:
                obs, _ = train_env.reset()

        last_value = agent.evaluate_value(obs)
        rollout = buffer.compute(last_value, args.gamma, args.gae_lambda)
        losses = agent.update(
            rollout,
            clip_range=args.clip_range,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            epochs=args.update_epochs,
            batch_size=args.batch_size,
        )

        comp_avg = reward_component_summary(reward_components, train_count)
        train_log = {
            "train/reward": train_reward_sum / max(1, train_count),
            "train/si_sdr_gain": train_gain_sum / max(1, train_count),
            "train/reward_gain": comp_avg["gain"],
            "train/reward_stoi": comp_avg["stoi"],
            "train/reward_estoi": comp_avg["estoi"],
            "train/reward_diversity": comp_avg["diversity"],
            "loss/policy": losses["policy_loss"],
            "loss/value": losses["value_loss"],
            "stats/entropy": losses["entropy"],
            "stats/approx_kl": losses["approx_kl"],
        }
        maybe_log(wandb_run, train_log, global_step)
        print(
            f"[train/PPO] reward components (avg last {max(1, train_count)} steps): "
            f"gain={comp_avg['gain']:.4f}, stoi={comp_avg['stoi']:.4f}, "
            f"estoi={comp_avg['estoi']:.4f}, diversity={comp_avg['diversity']:.4f}"
        )
        train_reward_sum = 0.0
        train_gain_sum = 0.0
        train_count = 0
        reward_components = {k: 0.0 for k in reward_components}

        if global_step >= next_eval:
            eval_stats = evaluate_policy(val_env, agent, args.eval_episodes)
            maybe_log(
                wandb_run,
                {
                    "eval/reward": eval_stats["reward"],
                    "eval/si_sdr": eval_stats["si_sdr"],
                    "eval/si_sdr_gain": eval_stats["si_sdr_gain"],
                    "eval/stoi": eval_stats["stoi"],
                    "eval/estoi": eval_stats["estoi"],
                },
                global_step,
            )
            next_eval += args.eval_interval

    return agent


def train_with_sac(train_env: DenoiseEnv, val_env: DenoiseEnv, args, wandb_run) -> None:
    device = torch.device(args.device)
    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    agent = SACAgent(
        obs_dim,
        act_dim,
        train_env.action_space.low,
        train_env.action_space.high,
        device,
        hidden_dims=tuple(args.hidden_dims),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        tau=args.tau,
        target_entropy_scale=args.target_entropy_scale,
        replay_size=args.replay_size,
    )
    obs, _ = train_env.reset()
    global_step = 0
    train_reward_sum = 0.0
    train_gain_sum = 0.0
    train_count = 0
    next_eval = args.eval_interval

    while global_step < args.total_steps:
        if global_step < args.random_steps:
            action_env = train_env.action_space.sample()
            action_tanh = agent.scaler.env_to_tanh(action_env)
        else:
            action_env, action_tanh = agent.act(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = train_env.step(action_env)
        done = bool(terminated or truncated)
        agent.replay.add(obs, action_tanh, reward, next_obs, float(done))
        obs = next_obs
        train_reward_sum += reward
        train_gain_sum += info.get("gain", 0.0)
        train_count += 1
        update_reward_components(reward_components, info, args)
        update_reward_components(reward_components, info, args)
        global_step += 1
        if done:
            obs, _ = train_env.reset()

        if len(agent.replay) >= args.batch_size and global_step > args.warmup_steps:
            for _ in range(args.updates_per_step):
                losses = agent.update(args.batch_size, args.gamma)
                maybe_log(
                    wandb_run,
                    {
                        "loss/critic": losses["critic_loss"],
                        "loss/actor": losses["actor_loss"],
                        "loss/alpha": losses["alpha_loss"],
                        "stats/alpha": losses["alpha"],
                    },
                    global_step,
                )

        if global_step % args.log_interval == 0:
            comp_avg = reward_component_summary(reward_components, train_count)
            train_log = {
                "train/reward": train_reward_sum / max(1, train_count),
                "train/si_sdr_gain": train_gain_sum / max(1, train_count),
                "train/reward_gain": comp_avg["gain"],
                "train/reward_stoi": comp_avg["stoi"],
                "train/reward_estoi": comp_avg["estoi"],
                "train/reward_diversity": comp_avg["diversity"],
            }
            maybe_log(wandb_run, train_log, global_step)
            print(
                f"[train/SAC] reward components (avg last {max(1, train_count)} steps): "
                f"gain={comp_avg['gain']:.4f}, stoi={comp_avg['stoi']:.4f}, "
                f"estoi={comp_avg['estoi']:.4f}, diversity={comp_avg['diversity']:.4f}"
            )
            train_reward_sum = 0.0
            train_gain_sum = 0.0
            train_count = 0
            reward_components = {k: 0.0 for k in reward_components}

        if global_step >= next_eval:
            eval_stats = evaluate_policy(val_env, agent, args.eval_episodes)
            maybe_log(
                wandb_run,
                {
                    "eval/reward": eval_stats["reward"],
                    "eval/si_sdr": eval_stats["si_sdr"],
                    "eval/si_sdr_gain": eval_stats["si_sdr_gain"],
                    "eval/stoi": eval_stats["stoi"],
                    "eval/estoi": eval_stats["estoi"],
                },
                global_step,
            )
            next_eval += args.eval_interval

    return agent


def train_with_td3(train_env: DenoiseEnv, val_env: DenoiseEnv, args, wandb_run) -> None:
    device = torch.device(args.device)
    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    agent = TD3Agent(
        obs_dim,
        act_dim,
        train_env.action_space.low,
        train_env.action_space.high,
        device,
        hidden_dims=tuple(args.hidden_dims),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        replay_size=args.replay_size,
    )
    obs, _ = train_env.reset()
    global_step = 0
    train_reward_sum = 0.0
    train_gain_sum = 0.0
    train_count = 0
    next_eval = args.eval_interval

    while global_step < args.total_steps:
        if global_step < args.random_steps:
            action_env = train_env.action_space.sample()
            action_tanh = agent.scaler.env_to_tanh(action_env)
        else:
            action_env, action_tanh = agent.act(obs, deterministic=False, noise_std=args.exploration_noise)
        next_obs, reward, terminated, truncated, info = train_env.step(action_env)
        done = bool(terminated or truncated)
        agent.replay.add(obs, action_tanh, reward, next_obs, float(done))
        obs = next_obs
        train_reward_sum += reward
        train_gain_sum += info.get("gain", 0.0)
        train_count += 1
        global_step += 1
        if done:
            obs, _ = train_env.reset()

        if len(agent.replay) >= args.batch_size and global_step > args.warmup_steps:
            for _ in range(args.updates_per_step):
                losses = agent.update(args.batch_size, args.gamma)
                maybe_log(wandb_run, losses, global_step)

        if global_step % args.log_interval == 0:
            comp_avg = reward_component_summary(reward_components, train_count)
            train_log = {
                "train/reward": train_reward_sum / max(1, train_count),
                "train/si_sdr_gain": train_gain_sum / max(1, train_count),
                "train/reward_gain": comp_avg["gain"],
                "train/reward_stoi": comp_avg["stoi"],
                "train/reward_estoi": comp_avg["estoi"],
                "train/reward_diversity": comp_avg["diversity"],
            }
            maybe_log(wandb_run, train_log, global_step)
            print(
                f"[train/TD3] reward components (avg last {max(1, train_count)} steps): "
                f"gain={comp_avg['gain']:.4f}, stoi={comp_avg['stoi']:.4f}, "
                f"estoi={comp_avg['estoi']:.4f}, diversity={comp_avg['diversity']:.4f}"
            )
            train_reward_sum = 0.0
            train_gain_sum = 0.0
            train_count = 0
            reward_components = {k: 0.0 for k in reward_components}

        if global_step >= next_eval:
            eval_stats = evaluate_policy(val_env, agent, args.eval_episodes)
            maybe_log(
                wandb_run,
                {
                    "eval/reward": eval_stats["reward"],
                    "eval/si_sdr": eval_stats["si_sdr"],
                    "eval/si_sdr_gain": eval_stats["si_sdr_gain"],
                    "eval/stoi": eval_stats["stoi"],
                    "eval/estoi": eval_stats["estoi"],
                },
                global_step,
            )
            next_eval += args.eval_interval

    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom RL trainer for denoising.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("train_data/train_state_features.csv"),
        help="Path to feature CSV",
    )
    parser.add_argument("--algo", choices=("ppo", "sac", "td3"), default="ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--reward-scale", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--stoi-weight", type=float, default=0.0)
    parser.add_argument("--estoi-weight", type=float, default=0.0)
    parser.add_argument("--diversity-weight", type=float, default=0.0)
    parser.add_argument("--quality-interval", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/rl_train"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--log-interval", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--replay-size", type=int, default=200_000)
    parser.add_argument("--random-steps", type=int, default=2_000)
    parser.add_argument("--warmup-steps", type=int, default=4_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--target-entropy-scale", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    args = parser.parse_args()
    args.log_dir = Path(args.log_dir)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_env, val_env, feature_dim = prepare_envs(args)

    wandb_run = None
    if args.wandb_project:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args) | {"feature_dim": feature_dim},
            save_code=True,
        )
        log_hparams(wandb_run, args)

    start = time.time()
    if args.algo == "ppo":
        agent = train_with_ppo(train_env, val_env, args, wandb_run)
    elif args.algo == "sac":
        agent = train_with_sac(train_env, val_env, args, wandb_run)
    else:
        agent = train_with_td3(train_env, val_env, args, wandb_run)
    elapsed = time.time() - start
    print(f"[rl] Training finished in {elapsed/3600:.2f} h using {args.algo.upper()}.")

    final_stats = evaluate_policy(val_env, agent, args.eval_episodes)
    if wandb_run is not None:
        wandb_run.log({f"final/{k}": v for k, v in final_stats.items()}, step=args.total_steps)
    checkpoint_path = save_checkpoint(args.algo, agent, args.log_dir, args, final_stats)
    print(f"[rl] Checkpoint saved to {checkpoint_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

