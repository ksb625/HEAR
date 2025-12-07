"""Utility to launch multiple RL sweeps sequentially."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_TOTAL_STEPS = 50_000
TAG_KEYS = [
    "diversity_weight",
    "reward_scale",
    "max_steps",
    "quality_interval",
    "stoi_weight",
    "estoi_weight",
    "target_entropy_scale",
    "exploration_noise",
    "actor_lr",
    "critic_lr",
]


@dataclass
class Experiment:
    """Definition of a sweep for a single algorithm."""

    name: str
    algo: str
    base_args: Dict[str, Any]
    sweep_params: Dict[str, List[Any]] = field(default_factory=dict)


def to_cli_str(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def format_tag_value(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) >= 1:
            text = f"{value:.1f}"
        else:
            text = f"{value:.2f}"
        text = text.rstrip("0").rstrip(".")
        text = text if text else "0"
        return text.replace("-", "m").replace(".", "p")
    if isinstance(value, int):
        return str(value)
    return str(value).replace("-", "m").replace(".", "p")


def iter_configs(exp: Experiment) -> Iterable[Dict[str, Any]]:
    if not exp.sweep_params:
        yield dict(exp.base_args)
        return
    keys = list(exp.sweep_params.keys())
    values = [exp.sweep_params[k] for k in keys]
    for combo in itertools.product(*values):
        cfg = dict(exp.base_args)
        cfg.update({k: v for k, v in zip(keys, combo)})
        yield cfg


def build_command(
    args: argparse.Namespace,
    exp: Experiment,
    cfg: Dict[str, Any],
    run_suffix: str,
) -> List[str]:
    cfg_local = dict(cfg)
    total_steps = cfg_local.pop("total_steps", DEFAULT_TOTAL_STEPS)
    final = {
        "algo": exp.algo,
        "features": args.features,
        "device": args.device,
        "total_steps": total_steps,
        "wandb_project": args.wandb_project,
        **cfg_local,
    }
    tag_parts = []
    for key in TAG_KEYS:
        if key in final:
            tag_parts.append(f"{key[:3]}{format_tag_value(final[key])}")
    tag_str = "-".join(tag_parts[:4])
    if tag_str:
        base_name = f"{exp.name}_{tag_str}"
    else:
        base_name = exp.name
    if args.run_name_prefix:
        run_name = f"{args.run_name_prefix}_{base_name}_{run_suffix}"
    else:
        run_name = f"{base_name}_{run_suffix}"
    final.setdefault("run_name", run_name)
    log_dir = Path(args.output_root) / exp.name / run_suffix
    final.setdefault("log_dir", str(log_dir))
    cmd = ["python", "-m", "rl.train_agent"]
    for k, v in final.items():
        if v is None:
            continue
        cli_key = f"--{k.replace('_', '-')}"
        cmd.extend([cli_key, to_cli_str(v)])
    return cmd


def run_experiments(args: argparse.Namespace) -> None:
    experiments = [
        Experiment(
            name="ppo_env_diversity",
            algo="ppo",
            base_args={
                "total_steps": DEFAULT_TOTAL_STEPS,
                "reward_scale": 5.0,
                "max_steps": 3,
                "quality_interval": 0,
                "stoi_weight": 0.0,
                "estoi_weight": 0.0,
            },
            sweep_params={
                "diversity_weight": [0.0, 0.1, 0.2],
                "reward_scale": [4.0, 5.0, 6.0],
            },
        ),
        Experiment(
            name="ppo_env_steps",
            algo="ppo",
            base_args={
                "total_steps": DEFAULT_TOTAL_STEPS,
                "reward_scale": 5.0,
                "diversity_weight": 0.1,
                "stoi_weight": 0.0,
                "estoi_weight": 0.0,
            },
            sweep_params={
                "max_steps": [2, 3, 4],
                "quality_interval": [0, 2, 4],
            },
        ),
        Experiment(
            name="ppo_env_intelligibility",
            algo="ppo",
            base_args={
                "total_steps": DEFAULT_TOTAL_STEPS,
                "reward_scale": 5.0,
                "max_steps": 3,
                "quality_interval": 2,
                "diversity_weight": 0.1,
            },
            sweep_params={
                "stoi_weight": [0.0, 0.3],
                "estoi_weight": [0.0, 0.3],
            },
        ),
        Experiment(
            name="ppo_env_quality_interval",
            algo="ppo",
            base_args={
                "total_steps": DEFAULT_TOTAL_STEPS,
                "reward_scale": 5.0,
                "max_steps": 3,
                "diversity_weight": 0.1,
                "stoi_weight": 0.3,
                "estoi_weight": 0.0,
            },
            sweep_params={
                "quality_interval": [0, 2, 4],
                "reward_scale": [4.0, 6.0],
            },
        ),
        # Algorithm comparison stage (environment settings fixed from PPO sweeps)
        Experiment(
            name="sac_algo_compare",
            algo="sac",
            base_args={
                "total_steps": DEFAULT_TOTAL_STEPS,
                "random_steps": 2_000,
                "warmup_steps": 4_000,
                "updates_per_step": 1,
                "reward_scale": 5.0,
                "max_steps": 3,
                "diversity_weight": 0.1,
                "quality_interval": 2,
                "stoi_weight": 0.3,
                "estoi_weight": 0.0,
            },
            sweep_params={
                "target_entropy_scale": [0.8, 1.0],
                "stoi_weight": [0.0, 0.3],
            },
        ),
        Experiment(
            name="td3_algo_compare",
            algo="td3",
            base_args={
                "total_steps": DEFAULT_TOTAL_STEPS,
                "quality_interval": 2,
                "reward_scale": 5.0,
                "diversity_weight": 0.1,
                "max_steps": 3,
                "stoi_weight": 0.3,
                "estoi_weight": 0.0,
            },
            sweep_params={
                "exploration_noise": [0.1, 0.2],
                "actor_lr": [3e-4, 1e-4],
            },
        ),
    ]

    enabled = {name.strip() for name in args.only.split(",")} if args.only else None
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for exp in experiments:
        if enabled and exp.name not in enabled:
            continue
        for idx, cfg in enumerate(iter_configs(exp)):
            run_suffix = f"{idx:02d}_{int(time.time())}"
            cmd = build_command(args, exp, cfg, run_suffix)
            header = f"[EXP] {exp.name} cfg={cfg}"
            print(header)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(header + "\n")
                fh.write("CMD: " + " ".join(cmd) + "\n")
            subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential experiment runner.")
    parser.add_argument(
        "--features",
        type=str,
        default="train_data/train_state_features.csv",
        help="Feature CSV path passed to rl.train_agent",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--wandb-project", type=str, default="rl-denoise")
    parser.add_argument(
        "--run-name-prefix",
        type=str,
        default="sweep",
        help="Prefix appended to each run_name",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated experiment names to run (leave empty for all).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="runs/experiment_runner.log",
        help="Where to append executed commands.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="runs/experiments",
        help="Base directory for log_dir/model checkpoints.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(args)

