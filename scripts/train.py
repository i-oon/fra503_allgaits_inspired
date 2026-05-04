"""Train an AllGaits policy with RSL-RL PPO.

Phase A (default): single-gait trot. Run from repo root:

    python scripts/train.py --task Isaac-AllGaits-B1-Trot-v0 \
        --num_envs 4096 --headless --max_iterations 3000

Outputs (logs + checkpoints): logs/rsl_rl/allgaits_b1/<timestamp>/
View training curves:

    tensorboard --logdir logs/rsl_rl/allgaits_b1
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI (must run before AppLauncher so --headless etc. work)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train AllGaits on B1 with RSL-RL PPO.")
parser.add_argument("--task", type=str, default="Isaac-AllGaits-B1-Trot-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", action="store_true",
                    help="Resume from the last checkpoint under logs/.")
parser.add_argument("--load_run", type=str, default=".*",
                    help="Run name (regex) to resume from. Default: most recent.")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt",
                    help="Checkpoint name (regex) to load. Default: latest.")
parser.add_argument("--experiment_name", type=str, default=None,
                    help="Override experiment name; default from runner cfg.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports that require Isaac Sim to be running
# ---------------------------------------------------------------------------
import gymnasium as gym
import torch  # noqa: F401 — needed for CUDA init through isaaclab

import allgaits.tasks  # noqa: F401 — triggers gym.register
from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def main() -> None:
    # Load config from registration
    env_cfg_fn = gym.spec(args.task).kwargs["env_cfg_entry_point"]
    env_cfg: DirectRLEnvCfg = env_cfg_fn() if callable(env_cfg_fn) else env_cfg_fn
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    runner_cfg_cls = gym.spec(args.task).kwargs["rsl_rl_cfg_entry_point"]
    runner_cfg: RslRlOnPolicyRunnerCfg = runner_cfg_cls()
    runner_cfg.max_iterations = args.max_iterations
    runner_cfg.seed = args.seed
    if args.experiment_name is not None:
        runner_cfg.experiment_name = args.experiment_name

    # --- Log directory ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", runner_cfg.experiment_name))
    log_dir = os.path.join(log_root, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"\n[train] logs: {log_dir}\n[train] task: {args.task}  num_envs: {env_cfg.scene.num_envs}  iters: {runner_cfg.max_iterations}\n")

    # --- Instantiate env ---
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    # --- Instantiate runner ---
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device=env.unwrapped.device)
    runner.add_git_repo_to_log(__file__)

    # --- Optional resume ---
    if args.resume:
        from isaaclab.utils.dict import print_dict
        from isaaclab.utils.io import load_yaml  # noqa: F401 — kept for parity w/ isaac-lab scripts
        print(f"[train] resume=True, load_run={args.load_run}, checkpoint={args.checkpoint}")
        resume_path = _find_latest_checkpoint(log_root, args.load_run, args.checkpoint)
        if resume_path is None:
            print(f"[train] no checkpoint found under {log_root}; starting fresh.")
        else:
            print(f"[train] loading {resume_path}")
            runner.load(resume_path)

    # --- Train ---
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    # Save a stable alias so play.py can use --model final_model
    import shutil
    last_ckpt = os.path.join(log_dir, f"model_{runner_cfg.max_iterations - 1}.pt")
    final_path = os.path.join(log_dir, "final_model.pt")
    if os.path.isfile(last_ckpt):
        shutil.copy2(last_ckpt, final_path)
        print(f"[train] saved final_model.pt → {final_path}")

    env.close()
    sim_app.close()


def _find_latest_checkpoint(log_root: str, run_regex: str, ckpt_regex: str) -> str | None:
    import re
    if not os.path.isdir(log_root):
        return None
    runs = sorted(
        d for d in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, d)) and re.match(run_regex, d)
    )
    if not runs:
        return None
    run_dir = os.path.join(log_root, runs[-1])
    ckpts = sorted(
        f for f in os.listdir(run_dir)
        if re.match(ckpt_regex, f)
    )
    return os.path.join(run_dir, ckpts[-1]) if ckpts else None


if __name__ == "__main__":
    main()
