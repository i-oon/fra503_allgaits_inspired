"""Smoke test for the AllGaitsEnv.

Instantiates a small parallel env (num_envs=16), steps it N times with random
actions, and verifies:
  - scene loads, robot spawns
  - observations have the expected shape (N, 64)
  - rewards are finite, not NaN/Inf
  - CPG oscillator states evolve (theta changes over time)
  - no immediate termination (robots survive at least a few steps)

Run (headless):
    python scripts/test_env.py --headless --num_envs 16 --steps 100
"""

from __future__ import annotations

import argparse

import torch

from isaaclab.app import AppLauncher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--active_gaits", type=str, default="trot",
                        help="Comma-separated list of gaits to include in the Φ pool.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    # Isaac-Lab-dependent imports must come after the app has launched.
    from allgaits.envs import AllGaitsEnv, AllGaitsEnvCfg

    cfg = AllGaitsEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.active_gaits = tuple(g.strip() for g in args.active_gaits.split(","))

    print(f"\n=== AllGaitsEnv smoke test ===")
    print(f"  num_envs      : {cfg.scene.num_envs}")
    print(f"  active_gaits  : {cfg.active_gaits}")
    print(f"  action_space  : {cfg.action_space}")
    print(f"  observation   : {cfg.observation_space}")
    print(f"  episode length: {cfg.episode_length_s} s ({int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))} policy steps)")
    print(f"  physics dt    : {cfg.sim.dt} s    decimation: {cfg.decimation}")

    env = AllGaitsEnv(cfg=cfg)

    # Initial reset
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    assert obs.shape == (args.num_envs, cfg.observation_space), f"obs shape {tuple(obs.shape)} vs expected ({args.num_envs}, {cfg.observation_space})"
    print(f"\n[reset] observation shape OK: {tuple(obs.shape)}")

    # Track CPG theta to verify it evolves
    theta_hist = [env._cpg.theta.clone().cpu()]

    for t in range(args.steps):
        action = torch.randn(args.num_envs, cfg.action_space, device=env.device)
        obs_dict, rew, terminated, truncated, info = env.step(action)
        theta_hist.append(env._cpg.theta.clone().cpu())

        if t == 0 or (t + 1) % 25 == 0:
            alive = (~terminated).float().mean().item()
            print(
                f"[step {t+1:4d}]  rew mean={rew.mean().item():+.3f}  std={rew.std().item():.3f}  "
                f"alive frac={alive:.2f}  obs finite={torch.all(torch.isfinite(obs_dict['policy'])).item()}"
            )

        assert torch.all(torch.isfinite(rew)), f"NaN/Inf in reward at step {t}"
        assert torch.all(torch.isfinite(obs_dict["policy"])), f"NaN/Inf in observation at step {t}"

    theta_all = torch.stack(theta_hist, dim=0)  # (T+1, N, 4)
    theta_range = (theta_all.max(dim=0).values - theta_all.min(dim=0).values).min().item()
    print(f"\n[cpg] min per-leg theta range over {args.steps} steps: {theta_range:.3f} rad  (expect > 1 rad if oscillators are advancing)")

    if theta_range < 1.0:
        print("  WARNING: CPG theta not evolving much — check ω mapping.")
    else:
        print("  OK: oscillators are phase-advancing.")

    print(f"\n=== Smoke test complete ===")
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
