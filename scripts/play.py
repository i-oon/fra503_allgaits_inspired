"""Evaluate a trained AllGaits policy with per-env diagnostics.

Differences from training:
  - Velocity command is held fixed (disables the 5-s resampling).
  - Style parameters (h, g_c, x_off, g_p) are either frozen to user-specified
    values or held at whatever the initial reset sampled — no mid-episode
    re-randomization.
  - Per-env diagnostic table printed every `--log_every` policy steps so
    you can see WHICH envs walk, at what speed, under which style params,
    and with what gait stability.

Usage:

    # Default: 4 envs, 2000 steps (~20 s), per-env table every 100 steps.
    python scripts/play.py --task Isaac-AllGaits-B1-Trot-v0 --num_envs 4 --vel_x 0.8

    # Freeze style params so all envs share the same config (isolates policy behavior):
    python scripts/play.py --num_envs 4 --vel_x 0.8 \
        --fix_h 0.42 --fix_g_c 0.05 --fix_x_off -0.02

    # Headless benchmark run:
    python scripts/play.py --headless --num_envs 16 --episode_length 3000
"""

from __future__ import annotations

import argparse
import math
import os
from contextlib import contextmanager

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a trained AllGaits policy.")
parser.add_argument("--task", type=str, default="Isaac-AllGaits-B1-Trot-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--episode_length", type=int, default=2000,
                    help="Number of policy steps to run (100 Hz → 2000 ≈ 20 s).")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to model_*.pt; default = latest under logs/.")
parser.add_argument("--load_run", type=str, default=".*")
parser.add_argument("--model", type=str, default=None,
                    help="Checkpoint to load: step number (e.g. 4999 → model_4999.pt) or name (e.g. final_model).")
parser.add_argument("--vel_x", type=float, default=0.8,
                    help="Forward velocity command (m/s) — held fixed throughout play.")
parser.add_argument("--log_every", type=int, default=100,
                    help="Print per-env table every N policy steps.")

# Style-parameter overrides (freeze to known values across all envs)
parser.add_argument("--fix_h", type=float, default=None, help="Override body height (m).")
parser.add_argument("--fix_g_c", type=float, default=None, help="Override swing-phase ground clearance (m).")
parser.add_argument("--fix_g_p", type=float, default=None, help="Override stance-phase penetration (m).")
parser.add_argument("--fix_x_off", type=float, default=None, help="Override sagittal foot offset (m).")

# Gait override — force the coupling matrix Φ to a specific gait regardless
# of what the env's active_gaits pool would sample. Useful for: (a) verifying
# the trained policy can do specific gaits, (b) testing novel gaits the
# policy was never trained on (paper §III-D2 demo), (c) switching gaits
# mid-run (Phase B/C feature).
parser.add_argument("--gait", type=str, default=None,
                    choices=[None, "walk", "amble", "trot", "pace", "bound", "pronk",
                             "canter", "transverse_gallop", "rotary_gallop"],
                    help="Force coupling matrix to this gait (overrides task's active_gaits pool).")
parser.add_argument("--gait_sequence", type=str, default=None,
                    help="Timed gait transitions: 'gait:steps[:vel_x],...'. "
                         "E.g. 'trot:600:0.8,walk:600:0.4,pace:600:0.4'. "
                         "Switches Φ (and optionally vel_x) at each step boundary. "
                         "Tip: set --episode_length to sum of all step counts.")
parser.add_argument("--metrics_out", type=str, default="play_metrics_log.csv",
                    help="Output CSV path for the rich metrics log (contacts, CPG state, yaw).")
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed for style-param sampling. Fixes the per-env param draw "
                         "so results are reproducible. Find a good seed once, then reuse it.")

# Policy sampling mode
parser.add_argument("--stochastic", action="store_true",
                    help="Sample actions from the policy distribution instead of "
                         "taking the deterministic mean. Useful when the mean "
                         "collapses to a non-moving action but training-time "
                         "samples produce gait.")

# CPG-bypass diagnostic
parser.add_argument("--bypass_policy", action="store_true",
                    help="Ignore the policy entirely; command constant (μ, ω_Hz) "
                         "from --bypass_mu / --bypass_omega. Useful to verify "
                         "the CPG + pattern + IK chain works mechanically.")
parser.add_argument("--bypass_mu", type=float, default=1.5,
                    help="Constant μ when --bypass_policy. Default 1.5 (mid-range).")
parser.add_argument("--bypass_omega_hz", type=float, default=2.0,
                    help="Constant ω (Hz) when --bypass_policy. Default 2 Hz.")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports that require Isaac Sim running
# ---------------------------------------------------------------------------
import gymnasium as gym
import torch

import allgaits.tasks  # noqa: F401 — triggers gym.register
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@contextmanager
def _inference():
    with torch.inference_mode():
        yield


def _find_latest_checkpoint(exp_name: str, run_name: str) -> str | None:
    import re
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", exp_name))
    if not os.path.isdir(log_root):
        return None
    # Exact match first (handles names with regex special chars like dates).
    if os.path.isdir(os.path.join(log_root, run_name)):
        run_dir = os.path.join(log_root, run_name)
    else:
        # Fall back to regex — sort by mtime so runs[-1] is most recent.
        runs = sorted(
            (d for d in os.listdir(log_root)
             if os.path.isdir(os.path.join(log_root, d)) and re.match(run_name, d)),
            key=lambda d: os.path.getmtime(os.path.join(log_root, d)),
        )
        if not runs:
            return None
        run_dir = os.path.join(log_root, runs[-1])
    # Sort checkpoints numerically (not as strings) so model_5999.pt beats model_950.pt.
    ckpts = sorted(
        (f for f in os.listdir(run_dir) if re.match(r"model_\d+\.pt$", f)),
        key=lambda f: int(re.search(r"(\d+)", f).group(1)),
    )
    return os.path.join(run_dir, ckpts[-1]) if ckpts else None


def _freeze_style_params(env_unwrapped, h=None, g_c=None, g_p=None, x_off=None) -> dict[str, float]:
    """Override style params on all envs; return the actually-applied values."""
    applied = {}
    if h is not None:
        env_unwrapped._h_per_env[:, 0] = h
        applied["h"] = h
    if g_c is not None:
        env_unwrapped._g_c_per_env[:, 0] = g_c
        applied["g_c"] = g_c
    if g_p is not None:
        env_unwrapped._g_p_per_env[:, 0] = g_p
        applied["g_p"] = g_p
    if x_off is not None:
        env_unwrapped._x_off_per_env[:, 0] = x_off
        applied["x_off"] = x_off
    return applied


def _disable_resampling(env_unwrapped) -> None:
    """Prevent the env's periodic v* / Φ resampling during play.

    Pushes the counters far below their thresholds so they never fire.
    """
    env_unwrapped._steps_since_vel_resample[:] = -(10 ** 9)
    env_unwrapped._steps_since_phi_resample[:] = -(10 ** 9)


def _freeze_gait(env_unwrapped, gait_name: str | None) -> None:
    """Force the coupling matrix Φ to a specific gait on all envs.

    Overwrites whatever the env sampled from `cfg.active_gaits`. Must be
    called every step because `_reset_idx` re-samples Φ for terminated envs.
    """
    if gait_name is None:
        return
    from allgaits.cpg.coupling import phase_offset_matrix
    phi = phase_offset_matrix(gait_name, device=env_unwrapped.device)
    env_unwrapped._phi[:] = phi.unsqueeze(0)


def _parse_gait_sequence(spec: str, default_vel: float) -> list[tuple[str, int, float]]:
    """Parse 'gait:steps[:vel_x],...' into [(gait, steps, vel_x), ...] tuples."""
    result = []
    for seg in spec.split(","):
        parts = seg.strip().split(":")
        if len(parts) < 2:
            raise ValueError(f"Bad gait_sequence segment {seg!r}; expected gait:steps[:vel_x]")
        result.append((parts[0], int(parts[1]), float(parts[2]) if len(parts) >= 3 else default_vel))
    return result


def _print_header(cfg, applied_style: dict[str, float], vel_x_cmd: float) -> None:
    print("\n" + "=" * 100)
    print("AllGaits Play — Per-Env Diagnostics")
    print("=" * 100)
    print(f"  task            {args.task}")
    print(f"  num_envs        {args.num_envs}")
    print(f"  episode_length  {args.episode_length} steps ({args.episode_length * cfg.sim.dt * cfg.decimation:.1f} s)")
    print(f"  vel_x_cmd       {vel_x_cmd:+.2f} m/s  (held fixed)")
    if applied_style:
        print(f"  style override  {applied_style}")
    else:
        print(f"  style override  (none — using per-env sampled values)")
    print(f"  active_gaits    {cfg.active_gaits}")
    print("=" * 100)


def _env_snapshot(env_unwrapped) -> dict[str, torch.Tensor]:
    """Grab per-env state vectors for the diagnostic table."""
    r = env_unwrapped._robot
    env_origins = env_unwrapped._terrain.env_origins  # per-env world origin
    # World-frame position RELATIVE to the env's spawn origin — this is the
    # actual drift of the robot in world xy, independent of which env it is.
    world_pos_rel = r.data.root_pos_w[:, :2] - env_origins[:, :2]
    # Body-frame forward axis in world: the body's +X unit vector, rotated by
    # the base orientation. A quick way via projected-gravity: if the robot
    # is upright, body +X projects onto world xy as a unit vector we can read.
    # Use the root quat directly: world_x_axis_of_body = R(quat) · [1,0,0]^T.
    quat = r.data.root_quat_w   # (N, 4) wxyz
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # First column of rotation matrix from quat = body's +X axis in world.
    body_x_world = torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y + z * w),
        2 * (x * z - y * w),
    ], dim=-1)
    return {
        "vel_x": r.data.root_lin_vel_b[:, 0].detach().cpu(),
        "vel_y": r.data.root_lin_vel_b[:, 1].detach().cpu(),
        "height": r.data.root_pos_w[:, 2].detach().cpu(),
        "tilt_xy": r.data.projected_gravity_b[:, :2].norm(dim=-1).detach().cpu(),
        "h": env_unwrapped._h_per_env[:, 0].detach().cpu(),
        "g_c": env_unwrapped._g_c_per_env[:, 0].detach().cpu(),
        "g_p": env_unwrapped._g_p_per_env[:, 0].detach().cpu(),
        "x_off": env_unwrapped._x_off_per_env[:, 0].detach().cpu(),
        "omega_hz": (env_unwrapped._omega / (2.0 * math.pi)).mean(dim=-1).detach().cpu(),
        "mu": env_unwrapped._mu.mean(dim=-1).detach().cpu(),
        "feet_down": env_unwrapped._foot_contact_booleans().sum(dim=-1).detach().cpu(),
        "steps_alive": env_unwrapped.episode_length_buf.detach().cpu(),
        # World-frame drift since spawn (x, y in env-local coords)
        "world_dx": world_pos_rel[:, 0].detach().cpu(),
        "world_dy": world_pos_rel[:, 1].detach().cpu(),
        # Body's forward axis projected on world xy (should be ≈ (+1, 0) at spawn)
        "body_x_world_x": body_x_world[:, 0].detach().cpu(),
        "body_x_world_y": body_x_world[:, 1].detach().cpu(),
    }


def _print_env_table(step: int, snap: dict[str, torch.Tensor], resets_this_window: list[list[int]]) -> None:
    num_envs = snap["vel_x"].numel()
    cols = [
        "env", "alive", "vx_body", "vy_body", "world_dx", "world_dy",
        "bodyX→w", "h_body", "tilt", "μ", "ω_Hz", "feet↓", "rst",
    ]
    widths = [3, 6, 9, 9, 10, 9, 11, 7, 6, 5, 6, 6, 4]
    print(f"\n--- step {step:5d} ---")
    hdr = "  ".join(f"{c:>{w}s}" for c, w in zip(cols, widths))
    print(hdr)
    print("-" * len(hdr))
    for i in range(num_envs):
        resets = len(resets_this_window[i]) if resets_this_window else 0
        bx = snap["body_x_world_x"][i].item()
        by = snap["body_x_world_y"][i].item()
        row = [
            f"{i:>3d}",
            f"{int(snap['steps_alive'][i].item()):>6d}",
            f"{snap['vel_x'][i].item():>+9.3f}",
            f"{snap['vel_y'][i].item():>+9.3f}",
            f"{snap['world_dx'][i].item():>+10.3f}",
            f"{snap['world_dy'][i].item():>+9.3f}",
            f"({bx:>+.2f},{by:>+.2f})",
            f"{snap['height'][i].item():>7.3f}",
            f"{snap['tilt_xy'][i].item():>6.3f}",
            f"{snap['mu'][i].item():>5.2f}",
            f"{snap['omega_hz'][i].item():>6.2f}",
            f"{int(snap['feet_down'][i].item()):>6d}",
            f"{resets:>4d}",
        ]
        print("  ".join(row))


def _print_foot_slip_table(
    step: int,
    contact: torch.Tensor,
    foot_slip_vx: torch.Tensor,
    leg_labels: list[str],
) -> None:
    """Per-env, per-foot contact state and world-frame x slip velocity.

    A foot with contact=1 but non-zero slip_vx is sliding against the ground
    instead of being held by static friction. |slip_vx| > 0.05 m/s is flagged.
    """
    N = contact.shape[0]
    ct = contact.detach().cpu()
    vx = foot_slip_vx.detach().cpu()
    print(f"\n--- foot slip @ step {step:5d} ---")
    hdr = f"  {'env':>3s}"
    for leg in leg_labels:
        hdr += f"  {leg+'_ct':>6s}  {leg+'_vx':>9s}"
    print(hdr)
    for i in range(N):
        row = f"  {i:>3d}"
        for j in range(4):
            c = int(ct[i, j].item())
            v = vx[i, j].item()
            flag = " *SLIP*" if c and abs(v) > 0.05 else ""
            row += f"  {c:>6d}  {v:>+9.3f}{flag}"
        print(row)


def _print_summary(per_env_stats: dict[str, list[torch.Tensor]], num_envs: int) -> None:
    """Average / max per env over the whole run."""
    print("\n" + "=" * 100)
    print("Summary (averaged over all policy steps)")
    print("=" * 100)
    cols = ["env", "mean_vx", "mean_h", "mean_tilt", "mean_ω_Hz", "mean_feet↓", "total_resets"]
    widths = [3, 9, 9, 9, 11, 11, 14]
    hdr = "  ".join(f"{c:>{w}s}" for c, w in zip(cols, widths))
    print(hdr)
    print("-" * len(hdr))

    vx = torch.stack(per_env_stats["vel_x"])       # (T, N)
    h = torch.stack(per_env_stats["height"])
    tilt = torch.stack(per_env_stats["tilt_xy"])
    om = torch.stack(per_env_stats["omega_hz"])
    feet = torch.stack(per_env_stats["feet_down"])
    resets = per_env_stats["resets"]                # list length N

    for i in range(num_envs):
        row = [
            f"{i:>3d}",
            f"{vx[:, i].mean().item():>+9.3f}",
            f"{h[:, i].mean().item():>9.3f}",
            f"{tilt[:, i].mean().item():>9.3f}",
            f"{om[:, i].mean().item():>11.2f}",
            f"{feet[:, i].float().mean().item():>11.2f}",
            f"{resets[i]:>14d}",
        ]
        print("  ".join(row))
    print("-" * len(hdr))
    total_vx = vx.mean().item()
    if args.gait_sequence:
        print(f"  Across all envs: mean vx = {total_vx:+.3f} m/s  (sequence: {args.gait_sequence})")
    else:
        print(f"  Across all envs: mean vx = {total_vx:+.3f} m/s  (cmd = {args.vel_x:+.2f} m/s,  error = {args.vel_x - total_vx:+.3f} m/s)")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if args.seed is not None:
        import random, numpy as np
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"[play] random seed: {args.seed}")

    env_cfg_fn = gym.spec(args.task).kwargs["env_cfg_entry_point"]
    env_cfg = env_cfg_fn() if callable(env_cfg_fn) else env_cfg_fn
    env_cfg.scene.num_envs = args.num_envs

    runner_cfg_cls = gym.spec(args.task).kwargs["rsl_rl_cfg_entry_point"]
    runner_cfg = runner_cfg_cls()

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if not args.headless else None)
    env = RslRlVecEnvWrapper(env)
    env_unwrapped = env.unwrapped

    # --- Load checkpoint ---
    if args.checkpoint:
        ckpt = args.checkpoint
    elif args.model is not None:
        base = _find_latest_checkpoint(runner_cfg.experiment_name, args.load_run)
        if base is None:
            ckpt = None
        else:
            run_dir = os.path.dirname(base)
            # Numeric → model_N.pt; named → <name>.pt (e.g. final_model.pt)
            fname = f"model_{args.model}.pt" if args.model.isdigit() else f"{args.model}.pt"
            ckpt = os.path.join(run_dir, fname)
    else:
        ckpt = _find_latest_checkpoint(runner_cfg.experiment_name, args.load_run)
    if ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint found under logs/rsl_rl/{runner_cfg.experiment_name}/ — train first."
        )
    print(f"\n[play] loading checkpoint: {ckpt}")

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=env_unwrapped.device)
    runner.load(ckpt)
    # Policy modes:
    #   default         deterministic mean       (policy.act_inference)
    #   --stochastic    sample from distribution (policy.act)
    #   --bypass_policy hand-crafted constant    (bypass_fn below)
    actor_critic = runner.alg.actor_critic
    if args.bypass_policy:
        # Build an 8-dim raw action that decodes to (bypass_mu, bypass_omega_hz)
        # via the env's linear-scale-and-clamp decoder:
        #   decoded_mu = 1.5 + 0.5 * raw_mu   → raw_mu = 2 * (bypass_mu - 1.5)
        #   decoded_omega_hz = 4 + 4 * raw_om → raw_om = (bypass_omega_hz - 4) / 4
        raw_mu = 2.0 * (args.bypass_mu - 1.5)
        raw_om = (args.bypass_omega_hz - 4.0) / 4.0
        constant_action = torch.tensor(
            [[raw_mu] * 4 + [raw_om] * 4], device=env_unwrapped.device
        ).expand(args.num_envs, -1).contiguous()
        def policy_fn(_obs):
            return constant_action
        print(f"[play] BYPASS policy: μ={args.bypass_mu} (raw {raw_mu:+.2f}), ω={args.bypass_omega_hz} Hz (raw {raw_om:+.2f})")
    elif args.stochastic:
        def policy_fn(obs):
            return actor_critic.act(obs)    # samples from Gaussian
        print(f"[play] stochastic sampling from policy distribution (noise_std = {actor_critic.std.mean().item():.3f})")
    else:
        def policy_fn(obs):
            return actor_critic.act_inference(obs)   # deterministic mean
        print(f"[play] deterministic policy mean (no exploration noise)")
    policy = policy_fn

    # --- Initial observation & one-time env overrides ---
    obs, _ = env.get_observations() if hasattr(env, "get_observations") else env.reset()

    # Freeze velocity command + disable periodic resampling.
    env_unwrapped._vel_cmd[:, 0] = args.vel_x
    env_unwrapped._vel_cmd[:, 1] = 0.0
    env_unwrapped._vel_cmd[:, 2] = 0.0
    _disable_resampling(env_unwrapped)

    # Optional style-param freeze.
    applied_style = _freeze_style_params(
        env_unwrapped, h=args.fix_h, g_c=args.fix_g_c, g_p=args.fix_g_p, x_off=args.fix_x_off
    )

    # Gait control: single --gait override or timed --gait_sequence transitions.
    _gait_seq: list[tuple[str, int, float]] | None = None
    _seq_seg_idx = -1
    _seq_gait: str | None = args.gait
    _seq_vel: float = args.vel_x

    if args.gait_sequence:
        _gait_seq = _parse_gait_sequence(args.gait_sequence, args.vel_x)
        _seq_seg_idx = 0
        _seq_gait, _, _seq_vel = _gait_seq[0]
        env_unwrapped._vel_cmd[:, 0] = _seq_vel
    _freeze_gait(env_unwrapped, _seq_gait)

    _print_header(env_unwrapped.cfg, applied_style, args.vel_x)
    if args.gait_sequence:
        total_seq_steps = sum(s for _, s, _ in _gait_seq)
        print(f"  gait_sequence   {args.gait_sequence}")
        print(f"  seq total steps {total_seq_steps}  (--episode_length should match)")
        print(f"[gait_sequence] step     0 → gait={_seq_gait}, vel_x={_seq_vel:.2f}")
        print("=" * 100)
    elif args.gait is not None:
        print(f"  gait override   {args.gait}  (overrides active_gaits pool)")
        print("=" * 100)

    # --- One-shot kinematics ground-truth print (after first reset) ---
    # Compares PhysX-reported foot-in-hip-frame against our FK prediction
    # from the same joint angles. If these disagree, the URDF convention
    # doesn't match our FK assumption and we need to adjust.
    r = env_unwrapped._robot
    hip_ids, hip_names = r.find_bodies([".*_hip$"])
    foot_ids, foot_names = r.find_bodies([".*_foot$"])
    hip_pos_w = r.data.body_pos_w[0, hip_ids]        # (4, 3) for env 0
    foot_pos_w = r.data.body_pos_w[0, foot_ids]
    # Physx-reported foot in hip frame = foot_world - hip_world (ignoring hip rotation here for simplicity)
    phys_foot_in_hip = foot_pos_w - hip_pos_w
    # Our FK prediction at current joint angles (env 0)
    q_all = r.data.joint_pos[0, env_unwrapped._cpg_to_usd_joint_idx].reshape(4, 3)
    from allgaits.kinematics.b1 import B1LegKinematics, LEG_DY
    d_y = LEG_DY(device=env_unwrapped.device)
    fk_pred = B1LegKinematics.forward(q_all[:, 0], q_all[:, 1], q_all[:, 2], d_y).detach().cpu()
    print("\n[PHYSX vs FK] foot-in-hip-frame at step 1 (env 0)")
    print(f"  {'leg':>3s}  {'hip_name':>12s}  {'foot_name':>12s}  "
          f"{'physx_x':>9s}  {'physx_y':>9s}  {'physx_z':>9s}   "
          f"{'fk_x':>9s}  {'fk_y':>9s}  {'fk_z':>9s}")
    for i, (h, f) in enumerate(zip(hip_names, foot_names)):
        px = phys_foot_in_hip[i].cpu()
        fk = fk_pred[i]
        print(f"  {i:>3d}  {h:>12s}  {f:>12s}  "
              f"{px[0].item():>+9.4f}  {px[1].item():>+9.4f}  {px[2].item():>+9.4f}   "
              f"{fk[0].item():>+9.4f}  {fk[1].item():>+9.4f}  {fk[2].item():>+9.4f}")
    print()

    # Foot articulation body indices (used below to read world-frame foot positions).
    _artic_foot_ids = torch.tensor(foot_ids, device=env_unwrapped.device, dtype=torch.long)
    # Leg labels from body names, e.g. "FL_foot" → "FL"
    _leg_labels: list[str] = [n.rsplit("_", 1)[0] for n in foot_names]
    _step_dt: float = env_unwrapped.cfg.sim.dt * env_unwrapped.cfg.decimation   # 0.01 s

    # --- Run loop with per-env tracking ---
    per_env_stats = {k: [] for k in ("vel_x", "height", "tilt_xy", "omega_hz", "feet_down")}
    reset_counts = [0] * args.num_envs
    prev_episode_buf = env_unwrapped.episode_length_buf.clone()
    resets_window: list[list[int]] = [[] for _ in range(args.num_envs)]
    _prev_foot_pos_w = env_unwrapped._robot.data.body_pos_w[:, _artic_foot_ids, :].clone()
    _slip_log: list[dict] = []
    _jnt_cols = [
        f"{leg}_{jt}" for leg in ("FL", "FR", "RL", "RR")
        for jt in ("hip", "thigh", "calf")
    ]
    _joint_log: list[dict] = []

    for step in range(args.episode_length):
        # Advance gait_sequence: detect segment boundary and switch Φ + vel_x.
        if _gait_seq is not None:
            acc = 0
            new_idx = len(_gait_seq) - 1
            for idx, (_, seg_steps, _) in enumerate(_gait_seq):
                if step < acc + seg_steps:
                    new_idx = idx
                    break
                acc += seg_steps
            if new_idx != _seq_seg_idx:
                _seq_seg_idx = new_idx
                _seq_gait, _, _seq_vel = _gait_seq[_seq_seg_idx]
                print(f"\n[gait_sequence] step {step:5d} → gait={_seq_gait}, vel_x={_seq_vel:.2f}")

        # Re-freeze every step in case env resets in _reset_idx (which re-samples
        # vel_cmd, style params, and Φ for the reset env_ids).
        env_unwrapped._vel_cmd[:, 0] = _seq_vel
        env_unwrapped._vel_cmd[:, 1] = 0.0
        env_unwrapped._vel_cmd[:, 2] = 0.0
        _freeze_style_params(
            env_unwrapped, h=args.fix_h, g_c=args.fix_g_c, g_p=args.fix_g_p, x_off=args.fix_x_off
        )
        _freeze_gait(env_unwrapped, _seq_gait)
        _disable_resampling(env_unwrapped)

        with _inference():
            action = policy(obs)
        obs, _rew, dones, _ext = env.step(action)

        # Track per-env stats
        snap = _env_snapshot(env_unwrapped)
        for k in per_env_stats:
            per_env_stats[k].append(snap[k])

        # Detect resets: episode_length_buf goes back to 0
        cur_buf = env_unwrapped.episode_length_buf
        reset_mask = cur_buf < prev_episode_buf
        if reset_mask.any():
            for i in reset_mask.nonzero(as_tuple=False).flatten().tolist():
                reset_counts[i] += 1
                resets_window[i].append(step)
        prev_episode_buf = cur_buf.clone()

        # --- Foot slip diagnostic ---
        _foot_pos_w = env_unwrapped._robot.data.body_pos_w[:, _artic_foot_ids, :]  # (N, 4, 3)
        _foot_vx = (_foot_pos_w[..., 0] - _prev_foot_pos_w[..., 0]) / _step_dt    # (N, 4)
        _contact_now = env_unwrapped._foot_contact_booleans()                       # (N, 4)
        _prev_foot_pos_w = _foot_pos_w.detach().clone()

        _jv = env_unwrapped._robot.data.joint_vel[:, env_unwrapped._cpg_to_usd_joint_idx].detach().cpu()  # (N, 12) CPG order
        _bv = env_unwrapped._robot.data.root_lin_vel_b.detach().cpu()   # (N, 3)
        _ct = _contact_now.detach().cpu()                                # (N, 4)
        _quat = env_unwrapped._robot.data.root_quat_w.detach().cpu()    # (N, 4) wxyz
        _cpg_r = env_unwrapped._cpg.r.detach().cpu()                    # (N, 4) amplitude per leg
        _cpg_theta = env_unwrapped._cpg.theta.detach().cpu()            # (N, 4) phase per leg, rad
        _yaw = torch.atan2(
            2 * (_quat[:, 0] * _quat[:, 3] + _quat[:, 1] * _quat[:, 2]),
            1 - 2 * (_quat[:, 2] ** 2 + _quat[:, 3] ** 2),
        )  # (N,)  rad
        for _ei in range(args.num_envs):
            for _li in range(4):
                _slip_log.append({
                    "step": step + 1,
                    "env": _ei,
                    "leg": _leg_labels[_li],
                    "contact": int(_ct[_ei, _li].item()),
                    "foot_world_x": _foot_pos_w[_ei, _li, 0].item(),
                    "foot_slip_vx": _foot_vx[_ei, _li].item(),
                })
            _joint_log.append({
                "step": step + 1,
                "env": _ei,
                "gait": _seq_gait if _seq_gait else "sampled",
                "vx_cmd": _seq_vel,
                "bvx": _bv[_ei, 0].item(),
                "bvy": _bv[_ei, 1].item(),
                "yaw_deg": math.degrees(_yaw[_ei].item()),
                # foot contacts
                "ct_FL": int(_ct[_ei, 0].item()),
                "ct_FR": int(_ct[_ei, 1].item()),
                "ct_RL": int(_ct[_ei, 2].item()),
                "ct_RR": int(_ct[_ei, 3].item()),
                # CPG amplitude r (limit-cycle radius per leg)
                "cpg_r_FL": _cpg_r[_ei, 0].item(),
                "cpg_r_FR": _cpg_r[_ei, 1].item(),
                "cpg_r_RL": _cpg_r[_ei, 2].item(),
                "cpg_r_RR": _cpg_r[_ei, 3].item(),
                # CPG phase θ in [0, 2π)
                "cpg_theta_FL": _cpg_theta[_ei, 0].item() % (2 * math.pi),
                "cpg_theta_FR": _cpg_theta[_ei, 1].item() % (2 * math.pi),
                "cpg_theta_RL": _cpg_theta[_ei, 2].item() % (2 * math.pi),
                "cpg_theta_RR": _cpg_theta[_ei, 3].item() % (2 * math.pi),
                **{col: _jv[_ei, j].item() for j, col in enumerate(_jnt_cols)},
            })

        if (step + 1) % args.log_every == 0 or step == 0:
            _print_env_table(step + 1, snap, resets_window)
            _print_foot_slip_table(step + 1, _contact_now, _foot_vx, _leg_labels)
            resets_window = [[] for _ in range(args.num_envs)]

    # Append reset_counts into per_env_stats for the summary
    per_env_stats["resets"] = reset_counts
    _print_summary(per_env_stats, args.num_envs)

    import csv
    _slip_path = "play_slip_log.csv"
    with open(_slip_path, "w", newline="") as _f:
        _w = csv.DictWriter(_f, fieldnames=["step", "env", "leg", "contact", "foot_world_x", "foot_slip_vx"])
        _w.writeheader()
        _w.writerows(_slip_log)
    print(f"\n[play] foot slip log → {_slip_path}  ({len(_slip_log)} rows)")

    _metrics_path = args.metrics_out
    _metrics_fields = (
        ["step", "env", "gait", "vx_cmd", "bvx", "bvy", "yaw_deg"]
        + ["ct_FL", "ct_FR", "ct_RL", "ct_RR"]
        + ["cpg_r_FL", "cpg_r_FR", "cpg_r_RL", "cpg_r_RR"]
        + ["cpg_theta_FL", "cpg_theta_FR", "cpg_theta_RL", "cpg_theta_RR"]
        + _jnt_cols
    )
    with open(_metrics_path, "w", newline="") as _f:
        _w = csv.DictWriter(_f, fieldnames=_metrics_fields)
        _w.writeheader()
        _w.writerows(_joint_log)
    print(f"[play] metrics log → {_metrics_path}  ({len(_joint_log)} rows)")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
