"""Plot joint-velocity jerk from a play_joint_log.csv produced by play.py.

Jerk = third derivative of joint position = second finite-difference of
joint velocity, divided by dt².  Units: rad/s³.

Usage:
    python scripts/plot_jerk.py                          # reads play_joint_log.csv
    python scripts/plot_jerk.py --csv play_joint_log.csv --env 0
    python scripts/plot_jerk.py --smooth 20 --env 0      # 20-step rolling window
    python scripts/plot_jerk.py --out jerk.png           # save instead of show
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot jerk from play_joint_log.csv.")
parser.add_argument("--csv", type=str, default="play_joint_log.csv")
parser.add_argument("--env", type=int, default=0, help="Which env index to plot.")
parser.add_argument("--smooth", type=int, default=20,
                    help="Rolling-window half-width for smoothing (steps). 0 = no smoothing.")
parser.add_argument("--out", type=str, default=None,
                    help="Save figure to this path instead of displaying it.")
parser.add_argument("--dt", type=float, default=0.01,
                    help="Policy step period in seconds (default 0.01 s = 100 Hz).")
args = parser.parse_args()

# ── Load ───────────────────────────────────────────────────────────────────────
if not os.path.isfile(args.csv):
    raise FileNotFoundError(
        f"{args.csv} not found — run play.py first to generate the log."
    )

df = pd.read_csv(args.csv)
df_env = df[df["env"] == args.env].reset_index(drop=True)
if df_env.empty:
    raise ValueError(f"No rows for env={args.env} in {args.csv}.")

# ── Joint columns ──────────────────────────────────────────────────────────────
LEGS = ("FL", "FR", "RL", "RR")
JOINTS = ("hip", "thigh", "calf")
jnt_cols = [f"{leg}_{jt}" for leg in LEGS for jt in JOINTS]   # 12 columns

steps = df_env["step"].values          # (T,)
jv = df_env[jnt_cols].values           # (T, 12)  joint velocity, rad/s

# ── Jerk via finite differences ────────────────────────────────────────────────
dt = args.dt
# accel: Δv/dt  →  shape (T-1, 12)
jerk_raw = np.diff(jv, n=2, axis=0) / (dt ** 2)   # (T-2, 12)  rad/s³
jerk_steps = steps[2:]                              # aligned step indices

# Mean absolute jerk per leg  (T-2, 4)
jerk_per_leg = np.abs(jerk_raw).reshape(len(jerk_raw), 4, 3).mean(axis=2)

# Body velocity and its jerk
bvx_raw = df_env["bvx"].values                     # (T,)
body_jerk = np.diff(bvx_raw, n=2) / (dt ** 2)     # (T-2,)  m/s³
vx_cmd = df_env["vx_cmd"].values                   # (T,)

# ── Gait segments (for shading) ────────────────────────────────────────────────
gait_col = df_env["gait"].values    # (T,)
segments: list[tuple[int, int, str]] = []   # (start_step, end_step, gait)
cur_gait, seg_start = gait_col[0], steps[0]
for i in range(1, len(steps)):
    if gait_col[i] != cur_gait:
        segments.append((seg_start, steps[i - 1], cur_gait))
        cur_gait, seg_start = gait_col[i], steps[i]
segments.append((seg_start, steps[-1], cur_gait))

GAIT_COLORS = {
    "trot":             "#4C72B0",
    "walk":             "#DD8452",
    "pace":             "#55A868",
    "bound":            "#C44E52",
    "pronk":            "#8172B2",
    "amble":            "#937860",
    "canter":           "#DA8BC3",
    "transverse_gallop":"#8C8C8C",
    "rotary_gallop":    "#CCB974",
    "sampled":          "#EEEEEE",
}

def _shade(ax, alpha: float = 0.12) -> None:
    """Shade each gait segment with its colour."""
    ymin, ymax = ax.get_ylim()
    for s0, s1, gait in segments:
        c = GAIT_COLORS.get(gait, "#CCCCCC")
        ax.axvspan(s0, s1, ymin=0, ymax=1, color=c, alpha=alpha, linewidth=0)
    ax.set_ylim(ymin, ymax)


def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr
    kernel = np.ones(w) / w
    if arr.ndim == 1:
        return np.convolve(arr, kernel, mode="same")
    return np.column_stack([np.convolve(arr[:, i], kernel, mode="same") for i in range(arr.shape[1])])


# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle(
    f"Joint-velocity Jerk — env {args.env}  (dt={dt*1000:.0f} ms, smooth={args.smooth})",
    fontsize=13, y=0.98,
)

LEG_COLORS = {"FL": "#4C72B0", "FR": "#DD8452", "RL": "#55A868", "RR": "#C44E52"}

# ── Subplot 1: per-leg mean |jerk| ─────────────────────────────────────────────
ax = axes[0]
for li, leg in enumerate(LEGS):
    y = _smooth(jerk_per_leg[:, li], args.smooth)
    ax.plot(jerk_steps, y, label=leg, color=LEG_COLORS[leg], linewidth=1.2)
ax.set_ylabel("Mean |jerk| (rad/s³)", fontsize=10)
ax.set_title("Per-leg mean absolute jerk (hip+thigh+calf average)", fontsize=10)
ax.legend(loc="upper right", fontsize=9, ncol=4)
ax.grid(True, linewidth=0.4, alpha=0.5)
_shade(ax)

# ── Subplot 2: per-joint-type mean |jerk| across all legs ──────────────────────
ax = axes[1]
JT_COLORS = {"hip": "#8172B2", "thigh": "#937860", "calf": "#DA8BC3"}
for ji, jt in enumerate(JOINTS):
    # mean over 4 legs for this joint type
    y = np.abs(jerk_raw[:, ji::3]).mean(axis=1)
    y = _smooth(y, args.smooth)
    ax.plot(jerk_steps, y, label=jt, color=JT_COLORS[jt], linewidth=1.2)
ax.set_ylabel("Mean |jerk| (rad/s³)", fontsize=10)
ax.set_title("Per-joint-type mean absolute jerk (averaged over FL/FR/RL/RR)", fontsize=10)
ax.legend(loc="upper right", fontsize=9, ncol=3)
ax.grid(True, linewidth=0.4, alpha=0.5)
_shade(ax)

# ── Subplot 3: body forward velocity + body-level jerk ─────────────────────────
ax = axes[2]
ax2 = ax.twinx()

ax.plot(steps, bvx_raw, color="#4C72B0", linewidth=1.0, alpha=0.6, label="body vx (m/s)")
ax.step(steps, vx_cmd, color="black", linewidth=1.2, linestyle="--", alpha=0.7, label="cmd vx (m/s)", where="post")
ax.set_ylabel("Velocity (m/s)", fontsize=10, color="#4C72B0")
ax.tick_params(axis="y", labelcolor="#4C72B0")

bj_smooth = _smooth(body_jerk, args.smooth)
ax2.plot(jerk_steps, bj_smooth, color="#C44E52", linewidth=1.2, alpha=0.8, label="body jerk (m/s³)")
ax2.set_ylabel("Body jerk (m/s³)", fontsize=10, color="#C44E52")
ax2.tick_params(axis="y", labelcolor="#C44E52")
ax2.axhline(0, color="#C44E52", linewidth=0.5, alpha=0.4)

ax.set_title("Body forward velocity tracking + body-level jerk", fontsize=10)
ax.set_xlabel("Policy step", fontsize=10)
ax.grid(True, linewidth=0.4, alpha=0.5)
_shade(ax)

# Merge legends from ax and ax2
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, ncol=3)

# ── Gait legend patches ────────────────────────────────────────────────────────
unique_gaits = list(dict.fromkeys(g for _, _, g in segments))   # preserve order
if len(unique_gaits) > 1:
    patches = [mpatches.Patch(color=GAIT_COLORS.get(g, "#CCC"), label=g) for g in unique_gaits]
    fig.legend(handles=patches, loc="lower center", ncol=len(unique_gaits),
               fontsize=9, title="Gait segment", title_fontsize=9,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)
    fig.subplots_adjust(bottom=0.09)

# Gait-boundary vertical lines on all axes
for s0, _, _ in segments[1:]:
    for ax_ in axes:
        ax_.axvline(s0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)

plt.tight_layout(rect=[0, 0.06, 1, 0.97])

if args.out:
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[plot_jerk] saved → {args.out}")
else:
    plt.show()
