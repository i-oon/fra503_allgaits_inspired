"""Paper-style evaluation plot from play_metrics_log.csv.

Four panels per condition:
  1. Velocity tracking   — v*_x (dashed) vs actual vx (solid)
  2. Foot contact diagram — FL / FR / RL / RR binary bars (black = stance)
  3. CPG phase θ         — per-leg phase in [0, 2π), shows coupling structure
  4. Heading (yaw)       — cumulative rotation from spawn, highlights drift

Single-condition usage:
    python scripts/plot_paper_metrics.py --csv play_metrics_log.csv

Side-by-side comparison (e.g. baseline vs ours):
    python scripts/plot_paper_metrics.py \\
        --csv baseline.csv ours.csv \\
        --labels "Phase A (trot-only)" "Phase B v3 (AllGaits)" \\
        --out comparison.png
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv", nargs="+", default=["play_metrics_log.csv"],
                    help="One or two CSV paths (second = comparison condition).")
parser.add_argument("--labels", nargs="+", default=None,
                    help="Display label for each CSV. Defaults to filename stem.")
parser.add_argument("--env", type=int, default=0, help="Which env index to plot.")
parser.add_argument("--smooth", type=int, default=10,
                    help="Rolling-window width for velocity trace (steps). 0 = off.")
parser.add_argument("--dt", type=float, default=0.01,
                    help="Policy step period (s). Default 0.01 s = 100 Hz.")
parser.add_argument("--out", type=str, default=None,
                    help="Save figure to this path instead of showing it.")
args = parser.parse_args()

LEGS = ("FL", "FR", "RL", "RR")
LEG_COLORS = {"FL": "#4C72B0", "FR": "#DD8452", "RL": "#55A868", "RR": "#C44E52"}

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

# ── Load data ──────────────────────────────────────────────────────────────────
if args.labels is None:
    args.labels = [os.path.splitext(os.path.basename(p))[0] for p in args.csv]

datasets: list[pd.DataFrame] = []
for path in args.csv:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found — run play.py first.")
    df = pd.read_csv(path)
    df = df[df["env"] == args.env].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows for env={args.env} in {path}.")
    datasets.append(df)

n_cond = len(datasets)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="same")


def _gait_segments(df: pd.DataFrame) -> list[tuple[int, int, str]]:
    steps = df["step"].values
    gaits = df["gait"].values
    segs: list[tuple[int, int, str]] = []
    cur, s0 = gaits[0], steps[0]
    for i in range(1, len(steps)):
        if gaits[i] != cur:
            segs.append((s0, steps[i - 1], cur))
            cur, s0 = gaits[i], steps[i]
    segs.append((s0, steps[-1], cur))
    return segs


def _shade_gaits(ax, segs, alpha=0.12):
    for s0, s1, g in segs:
        ax.axvspan(s0 * args.dt, s1 * args.dt, color=GAIT_COLORS.get(g, "#CCC"), alpha=alpha, linewidth=0)


def _step_to_s(steps: np.ndarray) -> np.ndarray:
    return steps * args.dt


# ── Figure layout ──────────────────────────────────────────────────────────────
# Rows: velocity | contact diagram | CPG phase | yaw
ROW_HEIGHTS = [2.0, 2.2, 1.8, 1.4]
fig, axes = plt.subplots(
    4, n_cond,
    figsize=(7 * n_cond, sum(ROW_HEIGHTS) + 0.5),
    gridspec_kw={"height_ratios": ROW_HEIGHTS},
    sharex="col",
    squeeze=False,
)
fig.suptitle(
    f"AllGaits evaluation — env {args.env}",
    fontsize=13, y=0.99,
)

cond_colors = ["#2166AC", "#D6604D"]   # blue for cond 0, red for cond 1

for ci, (df, label) in enumerate(zip(datasets, args.labels)):
    steps = df["step"].values
    t = _step_to_s(steps)
    segs = _gait_segments(df)

    col_color = cond_colors[ci % len(cond_colors)]

    # ── Row 0: Velocity tracking ────────────────────────────────────────────
    ax = axes[0, ci]
    vx_cmd  = df["vx_cmd"].values
    vx_act  = _smooth(df["bvx"].values, args.smooth)
    ax.step(t, vx_cmd, color="black", linewidth=1.2, linestyle="--",
            where="post", label="$v^*_x$ (cmd)", alpha=0.8)
    ax.plot(t, vx_act, color=col_color, linewidth=1.4, label="$v_x$ (actual)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Velocity (m/s)", fontsize=9)
    ax.set_title(label, fontsize=11, fontweight="bold", pad=6)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    _shade_gaits(ax, segs)

    # ── Row 1: Foot contact gait diagram ────────────────────────────────────
    ax = axes[1, ci]
    for i, leg in enumerate(LEGS):
        ct = df[f"ct_{leg}"].values.astype(bool)
        ax.fill_between(t, i - 0.42, i + 0.42, where=ct,
                        color="black", step="post", linewidth=0)
        ax.fill_between(t, i - 0.42, i + 0.42, where=~ct,
                        color="#E8E8E8", step="post", linewidth=0)
    ax.set_ylim(-0.5, len(LEGS) - 0.5)
    ax.set_yticks(range(len(LEGS)))
    ax.set_yticklabels(LEGS, fontsize=9)
    ax.set_ylabel("Leg", fontsize=9)
    for s0, _, _ in segs[1:]:
        ax.axvline(s0 * args.dt, color="tab:orange",
                   linewidth=1.5, linestyle="--", alpha=0.9)
    ax.set_title("Foot contact diagram (■ stance  □ swing)", fontsize=9, pad=3)

    # ── Row 2: CPG phase θ ──────────────────────────────────────────────────
    ax = axes[2, ci]
    for leg in LEGS:
        theta = df[f"cpg_theta_{leg}"].values / (2 * math.pi)   # normalise to [0, 1)
        # Scatter every 3rd sample to avoid line-wrap artifacts at 0/1 boundary
        ax.scatter(t[::3], theta[::3], s=0.8,
                   color=LEG_COLORS[leg], label=leg, rasterized=True)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("CPG phase (cycles)", fontsize=9)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper right", fontsize=8, ncol=4, markerscale=4,
              handletextpad=0.2, columnspacing=0.5, framealpha=0.8)
    ax.grid(True, linewidth=0.35, alpha=0.5)
    _shade_gaits(ax, segs)
    ax.set_title("CPG phase θ per leg", fontsize=9, pad=3)

    # ── Row 3: Yaw (heading drift) ──────────────────────────────────────────
    ax = axes[3, ci]
    yaw = df["yaw_deg"].values
    # Unwrap degrees to remove ±180 jumps for a clean cumulative-drift view
    yaw_unwrapped = np.degrees(np.unwrap(np.radians(yaw)))
    ax.plot(t, yaw_unwrapped, color="#8B6914", linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Yaw (°)", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.grid(True, linewidth=0.35, alpha=0.5)
    _shade_gaits(ax, segs)
    ax.set_title("Heading drift (yaw)", fontsize=9, pad=3)

# ── Gait legend ────────────────────────────────────────────────────────────────
all_gaits = list(dict.fromkeys(
    g for df in datasets for _, _, g in _gait_segments(df)
))
if len(all_gaits) > 1:
    patches = [
        mpatches.Patch(color=GAIT_COLORS.get(g, "#CCC"), label=g, alpha=0.6)
        for g in all_gaits
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(all_gaits),
               fontsize=9, title="Gait segment", title_fontsize=9,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)
    fig.subplots_adjust(bottom=0.07)

plt.tight_layout(rect=[0, 0.05, 1, 0.98])

if args.out:
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[plot_paper_metrics] saved → {args.out}")
else:
    plt.show()
