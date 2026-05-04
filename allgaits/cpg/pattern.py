"""Pattern formation: CPG state → foot target → IK → joint target.

Paper eqs 3–4 (AllGaits §II.A):

    x_foot = x_off − d_step · (r − 1) · cos(θ)
    z_foot = { −h + g_c · sin(θ)     if sin(θ) > 0   (swing)
             { −h + g_p · sin(θ)     otherwise       (stance penetration)

`y_foot` is NOT modulated by the CPG — the paper specifies only x (sagittal
forward-back) and z (vertical). We fix y at the leg's nominal lateral
hip-offset `d_y`, keeping hip abduction near neutral during locomotion.

Style parameters (user-settable at deployment, resampled per-reset during
training):
    h      — nominal body height (foot hangs below hip by this amount at rest)
    g_c    — max swing-phase ground clearance
    g_p    — max stance-phase ground penetration (typically ≤ 0.015 m)
    x_off  — forward/backward foot offset from the hip (resting pose)
    d_step — max forward step length (amplitude scale)

All inputs are batched; leading shape is typically `(num_envs, num_legs=4)`.
"""

from __future__ import annotations

import torch

from allgaits.kinematics.b1 import B1_COM_OFFSET_X, B1LegKinematics, LEG_DY


# ---------------------------------------------------------------------------
# B1 training-time style-parameter ranges
# ---------------------------------------------------------------------------
# AllGaits §II-B trained on Go1 (standing height ~0.28 m) with:
#     h ∈ [0.18, 0.35],  x_off ∈ [-0.08, 0.03],
#     g_c ∈ [0.02, 0.12], g_p ∈ [0, 0.015]
#
# B1 scales up: nominal standing height 0.42 m (1.5× Go1), leg length 0.70 m
# (L1+L2) vs Go1's ~0.41 m (1.7× Go1). We scale heights and clearances by
# ~1.5 and widen x_off while biasing it negative to match B1's COM offset
# (B1_COM_OFFSET_X ≈ -0.018 m behind the trunk's geometric center — see
# allgaits.kinematics.b1). The bias matters because centering foot
# oscillations slightly behind the hips keeps the COM over the support
# polygon, which the paper found to be COT-optimal (AllGaits §III-A).
B1_STYLE_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "h":     (0.27, 0.52),    # body height, m (scaled 1.5× from Go1's [0.18, 0.35])
    "g_c":   (0.03, 0.18),    # swing-phase ground clearance, m (1.5×)
    "g_p":   (0.0, 0.022),    # stance-phase penetration, m (1.5×)
    "x_off": (-0.12, 0.04),   # sagittal foot offset, m — biased negative per B1 COM
}


def cpg_to_foot_target(
    r: torch.Tensor,
    theta: torch.Tensor,
    h: torch.Tensor | float,
    g_c: torch.Tensor | float,
    g_p: torch.Tensor | float,
    x_off: torch.Tensor | float,
    d_step: torch.Tensor | float,
    d_y: torch.Tensor,
) -> torch.Tensor:
    """Map CPG state (r, θ) + style params → foot target in hip-joint frame.

    Args:
        r, theta: `(num_envs, num_legs)` CPG amplitude and phase.
        h, g_c, g_p, x_off, d_step: scalar or `(num_envs, 1)` / `(num_envs, num_legs)`.
        d_y: `(num_legs,)` signed per-leg lateral hip offset (see `kinematics.LEG_DY`).

    Returns:
        Foot targets of shape `(num_envs, num_legs, 3)` with (x, y, z) in
        the hip-joint frame.
    """
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    # Paper eq 3 verbatim. With stiffness 600 and verified FK, bypass trot
    # drifts at -0.15 m/s (unknown residual dynamics issue — likely body
    # pitch from asymmetric default thigh angles). Flipping the sign gave
    # -0.37 (WORSE). Moving on — trained PPO should overcome this baseline
    # by commanding different CPG parameters than the bypass-constant
    # (μ=1.5, ω=2 Hz) that diagnoses the mechanism.
    x = x_off - d_step * (r - 1.0) * cos_theta

    # Swing (sin θ > 0) uses ground clearance g_c; stance uses penetration g_p
    z_swing = -h + g_c * sin_theta
    z_stance = -h + g_p * sin_theta
    z = torch.where(sin_theta > 0, z_swing, z_stance)

    # Lateral: leg stays at nominal d_y offset (not modulated by CPG).
    # d_y has shape (num_legs,); broadcast to (num_envs, num_legs).
    y = d_y.expand_as(x)

    return torch.stack([x, y, z], dim=-1)


def foot_target_to_joints(
    foot_target: torch.Tensor,
    d_y: torch.Tensor,
) -> torch.Tensor:
    """Inverse kinematics per leg: (num_envs, num_legs, 3) → (num_envs, num_legs, 3).

    Returns joints in (q_hip, q_thigh, q_calf) order per leg.
    """
    return B1LegKinematics.inverse(foot_target, d_y)


class PatternFormation:
    """Stateless pipeline: CPG state → foot target → joint targets.

    Usage:
        pattern = PatternFormation(device=..., dtype=...)
        q_targets = pattern(r, theta, h, g_c, g_p, x_off, d_step)

    `d_step` and the style params can be reset via `set_defaults()` or passed
    on each call. The leg ordering is [FL, FR, RL, RR] (matches the rest of
    the package).
    """

    def __init__(
        self,
        d_step_default: float = 0.15,
        h_default: float = 0.42,
        g_c_default: float = 0.05,
        g_p_default: float = 0.005,
        x_off_default: float = 0.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.d_step = d_step_default
        self.h = h_default
        self.g_c = g_c_default
        self.g_p = g_p_default
        self.x_off = x_off_default
        self.device = torch.device(device)
        self.dtype = dtype
        self.d_y = LEG_DY(device=self.device, dtype=self.dtype)

    def __call__(
        self,
        r: torch.Tensor,
        theta: torch.Tensor,
        h: torch.Tensor | float | None = None,
        g_c: torch.Tensor | float | None = None,
        g_p: torch.Tensor | float | None = None,
        x_off: torch.Tensor | float | None = None,
        d_step: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute foot targets and joint-position targets.

        Returns:
            `(foot_target, joint_target)` both shaped `(num_envs, num_legs, 3)`.
        """
        foot_target = cpg_to_foot_target(
            r=r,
            theta=theta,
            h=self.h if h is None else h,
            g_c=self.g_c if g_c is None else g_c,
            g_p=self.g_p if g_p is None else g_p,
            x_off=self.x_off if x_off is None else x_off,
            d_step=self.d_step if d_step is None else d_step,
            d_y=self.d_y,
        )
        joint_target = foot_target_to_joints(foot_target, self.d_y)
        return foot_target, joint_target
