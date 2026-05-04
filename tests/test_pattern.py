"""Unit tests for allgaits.cpg.pattern (pattern formation + IK pipeline)."""

from __future__ import annotations

import math

import pytest
import torch

from allgaits.cpg.pattern import (
    B1_STYLE_PARAM_RANGES,
    PatternFormation,
    cpg_to_foot_target,
    foot_target_to_joints,
)
from allgaits.kinematics.b1 import B1_COM_OFFSET_X, HIP_ABDUCTION_ARM, LEG_DY, B1LegKinematics


# ---------------------------------------------------------------------------
# Paper eqs 3–4 — foot-target formulas
# ---------------------------------------------------------------------------
def test_resting_cpg_state_places_foot_at_nominal_hang():
    """r=1, sin(θ)<=0 (stance), x_off=0 → foot at (0, d_y, -h)."""
    r = torch.ones(1, 4)
    theta = torch.full((1, 4), 3 * math.pi / 2)  # sin=-1 → stance
    d_y = LEG_DY()
    foot = cpg_to_foot_target(
        r=r, theta=theta, h=0.42, g_c=0.05, g_p=0.0, x_off=0.0, d_step=0.15, d_y=d_y
    )
    # x = 0 - 0.15·(1-1)·cos(θ) = 0
    torch.testing.assert_close(foot[..., 0], torch.zeros_like(foot[..., 0]), atol=1e-6, rtol=0.0)
    # z = -h + 0·sin = -0.42
    torch.testing.assert_close(
        foot[..., 2], torch.full_like(foot[..., 2], -0.42), atol=1e-6, rtol=0.0
    )
    # y = d_y nominal
    torch.testing.assert_close(foot[..., 1], d_y.expand(1, 4), atol=1e-6, rtol=0.0)


def test_swing_phase_lifts_foot_by_gc():
    """θ=π/2 (sin=1) in swing → z = -h + g_c."""
    r = torch.ones(1, 4)
    theta = torch.full((1, 4), math.pi / 2)
    d_y = LEG_DY()
    foot = cpg_to_foot_target(
        r=r, theta=theta, h=0.42, g_c=0.08, g_p=0.0, x_off=0.0, d_step=0.15, d_y=d_y
    )
    torch.testing.assert_close(
        foot[..., 2], torch.full_like(foot[..., 2], -0.42 + 0.08), atol=1e-6, rtol=0.0
    )


def test_stance_phase_penetration():
    """sin(θ) < 0 → z uses g_p (stance penetration), not g_c."""
    r = torch.ones(1, 4)
    theta = torch.full((1, 4), 3 * math.pi / 2)  # sin=-1
    d_y = LEG_DY()
    foot = cpg_to_foot_target(
        r=r, theta=theta, h=0.42, g_c=0.05, g_p=0.01, x_off=0.0, d_step=0.15, d_y=d_y
    )
    # z = -0.42 + 0.01·(-1) = -0.43
    torch.testing.assert_close(
        foot[..., 2], torch.full_like(foot[..., 2], -0.43), atol=1e-6, rtol=0.0
    )


def test_amplitude_r_modulates_step_length():
    """Larger r → larger sagittal foot excursion (paper eq 3 verbatim)."""
    theta = torch.full((1, 4), 0.0)  # cos(0)=1
    d_y = LEG_DY()
    foot_small = cpg_to_foot_target(
        r=torch.full((1, 4), 1.0), theta=theta, h=0.42, g_c=0.05, g_p=0.0,
        x_off=0.0, d_step=0.15, d_y=d_y,
    )
    foot_big = cpg_to_foot_target(
        r=torch.full((1, 4), 2.0), theta=theta, h=0.42, g_c=0.05, g_p=0.0,
        x_off=0.0, d_step=0.15, d_y=d_y,
    )
    # x = 0 - 0.15·(r - 1)·1 → r=1: 0; r=2: -0.15
    torch.testing.assert_close(foot_small[..., 0], torch.zeros_like(foot_small[..., 0]))
    torch.testing.assert_close(
        foot_big[..., 0], torch.full_like(foot_big[..., 0], -0.15), atol=1e-6, rtol=0.0
    )


def test_x_off_shifts_foot_baseline():
    """Nonzero x_off shifts the whole sagittal trajectory."""
    r = torch.ones(1, 4)
    theta = torch.full((1, 4), 3 * math.pi / 2)
    d_y = LEG_DY()
    foot = cpg_to_foot_target(
        r=r, theta=theta, h=0.42, g_c=0.05, g_p=0.0, x_off=0.05, d_step=0.15, d_y=d_y
    )
    torch.testing.assert_close(
        foot[..., 0], torch.full_like(foot[..., 0], 0.05), atol=1e-6, rtol=0.0
    )


# ---------------------------------------------------------------------------
# Pipeline: foot target → joints → FK recovers foot target
# ---------------------------------------------------------------------------
def test_full_pipeline_fk_recovers_target():
    """CPG→foot→IK→FK should return the original foot target."""
    torch.manual_seed(0)
    N = 16
    r = 1.0 + 0.2 * torch.randn(N, 4)
    theta = 2 * math.pi * torch.rand(N, 4)
    d_y = LEG_DY()

    foot_target = cpg_to_foot_target(
        r=r, theta=theta, h=0.42, g_c=0.05, g_p=0.005, x_off=0.0, d_step=0.15, d_y=d_y
    )
    joint_target = foot_target_to_joints(foot_target, d_y)

    # Forward kinematics from joints should recover foot_target
    q_h, q_t, q_c = joint_target.unbind(-1)
    foot_recovered = B1LegKinematics.forward(q_h, q_t, q_c, d_y)
    torch.testing.assert_close(foot_recovered, foot_target, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# PatternFormation class API
# ---------------------------------------------------------------------------
def test_pattern_formation_class_end_to_end():
    """Class-level API returns matching (foot_target, joint_target) tensors."""
    N = 4
    pattern = PatternFormation(d_step_default=0.15, h_default=0.42)
    r = torch.ones(N, 4)
    theta = torch.linspace(0, 2 * math.pi, N * 4).reshape(N, 4)
    foot_target, joint_target = pattern(r, theta)
    assert foot_target.shape == (N, 4, 3)
    assert joint_target.shape == (N, 4, 3)


def test_pattern_formation_override_style_params():
    """Passing h, g_c, etc. at call time overrides defaults."""
    pattern = PatternFormation(h_default=0.42)
    r = torch.ones(1, 4)
    theta = torch.full((1, 4), 3 * math.pi / 2)  # stance
    # With custom h=0.30 and g_p=0, z should be -0.30
    foot_target, _ = pattern(r, theta, h=0.30, g_p=0.0)
    torch.testing.assert_close(
        foot_target[..., 2], torch.full_like(foot_target[..., 2], -0.30), atol=1e-6, rtol=0.0
    )


# ---------------------------------------------------------------------------
# Consistency with FK of the nominal stance
# ---------------------------------------------------------------------------
def test_nominal_pattern_yields_reasonable_default_pose():
    """At r=1, stance, the IK-derived joints should be within B1's joint limits."""
    r = torch.ones(1, 4)
    theta = torch.full((1, 4), 3 * math.pi / 2)
    pattern = PatternFormation(h_default=0.42, g_c_default=0.05, g_p_default=0.0, x_off_default=0.0)
    _, joints = pattern(r, theta)
    q_h, q_t, q_c = joints.unbind(-1)
    # URDF joint limits: hip [-0.75, 0.75], thigh [-1.0, 3.5], calf [-2.6, -0.6]
    assert torch.all(q_h.abs() < 0.75)
    assert torch.all((q_t >= -1.0) & (q_t <= 3.5))
    assert torch.all((q_c >= -2.6) & (q_c <= -0.6))


def test_cuda_placement():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pattern = PatternFormation(device="cuda")
    r = torch.ones(4, 4, device="cuda")
    theta = torch.zeros(4, 4, device="cuda")
    foot, joints = pattern(r, theta)
    assert foot.device.type == "cuda"
    assert joints.device.type == "cuda"


# ---------------------------------------------------------------------------
# B1 training-time style-parameter ranges
# ---------------------------------------------------------------------------
def test_b1_style_param_ranges_cover_required_keys():
    assert set(B1_STYLE_PARAM_RANGES) == {"h", "g_c", "g_p", "x_off"}


def test_b1_style_param_ranges_lower_less_than_upper():
    for key, (lo, hi) in B1_STYLE_PARAM_RANGES.items():
        assert lo < hi, f"{key}: lower bound must be less than upper ({lo} vs {hi})"


def test_b1_height_range_brackets_nominal_standing():
    """h range must contain the nominal standing height 0.42 m."""
    lo, hi = B1_STYLE_PARAM_RANGES["h"]
    assert lo < 0.42 < hi


def test_b1_x_off_range_biased_negative_per_com():
    """x_off range should be biased negative to match B1's backward COM offset."""
    lo, hi = B1_STYLE_PARAM_RANGES["x_off"]
    # Range should INCLUDE the COM offset (so policy can discover its optimality)
    assert lo <= B1_COM_OFFSET_X <= hi
    # Range should be biased negative (lower-bound magnitude >> upper-bound)
    assert abs(lo) > abs(hi), \
        f"x_off range should be biased negative: got [{lo}, {hi}]"


def test_b1_clearance_and_penetration_non_negative():
    """g_c (swing clearance) and g_p (stance penetration) must be ≥ 0."""
    assert B1_STYLE_PARAM_RANGES["g_c"][0] >= 0
    assert B1_STYLE_PARAM_RANGES["g_p"][0] >= 0
