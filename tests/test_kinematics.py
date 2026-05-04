"""Unit tests for allgaits.kinematics.b1 (forward + inverse kinematics)."""

from __future__ import annotations

import math

import pytest
import torch

from allgaits.kinematics.b1 import (
    B1_COM_OFFSET_X,
    HIP_ABDUCTION_ARM,
    HIP_POSITIONS,
    LEG_DY,
    LEG_DY_SIGNS,
    LOWER_LEG,
    UPPER_LEG,
    B1LegKinematics,
)


# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
def test_geometry_from_urdf():
    assert UPPER_LEG == pytest.approx(0.35)
    assert LOWER_LEG == pytest.approx(0.35)
    assert HIP_ABDUCTION_ARM == pytest.approx(0.12675)


def test_leg_dy_signs():
    # [FL, FR, RL, RR] → left legs positive, right legs negative
    assert LEG_DY_SIGNS == (+1, -1, +1, -1)
    d_y = LEG_DY()
    assert torch.all(d_y[[0, 2]] > 0)  # FL, RL
    assert torch.all(d_y[[1, 3]] < 0)  # FR, RR


def test_hip_positions_match_urdf():
    """Hip joint origins in trunk frame, canonical [FL, FR, RL, RR] order."""
    expected = (
        (+0.3455, +0.072, 0.0),
        (+0.3455, -0.072, 0.0),
        (-0.3455, +0.072, 0.0),
        (-0.3455, -0.072, 0.0),
    )
    assert HIP_POSITIONS == expected


def test_b1_com_is_behind_geometric_center():
    """B1's overall COM sits ~1.8 cm behind the geometric center of the body.

    Matches AllGaits' observation on Go1 (§III-A): mass asymmetry from the
    front/rear thigh-angle defaults (0.8 vs 1.0 rad) projects rear-leg mass
    further backward than front-leg mass forward. This motivates negative
    x_off in the B1 training range (see B1_STYLE_PARAM_RANGES).
    """
    # Value computed from URDF inertial sum at the default standing pose.
    # Recompute if robot config changes; current: −0.018 m.
    assert B1_COM_OFFSET_X < -0.005, "COM should be behind geometric center"
    assert B1_COM_OFFSET_X > -0.03, "COM offset shouldn't exceed ~3 cm"


# ---------------------------------------------------------------------------
# Forward kinematics against manually computed reference
# ---------------------------------------------------------------------------
def test_fk_zero_pose_foot_directly_below_hip_plus_abduction():
    """With q_hip=q_thigh=q_calf=0, foot = (0, d_y, -(L1+L2)) in hip frame."""
    q = torch.zeros(1)
    d_y = torch.tensor([HIP_ABDUCTION_ARM])
    p = B1LegKinematics.forward(q, q, q, d_y)
    expected = torch.tensor([[0.0, HIP_ABDUCTION_ARM, -(UPPER_LEG + LOWER_LEG)]])
    torch.testing.assert_close(p, expected, atol=1e-6, rtol=0.0)


def test_fk_b1_default_front_leg_pose():
    """B1 default front pose: q_hip=0.1, q_thigh=0.8, q_calf=-1.5 — foot below hip."""
    q_h = torch.tensor([0.1])
    q_t = torch.tensor([0.8])
    q_c = torch.tensor([-1.5])
    d_y = torch.tensor([HIP_ABDUCTION_ARM])
    p = B1LegKinematics.forward(q_h, q_t, q_c, d_y)
    # Foot should be below hip (z < 0) and close to directly under it in x
    assert p[0, 2].item() < -0.4, "foot not sufficiently below hip"
    assert p[0, 2].item() > -0.55, "foot unrealistically far below hip"
    assert abs(p[0, 0].item()) < 0.15, "foot too far forward/back of hip"


def test_fk_b1_default_rear_leg_pose():
    """B1 rear default q_thigh=1.0 should place foot slightly further back."""
    q_h = torch.tensor([0.1])
    q_t = torch.tensor([1.0])
    q_c = torch.tensor([-1.5])
    d_y = torch.tensor([HIP_ABDUCTION_ARM])
    p = B1LegKinematics.forward(q_h, q_t, q_c, d_y)
    assert p[0, 2].item() < -0.4


# ---------------------------------------------------------------------------
# FK ∘ IK round-trip
# ---------------------------------------------------------------------------
def _sample_reachable_foot_targets(num: int, d_y_val: float, seed: int = 0):
    """Generate `num` foot targets inside the reachable workspace.

    Samples sagittal (dx, dz) in polar coordinates inside an annulus
    [r_min, r_max] with r_max = 0.9·(L1+L2), restricted to the lower half
    (dz ≤ 0). This guarantees every sample is reachable — no rejection.
    """
    g = torch.Generator().manual_seed(seed)
    L1, L2 = UPPER_LEG, LOWER_LEG
    r_max = (L1 + L2) * 0.9
    r_min = 0.15   # stay away from the near-hip singularity

    # Uniform sampling inside an annulus (area-uniform via sqrt on r²)
    r_sq = r_min**2 + torch.rand(num, generator=g) * (r_max**2 - r_min**2)
    r_sample = torch.sqrt(r_sq)
    # φ ∈ [π, 2π] → sin(φ) ≤ 0 → dz ≤ 0 (leg hangs down)
    phi = math.pi + math.pi * torch.rand(num, generator=g)
    dx = r_sample * torch.cos(phi)
    dz = r_sample * torch.sin(phi)

    # Abduction angle q_h ∈ [-0.4, 0.4] rad — within B1's hip limits
    q_h = (torch.rand(num, generator=g) * 2 - 1) * 0.4
    d_y = torch.full((num,), d_y_val)
    cos_h = torch.cos(q_h)
    sin_h = torch.sin(q_h)
    px = dx
    py = d_y * cos_h - dz * sin_h
    pz = d_y * sin_h + dz * cos_h
    return torch.stack([px, py, pz], dim=-1), d_y


@pytest.mark.parametrize("d_y_val", [+HIP_ABDUCTION_ARM, -HIP_ABDUCTION_ARM])
def test_ik_then_fk_roundtrip(d_y_val):
    """IK followed by FK should return the original foot target."""
    foot_targets, d_y = _sample_reachable_foot_targets(num=128, d_y_val=d_y_val)
    q = B1LegKinematics.inverse(foot_targets, d_y)
    q_h, q_t, q_c = q.unbind(-1)
    foot_recovered = B1LegKinematics.forward(q_h, q_t, q_c, d_y)
    torch.testing.assert_close(foot_recovered, foot_targets, atol=1e-4, rtol=1e-4)


def test_ik_recovers_b1_default_front_pose():
    """Default front-leg joints (0.1, 0.8, -1.5) round-trip through FK→IK."""
    q_h = torch.tensor([0.1])
    q_t = torch.tensor([0.8])
    q_c = torch.tensor([-1.5])
    d_y = torch.tensor([HIP_ABDUCTION_ARM])
    p = B1LegKinematics.forward(q_h, q_t, q_c, d_y)
    q_rec = B1LegKinematics.inverse(p, d_y)
    torch.testing.assert_close(q_rec[:, 0], q_h, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(q_rec[:, 1], q_t, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(q_rec[:, 2], q_c, atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------
# Left/right symmetry
# ---------------------------------------------------------------------------
def test_fk_left_right_mirror():
    """Same sagittal pose, opposite d_y signs → y-coordinate mirrors."""
    q_h = torch.tensor([0.0])
    q_t = torch.tensor([0.8])
    q_c = torch.tensor([-1.5])
    p_L = B1LegKinematics.forward(q_h, q_t, q_c, torch.tensor([+HIP_ABDUCTION_ARM]))
    p_R = B1LegKinematics.forward(q_h, q_t, q_c, torch.tensor([-HIP_ABDUCTION_ARM]))
    # x, z identical; y mirrored
    assert p_L[0, 0].item() == pytest.approx(p_R[0, 0].item())
    assert p_L[0, 2].item() == pytest.approx(p_R[0, 2].item())
    assert p_L[0, 1].item() == pytest.approx(-p_R[0, 1].item())


# ---------------------------------------------------------------------------
# Workspace / out-of-reach handling
# ---------------------------------------------------------------------------
def test_is_reachable_flags_out_of_workspace():
    """Points beyond L1+L2 are unreachable; clamp gracefully (no NaN)."""
    # Inside workspace
    p_inside = torch.tensor([[0.0, HIP_ABDUCTION_ARM, -0.5]])
    # Way out (2 m away)
    p_outside = torch.tensor([[2.0, HIP_ABDUCTION_ARM, -0.5]])
    d_y = torch.tensor([HIP_ABDUCTION_ARM])
    assert B1LegKinematics.is_reachable(p_inside, d_y).item() is True
    assert B1LegKinematics.is_reachable(p_outside, d_y).item() is False


def test_ik_no_nan_at_boundary():
    """IK must return finite joints even for slightly out-of-reach targets."""
    # Target just beyond L1+L2=0.7
    p = torch.tensor([[0.0, HIP_ABDUCTION_ARM, -0.80]])
    d_y = torch.tensor([HIP_ABDUCTION_ARM])
    q = B1LegKinematics.inverse(p, d_y)
    assert torch.all(torch.isfinite(q))


# ---------------------------------------------------------------------------
# Batched / multi-leg shape
# ---------------------------------------------------------------------------
def test_batched_all_4_legs():
    """(num_envs, num_legs) batching with per-leg d_y works end-to-end."""
    N = 8
    # q has shape (num_envs, num_legs, 3) — the 3 is (q_hip, q_thigh, q_calf)
    q = torch.zeros(N, 4, 3)
    q[..., 1] = 0.8   # thigh
    q[..., 2] = -1.5  # calf
    d_y = LEG_DY()
    p = B1LegKinematics.forward(q[..., 0], q[..., 1], q[..., 2], d_y)
    assert p.shape == (N, 4, 3)
    # Left legs (FL=0, RL=2): y > 0; right legs (FR=1, RR=3): y < 0
    assert torch.all(p[:, [0, 2], 1] > 0)
    assert torch.all(p[:, [1, 3], 1] < 0)
    # All feet below hip
    assert torch.all(p[..., 2] < 0)


def test_cuda_placement():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    q = torch.zeros(4, 4, device="cuda")
    q[..., 1] = 0.8
    q[..., 2] = -1.5
    d_y = LEG_DY(device="cuda")
    p = B1LegKinematics.forward(q[..., 0], q[..., 1], q[..., 2], d_y)
    q_rec = B1LegKinematics.inverse(p, d_y)
    assert p.device.type == "cuda"
    assert q_rec.device.type == "cuda"
