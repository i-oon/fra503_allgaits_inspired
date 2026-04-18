"""Unit tests for allgaits.cpg.hopf (Hopf oscillator dynamics).

Validates:
  - State shape and initialization
  - Amplitude convergence r → μ (limit cycle)
  - Phase advance under ω with no coupling
  - Gait lock under strong coupling (trot, pace, pronk, bound)
  - Reset correctness
"""

from __future__ import annotations

import math

import pytest
import torch

from allgaits.cpg.coupling import coupling_matrix, phase_offset_matrix, weight_matrix
from allgaits.cpg.hopf import HopfCPG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run(cpg, mu, omega, phi, w, steps):
    """Run `steps` integration iterations; returns final state."""
    for _ in range(steps):
        cpg.step(mu, omega, phi, w)
    return cpg.state


def _phase_diff(theta_a, theta_b):
    """Unsigned phase difference wrapped to [0, π]."""
    d = (theta_a - theta_b) % (2.0 * math.pi)
    d = torch.where(d > math.pi, 2.0 * math.pi - d, d)
    return d


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def test_init_shapes():
    cpg = HopfCPG(num_envs=8, num_legs=4)
    assert cpg.state.shape == (8, 4, 3)
    assert cpg.r.shape == (8, 4)
    assert cpg.r_dot.shape == (8, 4)
    assert cpg.theta.shape == (8, 4)


def test_init_r_near_unity_rdot_zero():
    cpg = HopfCPG(num_envs=64, num_legs=4)
    # r initialized in [0.9, 1.1]
    assert torch.all(cpg.r >= 0.9 - 1e-5)
    assert torch.all(cpg.r <= 1.1 + 1e-5)
    # r_dot = 0
    torch.testing.assert_close(cpg.r_dot, torch.zeros_like(cpg.r_dot))
    # theta ∈ [0, 2π)
    assert torch.all(cpg.theta >= 0.0)
    assert torch.all(cpg.theta < 2.0 * math.pi)


def test_efference_copy_keys():
    cpg = HopfCPG(num_envs=2, num_legs=4)
    eff = cpg.efference_copy()
    for key in ("r", "r_dot", "theta", "theta_dot"):
        assert key in eff
        assert eff[key].shape == (2, 4)


# ---------------------------------------------------------------------------
# Amplitude limit cycle
# ---------------------------------------------------------------------------
def test_amplitude_converges_to_mu_no_coupling():
    """With w=0 and constant μ, r should converge to μ within ~200 ms."""
    cpg = HopfCPG(num_envs=4, num_legs=4, convergence_factor=150.0, dt=1e-3)
    mu_target = 1.5
    mu = torch.full((4, 4), mu_target)
    omega = torch.zeros(4, 4)
    phi = torch.zeros(4, 4)
    w = torch.zeros(4, 4)  # coupling disabled

    _run(cpg, mu, omega, phi, w, steps=500)  # 500 ms

    # |r - μ| < 0.01 after half a second
    err = (cpg.r - mu_target).abs()
    assert err.max().item() < 0.01, f"max err {err.max()}"


def test_amplitude_converges_to_different_mu_per_leg():
    """Different μ per leg → different limit-cycle amplitudes."""
    cpg = HopfCPG(num_envs=1, num_legs=4, convergence_factor=150.0)
    mu = torch.tensor([[1.0, 1.2, 1.5, 1.8]])
    omega = torch.zeros(1, 4)
    phi = torch.zeros(4, 4)
    w = torch.zeros(4, 4)

    _run(cpg, mu, omega, phi, w, steps=500)

    torch.testing.assert_close(cpg.r, mu, atol=0.02, rtol=0.0)


# ---------------------------------------------------------------------------
# Phase dynamics
# ---------------------------------------------------------------------------
def test_phase_advances_at_omega_no_coupling():
    """With no coupling, θ advances by ω·dt each step."""
    cpg = HopfCPG(num_envs=1, num_legs=4, dt=1e-3)
    # Force theta to 0 initially for a deterministic test
    cpg._state[..., 2] = 0.0
    omega_val = 2.0 * math.pi * 2.0  # 2 Hz
    omega = torch.full((1, 4), omega_val)
    mu = torch.ones(1, 4)
    phi = torch.zeros(4, 4)
    w = torch.zeros(4, 4)

    _run(cpg, mu, omega, phi, w, steps=100)  # 0.1 s

    expected = (omega_val * 0.1) % (2.0 * math.pi)
    err = (cpg.theta - expected).abs()
    assert err.max().item() < 1e-3


# ---------------------------------------------------------------------------
# Gait locking (the critical test — the coupling matrix actually enforces
# the desired phase relationships).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "gait,pairs_inphase,pairs_antiphase",
    [
        # Trot: diagonal pairs in-phase, lateral pairs anti-phase
        ("trot", [(0, 3), (1, 2)], [(0, 1), (2, 3)]),
        # Pace: lateral pairs in-phase, diagonal anti-phase
        ("pace", [(0, 2), (1, 3)], [(0, 1), (2, 3)]),
        # Bound: front pair in-phase, rear pair in-phase, fronts-vs-rears anti-phase
        ("bound", [(0, 1), (2, 3)], [(0, 2), (1, 3)]),
        # Pronk: all four in-phase
        ("pronk", [(0, 1), (1, 2), (2, 3)], []),
    ],
)
def test_gait_locks_phase_relationships(gait, pairs_inphase, pairs_antiphase):
    """Strong coupling should drive phases to the target Φ offsets.

    Convergence criterion: after ~3 seconds of simulation, specified leg
    pairs must be within 5° of the target relationship (in-phase: diff<5°;
    anti-phase: |diff − π|<5°).
    """
    cpg = HopfCPG(num_envs=16, num_legs=4, dt=1e-3)
    phi, w = coupling_matrix(gait, strength=10.0)
    mu = torch.ones(16, 4)
    omega = torch.full((16, 4), 2.0 * math.pi * 2.0)  # 2 Hz
    _run(cpg, mu, omega, phi, w, steps=3000)  # 3 s

    tol = math.radians(5.0)  # 5 degrees
    for i, j in pairs_inphase:
        d = _phase_diff(cpg.theta[:, i], cpg.theta[:, j])
        assert d.max().item() < tol, f"{gait}: legs {i},{j} not in-phase, max diff {math.degrees(d.max().item()):.1f}°"
    for i, j in pairs_antiphase:
        d = _phase_diff(cpg.theta[:, i], cpg.theta[:, j])
        err = (d - math.pi).abs()
        assert err.max().item() < tol, f"{gait}: legs {i},{j} not anti-phase, max err {math.degrees(err.max().item()):.1f}°"


def test_walk_four_beat_quarter_offsets():
    """Lateral sequence walk: adjacent-in-sequence legs ~90° apart."""
    cpg = HopfCPG(num_envs=8, num_legs=4, dt=1e-3)
    phi, w = coupling_matrix("walk", strength=10.0)
    mu = torch.ones(8, 4)
    omega = torch.full((8, 4), 2.0 * math.pi * 2.0)
    _run(cpg, mu, omega, phi, w, steps=4000)  # 4 s to settle

    # Footfall order in lateral seq: FR (0), RL (0.25), FL (0.5), RR (0.75)
    # → phase gap between consecutive-in-sequence legs = π/2 (90°)
    FL, FR, RL, RR = 0, 1, 2, 3
    gap_FR_RL = _phase_diff(cpg.theta[:, RL], cpg.theta[:, FR])
    err = (gap_FR_RL - math.pi / 2).abs()
    assert err.max().item() < math.radians(10.0), \
        f"walk FR→RL quarter-gap err {math.degrees(err.max().item()):.1f}°"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
def test_reset_all_envs():
    cpg = HopfCPG(num_envs=4, num_legs=4)
    cpg._state[..., 2] = 42.0  # corrupt phase
    cpg.reset()
    # Phase should be wrapped to [0, 2π)
    assert torch.all(cpg.theta >= 0)
    assert torch.all(cpg.theta < 2.0 * math.pi)


def test_reset_specific_envs_only():
    cpg = HopfCPG(num_envs=8, num_legs=4)
    old_theta = cpg.theta.clone()
    env_ids = torch.tensor([1, 3, 5])
    cpg.reset(env_ids=env_ids)
    # Envs 0, 2, 4, 6, 7 should keep their old phase
    untouched = torch.tensor([0, 2, 4, 6, 7])
    torch.testing.assert_close(cpg.theta[untouched], old_theta[untouched])


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def test_cuda_placement():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cpg = HopfCPG(num_envs=4, num_legs=4, device="cuda")
    phi = phase_offset_matrix("trot", device="cuda")
    w = weight_matrix(device="cuda")
    mu = torch.ones(4, 4, device="cuda")
    omega = torch.full((4, 4), 2.0 * math.pi, device="cuda")
    cpg.step(mu, omega, phi, w)
    assert cpg.state.device.type == "cuda"
