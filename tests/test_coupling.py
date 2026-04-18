"""Unit tests for allgaits.cpg.coupling."""

from __future__ import annotations

import math

import pytest
import torch

from allgaits.cpg.coupling import (
    CONTACT_TIMINGS,
    GAIT_NAMES,
    batch_coupling_matrices,
    coupling_matrix,
    phase_offset_matrix,
    weight_matrix,
)


def test_all_9_gaits_present():
    assert len(GAIT_NAMES) == 9
    expected = {"walk", "amble", "trot", "pace", "bound", "pronk",
                "canter", "transverse_gallop", "rotary_gallop"}
    assert set(GAIT_NAMES) == expected


def test_contact_timings_in_unit_range():
    for gait, timings in CONTACT_TIMINGS.items():
        assert len(timings) == 4, f"{gait} must have 4 leg timings"
        for t in timings:
            assert 0.0 <= t < 1.0, f"{gait} timing {t} out of [0, 1)"


def test_phi_matrix_shape_and_diagonal():
    for gait in GAIT_NAMES:
        phi = phase_offset_matrix(gait)
        assert phi.shape == (4, 4)
        torch.testing.assert_close(phi.diagonal(), torch.zeros(4))


def test_phi_matrix_antisymmetric():
    """φ_ij = -φ_ji by construction (φ = 2π(c_j − c_i))."""
    for gait in GAIT_NAMES:
        phi = phase_offset_matrix(gait)
        torch.testing.assert_close(phi, -phi.T)


def test_weight_matrix_zero_diagonal_and_symmetric():
    w = weight_matrix(num_legs=4, strength=10.0)
    assert w.shape == (4, 4)
    torch.testing.assert_close(w.diagonal(), torch.zeros(4))
    torch.testing.assert_close(w, w.T)
    # Off-diagonals = 10
    off_diag = w[~torch.eye(4, dtype=torch.bool)]
    torch.testing.assert_close(off_diag, torch.full((12,), 10.0))


def test_trot_phase_pattern():
    """Trot: FL-RR and FR-RL in phase; FL-FR opposite (π offset)."""
    phi = phase_offset_matrix("trot")
    FL, FR, RL, RR = 0, 1, 2, 3
    # Diagonal pairs → 0 offset
    assert abs(phi[FL, RR].item()) < 1e-5
    assert abs(phi[FR, RL].item()) < 1e-5
    # Lateral pairs → π offset (180°)
    assert abs(abs(phi[FL, FR].item()) - math.pi) < 1e-5


def test_pace_phase_pattern():
    """Pace: lateral pairs in phase (FL-RL, FR-RR); diagonals opposite."""
    phi = phase_offset_matrix("pace")
    FL, FR, RL, RR = 0, 1, 2, 3
    assert abs(phi[FL, RL].item()) < 1e-5
    assert abs(phi[FR, RR].item()) < 1e-5
    assert abs(abs(phi[FL, FR].item()) - math.pi) < 1e-5


def test_bound_phase_pattern():
    """Bound: front pair in phase, rear pair in phase, fronts-vs-rears opposite."""
    phi = phase_offset_matrix("bound")
    FL, FR, RL, RR = 0, 1, 2, 3
    assert abs(phi[FL, FR].item()) < 1e-5
    assert abs(phi[RL, RR].item()) < 1e-5
    assert abs(abs(phi[FL, RL].item()) - math.pi) < 1e-5


def test_pronk_all_zero():
    """Pronk: all limbs in phase → Φ is all zeros."""
    phi = phase_offset_matrix("pronk")
    torch.testing.assert_close(phi, torch.zeros(4, 4))


def test_walk_four_beat():
    """Lateral sequence walk: 4 distinct quarter-cycle timings."""
    c = CONTACT_TIMINGS["walk"]
    assert set(c) == {0.0, 0.25, 0.5, 0.75}


def test_coupling_matrix_returns_pair():
    phi, w = coupling_matrix("trot")
    assert phi.shape == w.shape == (4, 4)


def test_batch_coupling_heterogeneous():
    """Different envs with different gaits should produce independent matrices."""
    gaits = ["trot", "walk", "pronk", "bound"]
    phi, w = batch_coupling_matrices(gaits)
    assert phi.shape == (4, 4, 4)
    assert w.shape == (4, 4, 4)
    # Pronk row (index 2) should be all-zero Φ
    torch.testing.assert_close(phi[2], torch.zeros(4, 4))
    # W is gait-independent and shared
    torch.testing.assert_close(w[0], w[1])


def test_unknown_gait_raises():
    with pytest.raises(ValueError, match="Unknown gait"):
        phase_offset_matrix("moonwalk")


def test_device_placement():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    phi = phase_offset_matrix("trot", device="cuda")
    assert phi.device.type == "cuda"
