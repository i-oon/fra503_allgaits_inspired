"""Gait coupling matrices Φ (phase offsets) and W (coupling weights).

Derived from AllGaits Fig. 3 contact timings (fraction of cycle ∈ [0, 1))
for the four legs.

Canonical leg order: [FL, FR, RL, RR] (see allgaits/__init__.py).
The paper's matrix ordering is [FR, FL, HR, HL]; values below have been
transposed to our convention.

For oscillator dynamics `θ̇_i = ω_i + Σ_j r_j · w_ij · sin(θ_j − θ_i − φ_ij)`
the phase offset φ_ij is the TARGET value of (θ_j − θ_i) at equilibrium.
Given contact fractions c_i, c_j ∈ [0, 1):

    φ_ij = 2π · (c_j − c_i)

Strong coupling (w_ij = 10) forces the oscillators to converge to these
offsets; self-coupling w_ii = 0.

Values for canter, transverse gallop, and rotary gallop are approximated
from Fig. 3 — these are Phase C (week 4) gaits; verify visually by
rendering the contact-timing diagram before using them in training.
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# Contact timings (fraction of cycle) per gait, in [FL, FR, RL, RR] order.
# ---------------------------------------------------------------------------
CONTACT_TIMINGS: dict[str, tuple[float, float, float, float]] = {
    # 4-beat lateral-sequence gaits
    "walk":      (0.50, 0.00, 0.25, 0.75),   # Lateral Sequence Walk
    "amble":     (0.50, 0.00, 0.80, 0.30),   # Amble (TODO: refine from Fig. 3)
    # 2-beat symmetric gaits
    "trot":      (0.50, 0.00, 0.00, 0.50),   # diagonal pairs in-phase
    "pace":      (0.50, 0.00, 0.50, 0.00),   # lateral pairs in-phase
    "bound":     (0.50, 0.50, 0.00, 0.00),   # front pair / rear pair
    "pronk":     (0.00, 0.00, 0.00, 0.00),   # all-limbs in-phase
    # Asymmetric high-speed gaits (approximate — verify against Fig. 3)
    "canter":            (0.50, 0.30, 0.00, 0.00),
    "transverse_gallop": (0.50, 0.60, 0.00, 0.10),
    "rotary_gallop":     (0.60, 0.50, 0.00, 0.10),
}

GAIT_NAMES: tuple[str, ...] = tuple(CONTACT_TIMINGS.keys())
NUM_GAITS: int = len(GAIT_NAMES)


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------
def phase_offset_matrix(
    gait: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build Φ ∈ ℝ^(4×4) with φ_ij = 2π·(c_j − c_i).

    Rows are oscillator i, columns are j. Diagonal is zero.
    """
    if gait not in CONTACT_TIMINGS:
        raise ValueError(f"Unknown gait {gait!r}. Valid: {list(GAIT_NAMES)}")
    c = torch.tensor(CONTACT_TIMINGS[gait], device=device, dtype=dtype)
    # φ_ij = 2π (c_j − c_i)
    phi = 2.0 * math.pi * (c.unsqueeze(0) - c.unsqueeze(1))
    return phi


def weight_matrix(
    num_legs: int = 4,
    strength: float = 10.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build W ∈ ℝ^(L×L) with off-diagonal = `strength` and zero diagonal.

    Paper uses w_ij = 10 ("strong coupling") to enforce the gait.
    """
    w = strength * (
        torch.ones(num_legs, num_legs, device=device, dtype=dtype)
        - torch.eye(num_legs, device=device, dtype=dtype)
    )
    return w


def coupling_matrix(
    gait: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    strength: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (Φ, W) pair for the named gait."""
    phi = phase_offset_matrix(gait, device=device, dtype=dtype)
    w = weight_matrix(num_legs=phi.shape[-1], strength=strength, device=device, dtype=dtype)
    return phi, w


def batch_coupling_matrices(
    gait_names: list[str],
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    strength: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack per-env Φ and W matrices for heterogeneous gait assignments.

    Given `gait_names` of length N, returns:
        phi: (N, 4, 4)
        w:   (N, 4, 4)

    Used during training when different envs use different gaits (AllGaits
    re-samples the coupling matrix every 3 s per env).
    """
    phis = torch.stack(
        [phase_offset_matrix(g, device=device, dtype=dtype) for g in gait_names]
    )
    w_single = weight_matrix(strength=strength, device=device, dtype=dtype)
    w = w_single.unsqueeze(0).expand(len(gait_names), -1, -1).contiguous()
    return phis, w
