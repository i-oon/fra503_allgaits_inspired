"""Unitree B1 leg kinematics (forward + inverse) and mass-property utilities.

Geometry (from URDF, per leg in hip-joint frame):
    L1 = L2 = 0.35 m        upper-leg and lower-leg length
    |d_y| = 0.12675 m       hip-to-thigh lateral offset (abduction arm)
                            sign: +1 for left legs (FL, RL), -1 for right legs (FR, RR)

Joint axes (URDF-verified; see SETUP.md):
    q_hip    rotates about +X (abduction, side-to-side splay)
    q_thigh  rotates about +Y (forward-backward swing)
    q_calf   rotates about +Y (knee bend, in thigh frame)

Mass properties (URDF inertials, nominal standing pose):
    Total mass ≈ 62.6 kg. Overall COM sits at x ≈ -0.018 m in the trunk
    frame (1.8 cm BEHIND the geometric center of the hip footprint).
    This shapes the training range for `x_off` in pattern formation:
    centering foot oscillations slightly behind the hips keeps the COM
    over the support polygon. See `B1_COM_OFFSET_X`.

Forward kinematics (foot position in hip-joint frame):
    Δx = -(L1·sin(q_t) + L2·sin(q_t + q_c))
    Δz = -(L1·cos(q_t) + L2·cos(q_t + q_c))
    p  = R_x(q_h) · (Δx, d_y, Δz)

Inverse kinematics assumes the knee bends **backward** (q_calf ≤ 0), which
matches B1's default pose (q_calf = -1.5 rad at rest).

All operations are batched over the leading dimensions — typical shape is
(num_envs, num_legs=4). `d_y` is a per-leg signed scalar or broadcastable
tensor.
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# Constants (URDF-verified)
# ---------------------------------------------------------------------------
UPPER_LEG: float = 0.35       # L1, m
LOWER_LEG: float = 0.35       # L2, m
HIP_ABDUCTION_ARM: float = 0.12675   # |d_y|, m

# Per-leg d_y sign in canonical [FL, FR, RL, RR] order
LEG_DY_SIGNS: tuple[int, int, int, int] = (+1, -1, +1, -1)

# Hip-joint positions in trunk frame, canonical leg order [FL, FR, RL, RR]
HIP_POSITIONS: tuple[tuple[float, float, float], ...] = (
    (+0.3455, +0.072, 0.0),  # FL
    (+0.3455, -0.072, 0.0),  # FR
    (-0.3455, +0.072, 0.0),  # RL
    (-0.3455, -0.072, 0.0),  # RR
)

# Overall-body COM offset (trunk frame, x-axis) at the default standing pose.
# Computed from URDF inertial sum over trunk + legs + rotors. The trunk's
# own COM is nearly centered (+0.009 m); the ~1.8 cm backward bias of the
# total COM comes from the front/rear thigh-angle asymmetry (0.8 vs 1.0 rad)
# projecting rear-leg mass further backward than front-leg mass forward.
# See `compute_com_trunk_frame()` for the full computation.
B1_COM_OFFSET_X: float = -0.018   # meters


def LEG_DY(device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the (4,) tensor of signed d_y values for [FL, FR, RL, RR]."""
    return HIP_ABDUCTION_ARM * torch.tensor(LEG_DY_SIGNS, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
class B1LegKinematics:
    """Batched forward and inverse kinematics for a B1 leg.

    All tensors share the same leading dimensions (typically `(num_envs, num_legs)`).
    `d_y` is broadcast against them: shape `(num_legs,)` or `()`.
    """

    L1: float = UPPER_LEG
    L2: float = LOWER_LEG
    ABD: float = HIP_ABDUCTION_ARM

    @staticmethod
    def forward(
        q_hip: torch.Tensor,
        q_thigh: torch.Tensor,
        q_calf: torch.Tensor,
        d_y: torch.Tensor | float,
    ) -> torch.Tensor:
        """Forward kinematics: joints → foot position in hip-joint frame.

        Returns a tensor of shape `q_hip.shape + (3,)` containing (x, y, z).
        """
        # Sagittal plane (before abduction rotation). Paper/URDF right-hand-rule
        # about +Y axis. If this turns out not to match PhysX's interpretation
        # for the B1 USD, compare against body-frame foot positions reported
        # directly by the articulation (new diagnostic in play.py).
        dx = -(B1LegKinematics.L1 * torch.sin(q_thigh)
               + B1LegKinematics.L2 * torch.sin(q_thigh + q_calf))
        dz = -(B1LegKinematics.L1 * torch.cos(q_thigh)
               + B1LegKinematics.L2 * torch.cos(q_thigh + q_calf))

        # Abduction rotation R_x(q_hip) applied to (dx, d_y, dz).
        # d_y broadcasts via PyTorch rules: shape (num_legs,) broadcasts
        # against q_hip's trailing dim; no explicit expand needed.
        if not isinstance(d_y, torch.Tensor):
            d_y = torch.as_tensor(d_y, device=q_hip.device, dtype=q_hip.dtype)

        cos_h = torch.cos(q_hip)
        sin_h = torch.sin(q_hip)
        px = dx
        py = d_y * cos_h - dz * sin_h
        pz = d_y * sin_h + dz * cos_h

        return torch.stack([px, py, pz], dim=-1)

    @staticmethod
    def inverse(
        foot_target: torch.Tensor,
        d_y: torch.Tensor | float,
    ) -> torch.Tensor:
        """Inverse kinematics: foot position in hip frame → joints.

        Args:
            foot_target: shape `(..., 3)` with (x, y, z) in hip-joint frame.
            d_y: broadcastable per-leg lateral hip-offset (signed).

        Returns:
            Joint angles stacked as `(..., 3)` in order `(q_hip, q_thigh, q_calf)`.

        Behaviour at workspace boundary: `cos(q_calf)` is clamped to `[-1, 1]`
        so the solver returns the nearest reachable joint angle rather than NaN.
        """
        px = foot_target[..., 0]
        py = foot_target[..., 1]
        pz = foot_target[..., 2]

        if not isinstance(d_y, torch.Tensor):
            d_y = torch.as_tensor(d_y, device=foot_target.device, dtype=foot_target.dtype)

        # 1. Sagittal Δz (leg hangs down): dz = -sqrt(py² + pz² - d_y²)
        yz_sq = py * py + pz * pz
        dy_sq = d_y * d_y
        # Clamp to avoid negative under sqrt at workspace boundary
        under = torch.clamp(yz_sq - dy_sq, min=0.0)
        dz = -torch.sqrt(under)
        dx = px

        # 2. q_hip from complex-plane rotation: (py + i pz) = e^{i q_hip} · (d_y + i dz)
        q_hip = torch.atan2(pz, py) - torch.atan2(dz, d_y)
        # Wrap to [-π, π]
        q_hip = torch.atan2(torch.sin(q_hip), torch.cos(q_hip))

        # 3. q_calf from cosine rule (knee bends backward → q_calf ≤ 0)
        L1 = B1LegKinematics.L1
        L2 = B1LegKinematics.L2
        D = (dx * dx + dz * dz - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
        D = torch.clamp(D, min=-1.0, max=1.0)
        q_calf = -torch.acos(D)

        # 4. q_thigh from:  -a·sin(q_t) - b·cos(q_t) = dx
        #                   -a·cos(q_t) + b·sin(q_t) = dz
        #    where a = L1 + L2·cos(q_calf), b = L2·sin(q_calf).
        a = L1 + L2 * torch.cos(q_calf)
        b = L2 * torch.sin(q_calf)
        q_thigh = torch.atan2(-a * dx + b * dz, -b * dx - a * dz)

        return torch.stack([q_hip, q_thigh, q_calf], dim=-1)

    @staticmethod
    def is_reachable(
        foot_target: torch.Tensor,
        d_y: torch.Tensor | float,
        margin: float = 1e-4,
    ) -> torch.Tensor:
        """Boolean mask: True where the target is within the leg workspace.

        A point is reachable iff:
            - py² + pz² ≥ d_y²        (abduction circle is satisfiable)
            - sqrt(dx² + dz²) ≤ L1 + L2 - margin
        """
        px = foot_target[..., 0]
        py = foot_target[..., 1]
        pz = foot_target[..., 2]
        if not isinstance(d_y, torch.Tensor):
            d_y = torch.as_tensor(d_y, device=foot_target.device, dtype=foot_target.dtype)

        abd_ok = (py * py + pz * pz) >= (d_y * d_y)
        # Sagittal extension bound
        dz_sq = torch.clamp(py * py + pz * pz - d_y * d_y, min=0.0)
        planar = torch.sqrt(px * px + dz_sq)
        reach_ok = planar <= (B1LegKinematics.L1 + B1LegKinematics.L2 - margin)
        return abd_ok & reach_ok
