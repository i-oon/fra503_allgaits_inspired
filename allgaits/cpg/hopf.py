"""Hopf-like CPG with inter-oscillator coupling (AllGaits eqs 1-2).

State per oscillator i:
    r_i     amplitude
    r_dot_i amplitude velocity
    theta_i phase (rad, wrapped to [0, 2π))

Dynamics (continuous-time, paper eqs 1-2):
    r̈_i     = a · (a/4 · (μ_i − r_i) − ṙ_i)
    θ̇_i     = ω_i + Σ_j r_j · w_ij · sin(θ_j − θ_i − φ_ij)

Integrated with forward Euler at `dt` (paper uses 1 kHz → dt=1e-3).

All tensors are batched: leading dim is num_envs. Leg ordering is
[FL, FR, RL, RR] (see allgaits/__init__.py).
"""

from __future__ import annotations

import math

import torch


class HopfCPG:
    """Batched Hopf oscillator bank with inter-leg coupling.

    Args:
        num_envs: parallel environments.
        num_legs: oscillators per env (4 for a quadruped).
        convergence_factor: `a` in paper eq 1. Larger → faster amplitude
            convergence to μ. Paper is silent on the exact value; a=150
            (common in Bellegarda's prior CPG-RL work) gives critically
            damped convergence within ~100 ms.
        dt: integration timestep in seconds (paper: 1e-3).
        device, dtype: torch placement.
    """

    def __init__(
        self,
        num_envs: int,
        num_legs: int = 4,
        convergence_factor: float = 150.0,
        dt: float = 1e-3,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_envs = num_envs
        self.num_legs = num_legs
        self.a = float(convergence_factor)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.dtype = dtype

        # State packed as (num_envs, num_legs, 3): [r, r_dot, theta]
        self._state = torch.zeros(num_envs, num_legs, 3, device=self.device, dtype=self.dtype)
        self.reset()

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------
    @property
    def r(self) -> torch.Tensor:
        """Amplitude, shape (num_envs, num_legs)."""
        return self._state[..., 0]

    @property
    def r_dot(self) -> torch.Tensor:
        """Amplitude velocity, shape (num_envs, num_legs)."""
        return self._state[..., 1]

    @property
    def theta(self) -> torch.Tensor:
        """Phase in radians (wrapped [0, 2π)), shape (num_envs, num_legs)."""
        return self._state[..., 2]

    @property
    def state(self) -> torch.Tensor:
        """Full state, shape (num_envs, num_legs, 3)."""
        return self._state

    def efference_copy(self) -> dict[str, torch.Tensor]:
        """CPG state feedback for the policy observation.

        Returns dict with keys {r, r_dot, theta, theta_dot}, each of shape
        (num_envs, num_legs). `theta_dot` is derived from the last step and
        is 0 immediately after reset (see `step` for update).
        """
        return {
            "r": self.r.clone(),
            "r_dot": self.r_dot.clone(),
            "theta": self.theta.clone(),
            "theta_dot": self._last_theta_dot.clone(),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset oscillator state to near-unit-amplitude with random phases.

        Starting `r` slightly off 1.0 (drawn from U[0.9, 1.1]) and `r_dot=0`
        lets the limit cycle develop naturally from the first step. Random
        θ ∈ [0, 2π) prevents synchronization artefacts across envs.
        """
        if env_ids is None:
            idx = slice(None)
            n = self.num_envs
        else:
            idx = env_ids
            n = env_ids.numel() if isinstance(env_ids, torch.Tensor) else len(env_ids)

        r0 = 0.9 + 0.2 * torch.rand(n, self.num_legs, device=self.device, dtype=self.dtype)
        theta0 = 2 * math.pi * torch.rand(n, self.num_legs, device=self.device, dtype=self.dtype)

        self._state[idx, :, 0] = r0
        self._state[idx, :, 1] = 0.0
        self._state[idx, :, 2] = theta0

        if not hasattr(self, "_last_theta_dot"):
            self._last_theta_dot = torch.zeros(
                self.num_envs, self.num_legs, device=self.device, dtype=self.dtype
            )
        else:
            self._last_theta_dot[idx] = 0.0

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    def step(
        self,
        mu: torch.Tensor,
        omega: torch.Tensor,
        phi_offsets: torch.Tensor,
        coupling_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward-Euler integrate one `dt`.

        Args:
            mu: intrinsic amplitude target, shape (num_envs, num_legs).
            omega: intrinsic frequency in rad/s, shape (num_envs, num_legs).
            phi_offsets: target phase offset matrix φ_ij with row=i, col=j.
                Shape (num_legs, num_legs) or (num_envs, num_legs, num_legs).
            coupling_weights: coupling-strength matrix w_ij.
                Same shape semantics as `phi_offsets`.

        Returns:
            Updated state tensor, shape (num_envs, num_legs, 3).
        """
        r, r_dot, theta = self._state.unbind(-1)

        # Amplitude: r̈ = a(a/4 (μ − r) − ṙ)
        r_ddot = self.a * (self.a * 0.25 * (mu - r) - r_dot)

        # Phase: θ̇_i = ω_i + Σ_j r_j · w_ij · sin(θ_j − θ_i − φ_ij)
        # Broadcasting to pairwise tensors of shape (..., num_legs, num_legs):
        theta_i = theta.unsqueeze(-1)  # (N, L, 1) — row index i
        theta_j = theta.unsqueeze(-2)  # (N, 1, L) — col index j
        r_j = r.unsqueeze(-2)          # (N, 1, L)
        coupling = r_j * coupling_weights * torch.sin(theta_j - theta_i - phi_offsets)
        theta_dot = omega + coupling.sum(dim=-1)

        # Forward-Euler
        r_new = r + r_dot * self.dt
        r_dot_new = r_dot + r_ddot * self.dt
        theta_new = (theta + theta_dot * self.dt) % (2.0 * math.pi)

        self._state = torch.stack([r_new, r_dot_new, theta_new], dim=-1)
        self._last_theta_dot = theta_dot.detach().clone()
        return self._state
