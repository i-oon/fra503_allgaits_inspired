"""AllGaits DirectRLEnv for Unitree B1 in Isaac Lab.

Wires the Hopf CPG + pattern-formation layer into an Isaac Lab DirectRLEnv.
Action = [μ, ω] ∈ ℝ⁸ modulates the oscillators (paper §II-B). The coupling
matrix Φ is sampled per-env from the config's `active_gaits` pool and is
NOT part of the action — the policy infers the gait from its CPG efference
observation (paper's key design: the policy doesn't see Φ directly).

Leg order conventions:
    CPG module / pattern formation: [FL, FR, RL, RR]  (see allgaits/__init__.py)
    URDF / Isaac Lab joint names:   [FR_hip, FL_hip, RR_hip, RL_hip, ...] (USD order)

A permutation `self._cpg_to_usd_joint_idx` maps CPG-order joint targets to
the USD joint indices used by `set_joint_position_target`.
"""

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim import SimulationCfg  # noqa: F401  (kept for typing clarity)
import isaaclab.sim as sim_utils

from allgaits import LEG_ORDER
from allgaits.cpg.coupling import phase_offset_matrix, weight_matrix
from allgaits.cpg.hopf import HopfCPG
from allgaits.cpg.pattern import PatternFormation
from allgaits.envs.allgaits_env_cfg import (
    ACTION_MU_MAX,
    ACTION_MU_MIN,
    ACTION_OMEGA_HZ_MAX,
    ACTION_OMEGA_HZ_MIN,
    AllGaitsEnvCfg,
    REWARD_LIN_VEL_X_DIRECT,
    REWARD_HEADING,      # noqa: F401
    REWARD_CPG_ACTIVE,   # noqa: F401
    REWARD_CPG_RUNAWAY,  # noqa: F401
)


_CPG_JOINT_NAMES: list[str] = [
    f"{leg}_{jt}" for leg in LEG_ORDER for jt in ("hip_joint", "thigh_joint", "calf_joint")
]

# Body-name patterns for contact termination
# Body-name patterns for the contact sensor. Anchored with `$` so we don't
# match suffix variants (e.g. ".*_foot" would also match ".*_foot_rotor"
# if any USD had one). These patterns are resolved against the contact
# sensor's body list, NOT the articulation's — the two can differ in
# order, and `net_forces_w_history` is indexed by the sensor's list.
_BASE_THIGH_PATTERNS: tuple[str, ...] = (".*trunk$", ".*_thigh$", "base$")
_FOOT_PATTERN: str = ".*_foot$"


class AllGaitsEnv(DirectRLEnv):
    """Quadruped locomotion env implementing the AllGaits framework."""

    cfg: AllGaitsEnvCfg

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def __init__(self, cfg: AllGaitsEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        print("[AllGaitsEnv] super().__init__() starting…", flush=True)
        super().__init__(cfg, render_mode, **kwargs)
        print("[AllGaitsEnv] super().__init__() done, articulation + sensor ready.", flush=True)

        # --- Joint-index permutations (CPG order → USD order) ---
        joint_ids, _ = self._robot.find_joints(_CPG_JOINT_NAMES, preserve_order=True)
        self._cpg_to_usd_joint_idx = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        print(f"[AllGaitsEnv] articulation body_names:          {list(self._robot.body_names)}", flush=True)
        print(f"[AllGaitsEnv] contact-sensor body_names:        {list(self._contact_sensor.body_names)}", flush=True)

        # --- Body indices into the CONTACT SENSOR's body list (not the
        # articulation's — they can differ in order). `net_forces_w_history`
        # is shaped (N, H, sensor_B, 3) and is indexed by the sensor's list.
        base_thigh_ids, base_thigh_names = self._contact_sensor.find_bodies(list(_BASE_THIGH_PATTERNS))
        foot_ids, foot_names = self._contact_sensor.find_bodies(_FOOT_PATTERN)
        self._base_thigh_body_ids = torch.tensor(base_thigh_ids, device=self.device, dtype=torch.long)
        self._foot_body_ids = torch.tensor(foot_ids, device=self.device, dtype=torch.long)
        if self._foot_body_ids.numel() != 4:
            raise RuntimeError(
                f"Expected 4 foot bodies matching {_FOOT_PATTERN!r}, got "
                f"{self._foot_body_ids.numel()}: {foot_names}"
            )
        print(f"[AllGaitsEnv] matched foot bodies:       {list(foot_names)}  ids={foot_ids}", flush=True)
        print(f"[AllGaitsEnv] matched base/thigh bodies: {list(base_thigh_names)}  ids={base_thigh_ids}", flush=True)

        # --- CPG and pattern formation ---
        self._cpg = HopfCPG(
            num_envs=self.num_envs,
            num_legs=4,
            convergence_factor=self.cfg.cpg_convergence_factor,
            dt=self.cfg.sim.dt,
            device=self.device,
            dtype=torch.float32,
        )
        self._pattern = PatternFormation(
            d_step_default=self.cfg.cpg_d_step,
            device=self.device,
            dtype=torch.float32,
        )

        # --- Per-env coupling matrices (shape (N, 4, 4)), sampled in reset ---
        self._phi = torch.zeros(self.num_envs, 4, 4, device=self.device)
        self._w_couple = weight_matrix(
            num_legs=4, strength=self.cfg.cpg_coupling_strength, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1, -1).contiguous()

        # --- Per-env style parameters (resampled each reset) ---
        self._h_per_env = torch.zeros(self.num_envs, 1, device=self.device)
        self._g_c_per_env = torch.zeros(self.num_envs, 1, device=self.device)
        self._g_p_per_env = torch.zeros(self.num_envs, 1, device=self.device)
        self._x_off_per_env = torch.zeros(self.num_envs, 1, device=self.device)

        # --- Per-env velocity command (resampled every vel_resample_s) ---
        self._vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)   # v*_x, v*_y, ω*_z
        self._steps_since_vel_resample = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._steps_since_phi_resample = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._vel_resample_steps: int = max(1, int(self.cfg.vel_resample_s / (self.cfg.sim.dt * self.cfg.decimation)))
        self._phi_resample_steps: int = max(1, int(self.cfg.phi_resample_s / (self.cfg.sim.dt * self.cfg.decimation)))

        # --- Action / joint-target buffers ---
        self._mu = torch.ones(self.num_envs, 4, device=self.device)
        self._omega = torch.zeros(self.num_envs, 4, device=self.device)   # rad/s
        self._last_action = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        # Previous policy-step action, updated in _pre_physics_step before _last_action is overwritten.
        # Used by the action-rate reward term to penalize jerky / runaway actions.
        self._prev_action = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._joint_targets_cpg_order = torch.zeros(self.num_envs, 12, device=self.device)

        # --- Initial reset across all envs ---
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_velocity_command(all_ids)
        self._resample_coupling_matrix(all_ids)
        self._resample_style_params(all_ids)

    # ------------------------------------------------------------------
    # Action processing  (policy output → μ, ω)
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Decode raw (N, 8) action → μ ∈ [1, 2], ω_rads = 2π · Hz ∈ [0, 8 Hz].

        Decoding: **linear scale + clamp**, policy output centered at 0.
            policy out = 0  → physical mid-range      (μ=1.5, ω=4 Hz)
            policy out = ±1 → physical min / max      (μ=1 or 2, ω=0 or 8 Hz)
            |out| > 1       → clamped to physical range

        Why not tanh: a prior tanh-squash version had a saturation dead zone
        (policy output beyond ±3 saturated tanh), so Gaussian action-noise
        std had no gradient cost and ran away to 22 during training. With
        linear-scale-and-clamp the clamp edge has a real gradient penalty,
        keeping the policy's learned noise std in [0.3, 1.0].
        """
        mu_half = 0.5 * (ACTION_MU_MAX - ACTION_MU_MIN)
        mu_mid = 0.5 * (ACTION_MU_MIN + ACTION_MU_MAX)
        self._mu = (mu_mid + mu_half * actions[:, :4]).clamp(ACTION_MU_MIN, ACTION_MU_MAX)

        omega_half = 0.5 * (ACTION_OMEGA_HZ_MAX - ACTION_OMEGA_HZ_MIN)
        omega_mid = 0.5 * (ACTION_OMEGA_HZ_MIN + ACTION_OMEGA_HZ_MAX)
        omega_hz = (omega_mid + omega_half * actions[:, 4:]).clamp(
            ACTION_OMEGA_HZ_MIN, ACTION_OMEGA_HZ_MAX
        )
        self._omega = 2.0 * math.pi * omega_hz

        # Save previous action BEFORE overwriting — used by action-rate penalty.
        self._prev_action = self._last_action.clone()
        self._last_action = actions.clone()

    def _apply_action(self) -> None:
        """Called `decimation` times per policy step; integrates CPG + sets joint PD target."""
        self._cpg.step(self._mu, self._omega, self._phi, self._w_couple)

        foot_target, joint_target_cpg_order = self._pattern(
            self._cpg.r,
            self._cpg.theta,
            h=self._h_per_env,
            g_c=self._g_c_per_env,
            g_p=self._g_p_per_env,
            x_off=self._x_off_per_env,
        )
        # Flatten (N, 4, 3) → (N, 12) in canonical leg order.
        self._joint_targets_cpg_order = joint_target_cpg_order.reshape(self.num_envs, 12)

        # Map CPG-order joints to USD joint indices expected by set_joint_position_target.
        self._robot.set_joint_position_target(
            self._joint_targets_cpg_order,
            joint_ids=self._cpg_to_usd_joint_idx,
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        body = torch.cat(
            [
                self._robot.data.projected_gravity_b,   # (N, 3)
                self._robot.data.root_ang_vel_b,        # (N, 3)
                self._robot.data.root_lin_vel_b,        # (N, 3)
            ],
            dim=-1,
        )
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        foot_contacts = self._foot_contact_booleans()
        cpg_eff = self._cpg.efference_copy()
        cpg_cat = torch.cat(
            [cpg_eff["r"], cpg_eff["r_dot"], cpg_eff["theta"], cpg_eff["theta_dot"]],
            dim=-1,
        )

        obs = torch.cat(
            [
                self._vel_cmd,           # (N, 3)
                body,                    # (N, 9)
                joint_pos_rel,           # (N, 12)
                self._robot.data.joint_vel,  # (N, 12)
                foot_contacts,           # (N, 4)
                self._last_action,       # (N, 8)
                cpg_cat,                 # (N, 16)
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _foot_contact_booleans(self) -> torch.Tensor:
        """(N, 4) 0/1 tensor — one per leg (canonical order)."""
        # net_forces_w_history: (N, H, B, 3) where B is num bodies, H is history length
        forces = self._contact_sensor.data.net_forces_w_history[:, :, self._foot_body_ids]
        contact_any = (torch.norm(forces, dim=-1) > 1.0).any(dim=1)  # (N, 4)
        return contact_any.float()

    # ------------------------------------------------------------------
    # Reward  (paper Table I) + full diagnostics logging
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        dt = self.step_dt

        # --- Per-term reward components ---
        # World-frame x velocity: the command v*_x is a world-frame target.
        # Body-frame tracking (root_lin_vel_b) lets a yawing robot satisfy the
        # reward while spinning in place — confirmed in play (world_dx ≈ 0,
        # bodyX→w rotated 140° over 20 s). World-frame tracking forces the
        # policy to actually move in the world's +x direction.
        vel_err_x = self._vel_cmd[:, 0] - self._robot.data.root_lin_vel_w[:, 0]
        lin_vel_x_track = torch.exp(-(vel_err_x.pow(2)) / self.cfg.rew_tracking_sigma)

        lin_vel_yz = self._robot.data.root_lin_vel_b[:, 1:3].pow(2).sum(dim=-1)
        ang_vel_xyz = self._robot.data.root_ang_vel_b.pow(2).sum(dim=-1)

        torque = self._robot.data.applied_torque
        joint_vel = self._robot.data.joint_vel
        power = (torque * joint_vel).sum(dim=-1).abs()

        action_rate = (self._last_action - self._prev_action).pow(2).sum(dim=-1)

        # Direct linear term: rewards any positive world-frame forward velocity,
        # giving a nonzero gradient even at standstill where the Gaussian term
        # is nearly flat (gradient ≈ 0.009 vs 0.075 near the target).
        lin_vel_x_direct = self._robot.data.root_lin_vel_w[:, 0].clamp(min=0.0)

        # Heading alignment: cos(yaw error) from quaternion (wxyz convention).
        # 1 - 2*(qy² + qz²) equals the body +x axis projected onto world +x.
        # Gives positive gradient to face forward; replaces structural reliance
        # on ang_vel penalty, which suppressed pace/bound by punishing
        # oscillation-induced yaw before the policy could correct it.
        quat = self._robot.data.root_quat_w  # (N, 4) wxyz
        heading_cos = 1.0 - 2.0 * (quat[:, 2].pow(2) + quat[:, 3].pow(2))

        # CPG activity: reward ω linearly up to 3 Hz (natural stride range) then
        # penalise quadratically above 3 Hz. The two-sided shape gives a nonzero
        # gradient at ω=0 (breaks freeze basin) while hard-capping the incentive
        # so the policy cannot run away to 6-8 Hz (Phase B v5 failure mode).
        omega_hz = self._omega / (2.0 * math.pi)  # (N, 4)
        cpg_active = omega_hz.clamp(max=3.0).mean(dim=-1)           # (N,) reward part
        cpg_runaway = (omega_hz - 3.0).clamp(min=0.0).pow(2).mean(dim=-1)  # (N,) penalty part

        # --- Weighted contributions (× dt) ---
        r_track = self.cfg.rew_lin_vel_x_tracking * lin_vel_x_track * dt
        r_lin_direct = self.cfg.rew_lin_vel_x_direct * lin_vel_x_direct * dt
        r_heading = self.cfg.rew_heading * heading_cos * dt
        r_cpg_active = self.cfg.rew_cpg_active * cpg_active * dt
        r_cpg_runaway = -self.cfg.rew_cpg_runaway * cpg_runaway * dt
        r_yz_pen = -self.cfg.rew_lin_vel_yz_penalty * lin_vel_yz * dt
        r_ang_pen = -self.cfg.rew_ang_vel_xyz_penalty * ang_vel_xyz * dt
        r_power_pen = -self.cfg.rew_power_penalty * power * dt
        r_action_pen = -self.cfg.rew_action_rate_penalty * action_rate * dt

        reward = (r_track + r_lin_direct + r_heading + r_cpg_active + r_cpg_runaway
                  + r_yz_pen + r_ang_pen + r_power_pen + r_action_pen)

        # --- Diagnostic log (averaged over envs by the runner) ---
        self._populate_extras_log(
            r_track=r_track,
            r_lin_direct=r_lin_direct,
            r_heading=r_heading,
            r_cpg_active=r_cpg_active,
            r_cpg_runaway=r_cpg_runaway,
            r_yz_pen=r_yz_pen,
            r_ang_pen=r_ang_pen,
            r_power_pen=r_power_pen,
            r_action_pen=r_action_pen,
            reward=reward,
            action_rate_raw=action_rate,
        )
        return reward

    def _populate_extras_log(
        self,
        r_track: torch.Tensor,
        r_lin_direct: torch.Tensor,
        r_heading: torch.Tensor,
        r_cpg_active: torch.Tensor,
        r_cpg_runaway: torch.Tensor,
        r_yz_pen: torch.Tensor,
        r_ang_pen: torch.Tensor,
        r_power_pen: torch.Tensor,
        r_action_pen: torch.Tensor,
        reward: torch.Tensor,
        action_rate_raw: torch.Tensor,
    ) -> None:
        """Fill `self.extras["episode"]` with per-step diagnostics.

        RSL-RL's OnPolicyRunner reads `infos["episode"]` from each env step,
        accumulates it across the rollout, and both prints means to stdout
        (as `Mean episode <key>`) AND writes them to TensorBoard (under
        `Episode/<key>`). Keys here use `/` separators so TensorBoard groups
        them (rew/, locomotion/, action/).
        """
        vel_x_cmd = self._vel_cmd[:, 0]
        vel_x_actual = self._robot.data.root_lin_vel_w[:, 0]
        body_height = self._robot.data.root_pos_w[:, 2]
        proj_grav_xy = self._robot.data.projected_gravity_b[:, :2].norm(dim=-1)   # tilt ≈ 0 when upright

        # Action diagnostics (raw policy actions, pre-decode).
        raw_abs = self._last_action.abs()
        clipped_frac = (raw_abs > 1.0).float().mean(dim=-1)   # per-env frac of dims in clamp zone
        omega_hz = self._omega / (2.0 * math.pi)

        # All values stored as (N,) tensors so the runner can average
        # over time and envs correctly.
        self.extras["episode"] = {
            # --- Per-term reward contributions (already × dt) ---
            "rew/lin_vel_x_track": r_track,
            "rew/lin_vel_x_direct": r_lin_direct,
            "rew/heading": r_heading,
            "rew/cpg_active": r_cpg_active,
            "rew/cpg_runaway": r_cpg_runaway,
            "rew/lin_vel_yz_penalty": r_yz_pen,
            "rew/ang_vel_xyz_penalty": r_ang_pen,
            "rew/power_penalty": r_power_pen,
            "rew/action_rate_penalty": r_action_pen,
            "rew/total": reward,
            # --- Locomotion diagnostics ---
            "locomotion/vel_x_cmd": vel_x_cmd,
            "locomotion/vel_x_actual": vel_x_actual,
            "locomotion/vel_x_error": (vel_x_cmd - vel_x_actual),
            "locomotion/height": body_height,
            "locomotion/tilt_xy": proj_grav_xy,   # 0 upright, ≈1 on side, 2 upside-down
            # --- Action diagnostics ---
            "action/raw_mean_abs": raw_abs.mean(dim=-1),
            "action/raw_clipped_frac": clipped_frac,
            "action/mu_mean": self._mu.mean(dim=-1),
            "action/omega_hz_mean": omega_hz.mean(dim=-1),
            "action/rate_raw": action_rate_raw,   # unweighted ||Δa||²
        }

    # ------------------------------------------------------------------
    # Termination  (paper §II-C: reset on base/thigh ground contact)
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Any base/thigh body with contact force > threshold at any point in history
        forces = self._contact_sensor.data.net_forces_w_history[:, :, self._base_thigh_body_ids]
        force_mag = torch.norm(forces, dim=-1)                         # (N, H, B)
        max_force = force_mag.amax(dim=(1, 2))                         # (N,)
        fallen = max_force > self.cfg.base_contact_threshold_n

        # Termination diagnostics: 1.0 on envs ending this step.
        if "episode" in self.extras:
            self.extras["episode"]["termination/fall_rate"] = fallen.float()
            self.extras["episode"]["termination/timeout_rate"] = time_out.float()
            self.extras["episode"]["termination/any_rate"] = (fallen | time_out).float()
            # Peak contact force on base/thigh — helps tune the threshold.
            self.extras["episode"]["termination/max_base_thigh_force_N"] = max_force

        return fallen, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # --- Robot state: default pose at env origin, zero velocities ---
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        # UNITREE_B1_CFG spawns trunk at z=0.42, but with default joints
        # (q_hip=0.1, q_thigh=0.8/1.0, q_calf=-1.5) the feet land at
        # world z ≈ -0.077 m — 7.7 cm below ground. PhysX resolves this
        # with a violent settling push that transmits forces through
        # thighs and can trip the base-contact termination. Offset the
        # spawn z by the FK-computed foot-below-hip distance (~0.08 m)
        # so feet start just above ground and settle cleanly.
        default_root_state[:, 2] += 0.08
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # --- Per-env CPG/style/velocity resets ---
        self._cpg.reset(env_ids=env_ids)
        self._resample_velocity_command(env_ids)
        self._resample_coupling_matrix(env_ids)
        self._resample_style_params(env_ids)
        self._last_action[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._steps_since_vel_resample[env_ids] = 0
        self._steps_since_phi_resample[env_ids] = 0

    def _resample_velocity_command(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        vx = self._uniform(n, self.cfg.vel_x_range)
        vy = self._uniform(n, self.cfg.vel_y_range)
        wz = self._uniform(n, self.cfg.yaw_rate_range)
        self._vel_cmd[env_ids, 0] = vx
        self._vel_cmd[env_ids, 1] = vy
        self._vel_cmd[env_ids, 2] = wz

    def _resample_coupling_matrix(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        gaits = self.cfg.active_gaits
        idx = torch.randint(0, len(gaits), (n,), device=self.device)
        new_phi = torch.stack(
            [phase_offset_matrix(gaits[i.item()], device=self.device) for i in idx]
        )
        self._phi[env_ids] = new_phi

    def _resample_style_params(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        self._h_per_env[env_ids, 0] = self._uniform(n, self.cfg.h_range)
        self._g_c_per_env[env_ids, 0] = self._uniform(n, self.cfg.g_c_range)
        self._g_p_per_env[env_ids, 0] = self._uniform(n, self.cfg.g_p_range)
        self._x_off_per_env[env_ids, 0] = self._uniform(n, self.cfg.x_off_range)

    def _uniform(self, n: int, r: tuple[float, float]) -> torch.Tensor:
        lo, hi = r
        return lo + (hi - lo) * torch.rand(n, device=self.device)

    # ------------------------------------------------------------------
    # Periodic (per-step) resampling of v* and Φ
    # ------------------------------------------------------------------
    def _post_step_resampling(self) -> None:
        """Paper §II-C: re-sample v* every 5 s, Φ every 3 s."""
        self._steps_since_vel_resample += 1
        self._steps_since_phi_resample += 1

        vel_ids = (self._steps_since_vel_resample >= self._vel_resample_steps).nonzero(as_tuple=False).flatten()
        if vel_ids.numel() > 0:
            self._resample_velocity_command(vel_ids)
            self._steps_since_vel_resample[vel_ids] = 0

        phi_ids = (self._steps_since_phi_resample >= self._phi_resample_steps).nonzero(as_tuple=False).flatten()
        if phi_ids.numel() > 0:
            self._resample_coupling_matrix(phi_ids)
            self._steps_since_phi_resample[phi_ids] = 0

    # DirectRLEnv calls _get_rewards → _get_dones → _reset_idx each step.
    # We hook into post-step to drive the periodic resampling.
    def step(self, action: torch.Tensor):
        obs, rew, terminated, truncated, extras = super().step(action)
        self._post_step_resampling()
        return obs, rew, terminated, truncated, extras
