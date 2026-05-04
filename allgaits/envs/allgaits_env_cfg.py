"""Configuration for the AllGaits Isaac Lab environment.

Paper §II (AllGaits, Bellegarda et al. 2024) + CPG-RL Table I (Bellegarda &
Ijspeert 2022 RAL, ref [36]) for timing/PPO hyperparameters.

Control timing:
    physics      1000 Hz   (dt = 1e-3)
    policy        100 Hz   (decimation = 10)
    CPG / pattern 1000 Hz  (integrated once per physics step inside _apply_action)

Action space (8D): [μ_FL, μ_FR, μ_RL, μ_RR, ω_FL, ω_FR, ω_RL, ω_RR]
    μ ∈ [1, 2]   intrinsic oscillator amplitude target
    ω ∈ [0, 8]   intrinsic frequency, Hz

Observation space (full, 64D):
    velocity cmd    (3)   v*_x, v*_y, ω*_z
    body state      (9)   projected gravity (3) + body ang vel (3) + lin vel (3)
    joint state    (24)   joint pos (12) + joint vel (12)
    foot contacts   (4)   booleans
    last action     (8)
    CPG efference  (16)   r, ṙ, θ, θ̇   (4 values × 4 legs)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from allgaits.envs.b1_cfg import UNITREE_B1_ALLGAITS_CFG


# ---------------------------------------------------------------------------
# Episode / physics / control timing
# ---------------------------------------------------------------------------
_PHYSICS_DT: float = 1.0 / 1000.0   # 1 kHz
_DECIMATION: int = 10               # policy @ 100 Hz
_EPISODE_S: float = 20.0            # paper §II-C


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------
_ACTION_DIM: int = 8                # [μ(4), ω(4)]
# Observation layout (must match AllGaitsEnv._get_observations exactly):
#   3 (vel cmd) + 9 (body) + 12 (qp) + 12 (qv) + 4 (contacts) + 8 (last a) + 16 (CPG) = 64
_OBS_DIM: int = 64

# Paper action ranges
ACTION_MU_MIN: float = 1.0
ACTION_MU_MAX: float = 2.0
ACTION_OMEGA_HZ_MIN: float = 0.0
ACTION_OMEGA_HZ_MAX: float = 8.0


# ---------------------------------------------------------------------------
# Velocity command range (B1-adjusted from paper's [0.2, 3] m/s on Go1)
# ---------------------------------------------------------------------------
VEL_X_MIN: float = 0.2
VEL_X_MAX: float = 2.5       # extended from 1.5 — bound/gallop need 2.5+ m/s to develop
                              # aerial phases; B1 is heavier but longer legs (L1+L2=0.70m
                              # vs Go1's ~0.41m) give enough reach to operate at this speed
# Paper §II-C says v* every 5 s, but with our env (20 s episodes) that lets
# the policy "chase" 4 cmd changes per episode and learn cmd-derivatives
# rather than steady-state tracking. After v1+v2 trained policies tracked
# at training-time but drifted backward in play with FIXED cmd, we bumped
# this to match episode length: 1 cmd per episode → policy must hold a
# velocity for the full 20 s, matching deployment conditions.
VEL_RESAMPLE_S: float = 20.0


# ---------------------------------------------------------------------------
# Gait-coupling resampling
# ---------------------------------------------------------------------------
PHI_RESAMPLE_S: float = 3.0  # paper §II-C: new Φ every 3 s
# Paper trains with all 9 gaits in the pool; Phase A of our plan locks trot
# only; Phase B uses {walk, trot, pace}; Phase C uses all 9.
PHASE_A_GAITS: tuple[str, ...] = ("trot",)
PHASE_B_GAITS: tuple[str, ...] = ("walk", "trot", "pace")
PHASE_C_GAITS: tuple[str, ...] = (
    "walk", "amble", "trot", "pace", "bound", "pronk",
    "canter", "transverse_gallop", "rotary_gallop",
)


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------
# Threshold on base/thigh contact force.
#   1 N   — way too sensitive; 44% of envs died on step 1 (smoke test)
#  10 N   — masked the settling in training (via init_at_random_ep_len
#           staggering) but play.py showed every env still dying every
#           4 steps due to per-reset spike forces
#  50 N   — catches real falls (body-slam peaks at 300+ N on 62 kg B1)
#           while absorbing per-reset settling transients
TERMINATION_BASE_CONTACT_THRESHOLD_N: float = 50.0


# ---------------------------------------------------------------------------
# Reward weights (paper Table I — multiplied by control dt = 0.01 s internally)
# ---------------------------------------------------------------------------
REWARD_LIN_VEL_X_TRACKING: float = 8.0   # was 6.0 — boosted with sigma widening to restore gradient at realistic error (~0.3 m/s)
REWARD_LIN_VEL_YZ_PENALTY: float = 2.0
REWARD_ANG_VEL_XYZ_PENALTY: float = 0.35  # 2.0 killed all gaits (ω=0 collapse); 0.1 allows spinning; 0.35 suppresses runaway yaw while letting oscillation-induced transients pass
REWARD_POWER_PENALTY: float = 0.001
REWARD_TRACKING_SIGMA: float = 0.25   # widened from 0.15 — lin_vel_x_direct now prevents standstill basin, so sigma doesn't need to be aggressive; at 0.15 the gradient at typical 0.3 m/s error was ≈0, starving tracking signal
# Action-rate penalty is NOT in the paper's Table I, but it's required here
# because our Gaussian policy's learnable std would otherwise run away (the
# clamp in the action decoder has zero gradient outside the range). Gives
# the policy a direct cost for large/oscillating actions, which the paper's
# reward doesn't need because their setup presumably has tighter bounds or
# implicit regularization not stated.
#
# Tuning log:
#   0.01  — bounded std at 2.0, ep_len plateaued at 248 (too conservative)
#   0.003 — std ran to 4.1, 96% of actions clipped (too weak)
#   0.025 — target: std equilibrium ≈ 1.0 paired with entropy_coef=0.001
#           (math: σ² ≈ entropy_coef / (4·w·dt) = 0.001 / (4·0.025·0.01) = 1.0)
REWARD_ACTION_RATE_PENALTY: float = 0.025
# Direct linear velocity reward: gives constant nonzero gradient at standstill
# so the policy cannot collapse to ω=0 as a local optimum.  The Gaussian
# tracking term is flat when vel_x ≈ 0 (gradient ≈ 0.009 at standstill vs
# 0.075 near target), which lets "stand still + no penalties" beat "try to
# walk but risk falling."  Adding a linear term breaks that basin:
#   standstill: r_lin = 0
#   0.4 m/s:    r_lin = 2.0 × 0.4 × 0.01 = 0.008/step  (+ Gaussian on top)
REWARD_LIN_VEL_X_DIRECT: float = 2.0
# Heading alignment: cos(yaw error) extracted from root quaternion.
# 1.0 when facing world +x, 0.0 sideways, -1.0 backward. Gives a positive
# incentive to maintain forward heading, replacing the structural problem where
# ang_vel penalty alone suppressed pace/bound by punishing oscillation-induced
# yaw before the policy could correct it.
REWARD_HEADING: float = 3.0
# Doubled 1.5 → 3.0: heading reward at 1.5 was losing to the thigh-asymmetry
# yaw-spin torque — all 4 envs rotated ~90° in 18 s during play even though
# the policy was otherwise stable. 3.0 gives it enough signal to fight back.
# CPG activity: rewards mean leg frequency (Hz), providing a nonzero gradient
# at ω=0 so that the policy cannot treat "freeze all legs" as a local optimum
# for gaits whose instability penalties previously outweighed locomotion rewards.
REWARD_CPG_ACTIVE: float = 0.5   # reward ω linearly up to 3 Hz (natural stride range); paired with CPG_RUNAWAY penalty to prevent 6-8 Hz blowup seen in Phase B v5
REWARD_CPG_RUNAWAY: float = 2.0  # quadratic penalty on ω above 3 Hz — hard ceiling without removing gradient below it
# Foot slip: penalise the world-frame horizontal speed of any foot that is in
# contact with the ground.  contact × foot_speed_xy encourages the policy to
# place feet and hold them stationary rather than sliding.  Slip velocities of
# 1–3 m/s were observed during play (phase_b_dr_v1) on planted feet, causing
# poor velocity tracking and inefficient gait.
REWARD_FOOT_SLIP_PENALTY: float = 1.0


# ---------------------------------------------------------------------------
# Domain Randomization
# ---------------------------------------------------------------------------
# Push disturbances: every dr_push_interval_s, a random horizontal velocity
# impulse is applied to the base (Δv_xy + Δω_z). Forces the policy to learn
# active recovery rather than relying on stable steady-state gaits.
DR_PUSH_INTERVAL_S: float = 4.0     # seconds between pushes per env
DR_PUSH_LIN_VEL: float = 0.5        # max |Δv| in world xy, m/s
#   Paper (Go1, ~12 kg) applies ~30 N for 0.1 s → Δv ≈ 0.25 m/s.
#   B1 is 5× heavier and more stable, so 0.5 m/s is intentionally harder —
#   it can reverse direction at low cmd speeds, forcing real recovery learning.
DR_PUSH_ANG_VEL: float = 0.3        # max |Δω| around world z, rad/s
#   Reduced from 0.4: B1 already has systematic yaw drift from thigh asymmetry;
#   0.4 rad/s yaw push fights ang_vel penalty every push → noisy gradients.
# Joint-position noise: ± this many rad added to default joint pos at each
# episode reset. Prevents the policy from memorising a single rest pose.
#   Paper (Go1, stiffness ~200 N·m/rad) uses ±0.2 rad.
#   B1 stiffness = 600 N·m/rad → 0.2 × (200/600) ≈ 0.067 → 0.07 rad.
#   Larger noise would produce ~600 × 0.2 = 120 N·m per joint at reset,
#   ~340 N through the leg — enough to trip the 50 N base-contact threshold.
DR_JOINT_POS_NOISE: float = 0.07    # rad


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@configclass
class AllGaitsEnvCfg(DirectRLEnvCfg):
    """Top-level env config. Matches paper's §II-B and §II-C specifications."""

    # --- Episode ---
    episode_length_s: float = _EPISODE_S
    decimation: int = _DECIMATION

    # --- Spaces (fed to the RL framework) ---
    observation_space: int = _OBS_DIM
    action_space: int = _ACTION_DIM
    state_space: int = 0

    # --- Simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=_PHYSICS_DT,
        render_interval=_DECIMATION,
        # B1's larger contact surfaces overflow Go2's default patch budget
        # at >2000 envs (lesson from cpg-drl-transition). Doubling the
        # rigid-patch buffer from 10·2¹⁵ to 20·2¹⁵ avoids
        # "PhysX patch buffer overflow" warnings during training.
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=20 * 2**15,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # --- Scene ---
    # env_spacing=3.5 m: B1's footprint (~0.7 m × 1.0 m) is 1.7× Go2's, so
    # 2.5 m spacing risks neighbor-env contact at large num_envs.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.5, replicate_physics=True
    )

    # --- Terrain (flat) ---
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # --- Robot (AllGaits-tuned: stiffness 600, damping 15 — see b1_cfg.py) ---
    robot: ArticulationCfg = UNITREE_B1_ALLGAITS_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # --- Contact sensor (all bodies tracked; we filter by body-id in env) ---
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.0,      # every physics step
        track_air_time=True,
    )

    # --- CPG ---
    cpg_convergence_factor: float = 150.0   # CPG-RL §III-A
    cpg_coupling_strength: float = 10.0     # AllGaits §II-A ("w_ij = 10")
    cpg_d_step: float = 0.15                # max step length, m (CPG-RL Fig 3)

    # --- Active gait pool (swap per phase) ---
    active_gaits: tuple[str, ...] = PHASE_A_GAITS    # default: Phase A (trot only)

    # --- Style parameter sampling ranges (B1-adjusted; see pattern.B1_STYLE_PARAM_RANGES) ---
    h_range: tuple[float, float] = (0.27, 0.52)
    g_c_range: tuple[float, float] = (0.03, 0.18)
    g_p_range: tuple[float, float] = (0.0, 0.022)
    x_off_range: tuple[float, float] = (-0.12, 0.04)

    # --- Velocity command range ---
    vel_x_range: tuple[float, float] = (VEL_X_MIN, VEL_X_MAX)
    vel_y_range: tuple[float, float] = (0.0, 0.0)    # forward-only for Phase A/B
    yaw_rate_range: tuple[float, float] = (0.0, 0.0)

    # --- Termination ---
    base_contact_threshold_n: float = TERMINATION_BASE_CONTACT_THRESHOLD_N

    # --- Reward weights ---
    rew_lin_vel_x_tracking: float = REWARD_LIN_VEL_X_TRACKING
    rew_lin_vel_x_direct: float = REWARD_LIN_VEL_X_DIRECT
    rew_heading: float = REWARD_HEADING
    rew_cpg_active: float = REWARD_CPG_ACTIVE
    rew_cpg_runaway: float = REWARD_CPG_RUNAWAY
    rew_foot_slip_penalty: float = REWARD_FOOT_SLIP_PENALTY
    rew_lin_vel_yz_penalty: float = REWARD_LIN_VEL_YZ_PENALTY
    rew_ang_vel_xyz_penalty: float = REWARD_ANG_VEL_XYZ_PENALTY
    rew_power_penalty: float = REWARD_POWER_PENALTY
    rew_tracking_sigma: float = REWARD_TRACKING_SIGMA
    rew_action_rate_penalty: float = REWARD_ACTION_RATE_PENALTY

    # --- Resampling periods ---
    vel_resample_s: float = VEL_RESAMPLE_S
    phi_resample_s: float = PHI_RESAMPLE_S

    # --- Domain Randomization ---
    dr_push_interval_s: float = DR_PUSH_INTERVAL_S
    dr_push_lin_vel: float = DR_PUSH_LIN_VEL
    dr_push_ang_vel: float = DR_PUSH_ANG_VEL
    dr_joint_pos_noise: float = DR_JOINT_POS_NOISE
