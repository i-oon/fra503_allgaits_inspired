# Adding Unitree B1 Robot to Isaac Lab

**Author:** Disthorn Suttawet  
**Date:** April 2026  
**Isaac Lab Version:** 0.36.3  
**Isaac Sim Version:** 4.5.0

---

## Overview

This guide documents the process of adding Unitree B1 quadruped robot to Isaac Lab simulation environment. The B1 model is not included in Isaac Lab by default, so we convert it from the official Unitree URDF files.

**What you'll get:**
- ✅ Official Unitree B1 geometry and physics
- ✅ Proper joint configuration (12 DOF quadruped)
- ✅ DC motor actuator model (23.7 N·m, 30 rad/s)
- ✅ Ready-to-use `UNITREE_B1_CFG` in Isaac Lab

**Time required:** ~15 minutes

---

## Prerequisites

- Isaac Lab 0.36.3+ installed
- Isaac Sim 4.5+
- Conda environment activated: `conda activate env_isaaclab`
- Git installed

---

## Step 1: Obtain Official B1 URDF Files

Clone Unitree's official ROS repository:

```bash
cd ~/Downloads
git clone https://github.com/unitreerobotics/unitree_ros
cd unitree_ros/robots/b1_description
```

Verify files:
```bash
ls meshes/
# Expected output: calfb.dae, hipb.dae, thighb.dae, trunkb.dae, etc.

ls xacro/b1.urdf
# Expected output: xacro/b1.urdf
```

---

## Step 2: Fix URDF Mesh Paths

URDF files use ROS-specific `package://` paths. Convert to absolute file paths:

```bash
cd ~/Downloads/unitree_ros/robots/b1_description

# Create fixed version
cp xacro/b1.urdf xacro/b1_fixed.urdf

# Replace package paths with absolute paths
MESHDIR=$(pwd)/meshes
sed -i "s|package://b1_description/meshes|file://${MESHDIR}|g" xacro/b1_fixed.urdf
```

**Verify the fix:**
```bash
grep "file://" xacro/b1_fixed.urdf | head -3
```

Expected output:
```
<mesh filename="file:///home/USERNAME/Downloads/unitree_ros/robots/b1_description/meshes/trunkb.dae" scale="1 1 1"/>
<mesh filename="file:///home/USERNAME/Downloads/unitree_ros/robots/b1_description/meshes/trunkb.dae" scale="1 1 1"/>
<mesh filename="file:///home/USERNAME/Downloads/unitree_ros/robots/b1_description/meshes/hipb.dae" scale="1 1 1"/>
```

---

## Step 3: Convert URDF to USD

Use Isaac Lab's built-in URDF converter:

```bash
cd ~/IsaacLab

python scripts/tools/convert_urdf.py \
    ~/Downloads/unitree_ros/robots/b1_description/xacro/b1_fixed.urdf \
    ~/Downloads/b1_usd/b1.usd \
    --joint-stiffness 25.0 \
    --joint-damping 0.5 \
    --headless
```

**Expected output:**
```
--------------------------------------------------------------------------------
Input URDF file: /home/USERNAME/Downloads/.../b1_fixed.urdf
URDF importer config:
    asset_path: .../b1_fixed.urdf
    joint_drive:
        gains:
            stiffness: 25.0
            damping: 0.5
--------------------------------------------------------------------------------
URDF importer output:
Generated USD file: /home/USERNAME/Downloads/b1_usd/b1.usd
--------------------------------------------------------------------------------
```

**Notes:**
- Warnings about "No mass specified for link base" are normal
- Conversion takes ~30-60 seconds
- Output USD file should be ~1-2 KB

---

## Step 4: Copy USD to Isaac Lab Assets

```bash
# Create B1 directory
mkdir -p ~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1

# Copy USD file
cp ~/Downloads/b1_usd/b1.usd \
   ~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/

# Verify
ls -lh ~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/
```

Expected output:
```
-rw-rw-r-- 1 user user 1.6K Apr 12 16:47 b1.usd
```

---

## Step 5: Add B1 Configuration to Isaac Lab

Edit the Unitree robot definitions:

```bash
cd ~/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots
nano unitree.py
```

### 5.1: Add import (if not present)

At the top of the file, ensure `os` is imported:

```python
import os
```

### 5.2: Add B1 configuration

At the bottom of the file (after `G1_MINIMAL_CFG`), add:

```python


UNITREE_B1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.expanduser('~')}/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/b1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),  # B1 standing height (meters)
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.7,      # B1 spec: 23.7 N·m per joint
            saturation_effort=23.7,
            velocity_limit=30.0,    # B1 spec: 30 rad/s
            stiffness=25.0,         # PD control stiffness
            damping=0.5,            # PD control damping
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree B1 using converted USD from official URDF.

Reference: https://github.com/unitreerobotics/unitree_ros
Motor specs: 23.7 N·m max torque, 30 rad/s max velocity
"""
```

**Save and exit:** `Ctrl+X`, `Y`, `Enter`

---

## Step 6: Verify Installation

Create a test script `test_b1.py`:

```python
"""Test Unitree B1 configuration."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args)
sim_app = app.app

# Import B1 configuration
from isaaclab_assets.robots.unitree import UNITREE_B1_CFG

print("\n" + "="*70)
print("Unitree B1 Configuration Test")
print("="*70)
print(f"✓ USD path: {UNITREE_B1_CFG.spawn.usd_path}")
print(f"✓ Initial height: {UNITREE_B1_CFG.init_state.pos[2]} m")
print(f"✓ Actuators: {list(UNITREE_B1_CFG.actuators.keys())}")
print(f"✓ Motor effort limit: {UNITREE_B1_CFG.actuators['base_legs'].effort_limit} N·m")
print(f"✓ Motor velocity limit: {UNITREE_B1_CFG.actuators['base_legs'].velocity_limit} rad/s")
print("="*70)
print("SUCCESS: B1 loaded successfully!\n")

sim_app.close()
```

Run test:
```bash
python test_b1.py --headless
```

**Expected output:**
```
======================================================================
Unitree B1 Configuration Test
======================================================================
✓ USD path: /home/USERNAME/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/b1.usd
✓ Initial height: 0.42 m
✓ Actuators: ['base_legs']
✓ Motor effort limit: 23.7 N·m
✓ Motor velocity limit: 30.0 rad/s
======================================================================
SUCCESS: B1 loaded successfully!
```

---

## Usage in Your Code

```python
from isaaclab_assets.robots.unitree import UNITREE_B1_CFG

# Use in Isaac Lab environment
robot_cfg = UNITREE_B1_CFG
```

---

## Troubleshooting

### Issue 1: ImportError - Cannot import UNITREE_B1_CFG

**Cause:** Configuration not added to `unitree.py` or syntax error

**Fix:**
```bash
cd ~/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots
python -c "from isaaclab_assets.robots.unitree import UNITREE_B1_CFG; print('OK')"
```

If error persists, check for Python syntax errors in `unitree.py`.

### Issue 2: USD file not found

**Cause:** USD file not copied to correct location

**Fix:**
```bash
ls ~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/b1.usd
# If not found, repeat Step 4
```

### Issue 3: URDF conversion fails with "Missing values"

**Cause:** `joint_drive` configuration missing

**Fix:**
Ensure conversion command includes:
```bash
--joint-stiffness 25.0 \
--joint-damping 0.5
```

### Issue 4: Mesh files not found during conversion

**Cause:** URDF still has `package://` paths

**Fix:**
Verify Step 2 was completed:
```bash
grep "package://" ~/Downloads/unitree_ros/robots/b1_description/xacro/b1_fixed.urdf
# Should return NO results

grep "file://" ~/Downloads/unitree_ros/robots/b1_description/xacro/b1_fixed.urdf
# Should show mesh paths with file:// prefix
```

---

## Technical Details

### B1 Specifications

- **DOF:** 12 (4 legs × 3 joints: hip, thigh, calf)
- **Joint pattern:** `[FL,FR,RL,RR]_[hip,thigh,calf]_joint`
- **Motor torque:** 23.7 N·m max
- **Motor velocity:** 30 rad/s max
- **Standing height:** 0.42 m
- **Mass:** ~12 kg (from URDF)

### Actuator Model

Uses `DCMotorCfg` (DC motor model) with:
- **Stiffness:** 25.0 N·m/rad (position control gain)
- **Damping:** 0.5 N·m/(rad/s) (velocity damping)
- **Control mode:** Position control with PD gains

### File Locations

```
~/IsaacLab/
├── source/isaaclab_assets/
│   ├── data/Robots/Unitree/B1/
│   │   └── b1.usd                    # Converted USD file
│   └── isaaclab_assets/robots/
│       └── unitree.py                # B1 configuration added here
└── scripts/tools/
    └── convert_urdf.py               # URDF→USD converter

~/Downloads/
└── unitree_ros/robots/b1_description/
    ├── meshes/                        # B1 geometry files
    └── xacro/
        ├── b1.urdf                    # Original URDF
        └── b1_fixed.urdf              # Fixed mesh paths
```

---

## Working with B1 — Lessons Learned

The setup above gives you a B1 that loads. It does not give you a B1 that trains well. This section documents the **B1-specific** gotchas discovered across Phase 1 (CPG-RBF + PPO) and Phase 2 (residual transition learning) — the kind of thing that takes weeks to find by trial and error.

### B1 is heavy: Go2's reward weights are wrong by orders of magnitude

Stock Isaac Lab locomotion configs are calibrated for the ~15 kg Go2. B1 is ~50 kg with a 23.7 N·m motor budget per joint (vs Go2's ~23 N·m on a much smaller body). The reward and domain-randomization weights that work on Go2 either saturate or fail to act on B1:

| Term | Stock (Go2) | B1 override | Why |
|---|---|---|---|
| `dof_torques_l2` | −2e-4 | **−1e-6** (~200× smaller) | B1 effort 280 → torque² is ~150× larger; stock value saturates motors |
| `add_base_mass` distribution | (−5, +5) kg | **(−10, +10) kg** | Proportional to 50 kg body |
| `reset_base.velocity_range` (z, roll, pitch) | ±0.5 | **0** | A 50 kg body at 0.5 m/s vertical velocity face-plants in ~100 ms |
| `reset_robot_joints.position_range` | (0.5, 1.5) | **(1.0, 1.0)** | ±50 % joint randomisation triggers immediate falls |
| `base_contact.threshold` | 1 N | **50 N** | Settling produces 20–40 N transients on a 50 kg body |
| `env_spacing` | 2.5 m | **3.5 m** | B1 footprint is 1.7× Go2's |
| `gpu_max_rigid_patch_count` | 10·2¹⁵ | **20·2¹⁵** | Larger bodies → more contact patches |

**Takeaway:** when porting any Go2/Anymal cfg to B1, don't assume defaults. Audit every weight that involves mass, force, or velocity scales.

### Asset/cfg name overrides — silent failures if you miss them

B1's USD keeps the URDF original names that Go2's renamed:

```python
# Trunk body name is "trunk" (NOT "base" like Go2).
# Stock LocomotionVelocityRoughEnvCfg uses body_names="base" everywhere.
self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"
self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"

# Foot link names are *_foot (NOT *_calf). Contact-sensor regex:
foot_pattern = ".*_foot$"   # matches FL_foot, FR_foot, RL_foot, RR_foot
```

**Symptom of forgetting these:** `ValueError: Cfg requires 1 body but 0 matched` at env build time.

### Stock `UNITREE_B1_CFG` needs three overrides for actual locomotion

The defaults from Step 5 above (stiffness=25.0, damping=0.5, init_z=0.42) **load** correctly but produce a robot that can't walk:

| Field | Stock | Required | Symptom of leaving stock |
|---|---:|---:|---|
| `actuators["base_legs"].stiffness` | 25.0 | **400.0** | Body sags 9 cm under self-weight at default joints |
| `actuators["base_legs"].damping` | 0.5 | **10.0** | Joint oscillation (proportional to stiffness — keep ratio ~25:1 → 40:1) |
| `init_state.pos[2]` | 0.42 | **0.50** | Feet are 7.7 cm under the ground at default joint angles → spawn-time termination |

**Critical:** never mutate the shared `UNITREE_B1_CFG` directly. Other code (legacy envs, other tasks) imports the same object. Always **deep-copy first**, then mutate the copy:

```python
import copy
UNITREE_B1_CFG = copy.deepcopy(_UNITREE_B1_CFG_FROM_ASSETS)
UNITREE_B1_CFG.actuators["base_legs"].stiffness = 400.0
UNITREE_B1_CFG.actuators["base_legs"].damping = 10.0
UNITREE_B1_CFG.init_state.pos = (0.0, 0.0, 0.50)
```

### Morphology — front/rear thigh asymmetry is permanent

B1's URDF default joint angles encode a **+0.2 rad asymmetry** between front and rear thighs:

```python
# From the Step-5 init_state.joint_pos:
"F[L,R]_thigh_joint": 0.8,     # front legs
"R[L,R]_thigh_joint": 1.0,     # rear legs (+0.2 rad more flexed)
".*L_hip_joint":  0.1,         # left mirror
".*R_hip_joint": -0.1,         # right mirror
".*_calf_joint": -1.5,
```

This isn't a bug — it's how Unitree shipped B1 — but it propagates everywhere:

- **Rear-heavy duty factors** in every trained gait (rear legs work harder than front)
- **Asymmetric leg roles** emerge in trained policies even with bilateral-symmetry rewards
- **Shared-W CPG-RBF encodings cannot represent both leg pairs simultaneously** — there is no single 20×3 weight matrix that satisfies both 0.8 rad and 1.0 rad thigh defaults
- **Phase-2 residual MLPs apply consistently larger corrections to rear legs** (`|Δα_RL|, |Δα_RR| > |Δα_FL|, |Δα_FR|`) — the rear-bias is data, not noise

**Implication for design:** any architecture that assumes per-leg symmetry has a built-in error on B1. Either accept it, model per-leg explicitly, or compensate with hip-deviation/symmetry rewards (and even then, expect residual asymmetry).

### Per-gait stability rewards must be relaxed — one size doesn't fit

Bound has natural fore-aft pitch (body bobs during leap). Pace has natural lateral roll (body sways side-to-side). Stock orientation/velocity penalties **fight** these motions, and the policy compensates by **squatting low** to minimise body motion — breaking the gait. Required per-gait overrides:

```python
# Bound and pace base configs:
flat_orientation_l2 weight: -2.5 → -0.5    # was suppressing pitch/roll → squat
lin_vel_z_l2 weight:        -2.0 → -0.5    # was suppressing fore-aft heave → squat
ang_vel_xy_l2 weight:       -0.05 → -0.02  # was suppressing roll/pitch rate

# Compensate the relaxed stability with tighter height penalty:
base_height_l2 weight:      -50  → -150    # prevent the squat trade-off

# Pace and steer specifically:
joint_lr_symmetry_penalty weight: → 0      # disabled (lateral or asymmetric coordination is inherent)
```

### PPO failure-mode catalog on B1

These are the local optima PPO finds on B1's reward landscape, in order of how often they occur. Each got a custom MDP term in `envs/b1_velocity_mdp.py`:

| Failure mode | Cause | Fix |
|---|---|---|
| **Trot attractor** | Trot is PPO's universal attractor on quadrupeds — bound/pace/walk all need explicit *anti-trot* rewards | Signed XOR-style coordination terms (`true_bound_reward`, `true_pace_reward`) — pure trot pays a penalty |
| **Standstill exploit** | `track_lin_vel_xy_exp(std=0.5)` pays 88 % reward at vx=0 | Tighten std=0.25 + bump weight 1.0 → 1.5 |
| **Crawl exploit (body sags 0.18 m)** | No height penalty in stock | `base_height_l2(target=0.42)` weight −50 (or −150 for bound/pace) |
| **2-leg trot** (one diagonal pair planted forever) | No per-foot time bound | `excessive_air_time(max=0.5)` + `excessive_contact_time(max=0.5)` together |
| **Tap-tap-tap** (5 Hz tap on planted foot to reset timer) | Cumulative time penalties don't catch frequency | `short_swing_penalty(min_swing_time=0.3 s)` |
| **3+1 asymmetric** (FL cycles half-rate of others) | Per-foot bounds ok individually | `air_time_variance_penalty` (variance of last_air_time across 4 feet) |
| **Bilateral L/R asymmetry** (FR hip 2× FL) | No L/R constraint (when expected) | `joint_lr_symmetry_penalty` — but **disable** for pace and steer |
| **Walk never converges** | Low-velocity 3-stance pattern + B1 morphology | After 6 versions: not fixable with reward shaping. Drop walk from gait portfolio |

### Black-box / population-based optimisation is structurally infeasible on B1

PI^BB and similar optimisers collapse on B1 because **most exploratory perturbations cause falls**. All samples cluster near the reward floor (terminated, height-penalty dominated), softmax weights equalise (`p_i → 1/N`), and the update becomes a noise-weighted average of random perturbations — effectively zero. Even after fixing all 5 implementation bugs and full retraining (2000 iters, 60 params shared-W indirect encoding):

| Method | Best vx | vx std | Coordination |
|---|---:|---:|---|
| PIBB CPG-RBF (retrained, all bugs fixed) | +0.091 m/s | 0.171 (oscillates) | Lunge-fall-recover |
| PPO velocity tracking (trot) | **+0.434 m/s** | ~0.02 | Stable diagonal trot |

**A 4.8× structural gap that does not close with implementation polish.** Lighter platforms (Go2 ~15 kg, Thor's hexapod ~5 kg) survive PIBB exploration; B1 does not. **Recommendation: on a 50 kg quadruped, use gradient-based RL (PPO/SAC), not population-based or black-box optimisation.**

### Measurement gotchas that produced wrong conclusions

These are the ways your evaluation script can lie to you:

- **Foot contact**: in playback, use `current_contact_time > 0`, **not** single-frame `net_forces > threshold`. Single-frame snapshots miss brief 1–2 step contacts in fast gaits (bound/pace at 2.5 Hz). Symptom: gait diagrams show wrong duty factors.
- **Foot apex during swing comes from calf flexion, not body height.** Bumping `lin_vel_z_l2` to "stop bouncing" makes the body **squat** instead of reducing foot lift. To raise foot clearance, tune calf joints / RBF amplitudes.
- **`jacc_RMS` is not a smoothness metric.** A constant-high-acceleration trajectory has zero jerk yet large `jacc²`. The motor-relevant smoothness signal is **jerk** = `(q̈_t − q̈_{t-1}) / dt` (rad/s³). Use this for any "smooth motion" claim.
- **Phase-2 specifically**: each frozen base policy must be queried with its **own** previous output as `last_action`. Passing zeros causes all base policies to collapse to default pose (cost us ~3 days in v3).

### Compute envelope (RTX 4070 Ti SUPER, 16.7 GB VRAM)

| Workload | Settings | Wall time |
|---|---|---|
| Phase 1 PPO trot training | 4096 envs, 1500 iters | ~30 min |
| Phase 1 PPO bound (harder reward stack) | 4096 envs, 4000 iters | ~80 min |
| Phase 2 residual MLP training | 2048 envs, 2000 iters | ~60 min |
| Phase 2 playback (2000 steps, plots) | 4 envs | ~3 min |

Adding `gpu_max_rigid_patch_count = 20·2¹⁵` to the sim config is required when going past ~2000 envs — B1's larger contact surfaces overflow Go2's default patch budget.

### Architectural ceilings (the things tuning won't fix)

Two findings that look like polish targets but are actually physical limits:

- **Tilt floor at trot↔bound transitions**: across all Phase 2 polish iterations (v5/v6/v7), `tilt_max` converged to **0.19 ± 0.003** regardless of reward tweaks. At the trot→bound midpoint, the 50 kg body **must** pitch to absorb the momentum shift from diagonal to fore-aft contact. Don't waste training time pushing tilt below 0.19 with a bounded residual.
- **Smoothness ceiling on residual blending**: frozen base policies with different gait phases produce intrinsically jerky blends — mid-α blending sums two out-of-phase oscillations. Per-leg α corrections cannot eliminate this; they can only shorten the time spent in the jerky middle (E2E PPO's strategy) or accept the jerk (v7's strategy). To break the ceiling, the architecture would need phase-aware base policies or a learned per-pair α curve, not a residual on a fixed smoothstep.

---

## References

---

## Changelog

**2026-04-12:** Initial documentation
- B1 URDF conversion completed
- Configuration added to Isaac Lab 0.36.3
- Verified on Ubuntu 22.04, Isaac Sim 4.5