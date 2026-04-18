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
- ✅ DC motor actuator model tuned for B1's mass (effort 280 N·m ceiling, velocity 21 rad/s, stiffness 200, damping 5)
- ✅ Ready-to-use `UNITREE_B1_CFG` in Isaac Lab

**Time required:** ~15 minutes

---

## Development Environment

### Hardware
- **GPU:** NVIDIA RTX 4070 Ti SUPER (16 GB VRAM)
- **CPU:** Intel i5-14400F (10 cores, 16 threads)
- **RAM:** 32 GB
- **OS:** Ubuntu 22.04

### Software Stack
- **Python:** 3.10.19
- **PyTorch:** 2.5.1+cu121
- **Isaac Lab:** 0.36.3 (core `isaaclab` pkg; overall release `VERSION` 2.0.2) at `~/IsaacLab/`
- **Isaac Sim:** 4.5.0
- **RSL-RL:** 2.2.4 (`rsl-rl-lib`). Use this for PPO — **not** Stable-Baselines3.
- **Conda env:** `env_isaaclab` — always `conda activate env_isaaclab` before any Isaac Lab commands.

### Robot Platform
- **Model:** Unitree B1 quadruped (12 DOF = 4 legs × 3 joints)
- **Mass:** ~50 kg (Unitree datasheet) / ~62.6 kg (URDF inertial sum)
- **Asset:** Custom USD from the official URDF, registered as `UNITREE_B1_CFG` in `isaaclab_assets.robots.unitree`
- **USD location:** `~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/b1.usd`
- **Simulation scope:** Flat terrain only (for now)

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
- Output USD file should be ~1-2 KB (plus `configuration/` sublayers and `config.yaml`)
- The `--joint-stiffness` / `--joint-damping` args here write default drive gains into the USD. These are **overridden** at runtime by the `DCMotorCfg` gains set in Step 5, so the exact values passed to the converter don't affect the trained policy. They're left at 25.0 / 0.5 for compatibility with the existing `config.yaml`.

---

## Step 4: Copy USD to Isaac Lab Assets

The URDF converter produces a **layered USD**: a top-level `b1.usd` that references sub-USDs in a `configuration/` folder, plus a `config.yaml` conversion log. **All of these must be copied together** — `b1.usd` alone will fail to resolve its sublayers.

```bash
# Create B1 directory
mkdir -p ~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1

# Copy the full converter output (not just b1.usd)
cp -r ~/Downloads/b1_usd/. \
   ~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/

# Verify
ls -lh ~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/
```

Expected output:
```
-rw-rw-r-- 1 user user 1.6K  b1.usd               # top-level stage (references configuration/)
drwxrwxr-x 2 user user 4.0K  configuration/       # b1_base.usd, b1_physics.usd, b1_sensor.usd
-rw-rw-r-- 1 user user  610  config.yaml          # URDF converter metadata (safe to keep)
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
            effort_limit=280.0,     # Permissive ceiling; URDF per-joint: 91 (hip), 93 (thigh), 140 (calf) N·m
            saturation_effort=280.0,
            velocity_limit=21.0,    # Average of URDF per-joint: 19.7 / 23.3 / 15.6 rad/s
            stiffness=200.0,        # Scaled for ~50 kg robot (A1's 25.0 cannot overcome B1's mass)
            damping=5.0,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree B1 using converted USD from official URDF.

Reference: https://github.com/unitreerobotics/unitree_ros
URDF per-joint limits: hip 91 N·m / 19.7 rad/s, thigh 93 N·m / 23.3 rad/s, calf 140 N·m / 15.6 rad/s
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
✓ Motor effort limit: 280.0 N·m
✓ Motor velocity limit: 21.0 rad/s
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

## B1 Robot Specifics

### Basic Specs

- **DOF:** 12 (4 legs × 3 joints: hip, thigh, calf)
- **Joint naming:** `[FL,FR,RL,RR]_[hip,thigh,calf]_joint`
- **Standing height:** 0.42 m
- **Mass:** ~50 kg (Unitree datasheet) / ~62.6 kg (URDF inertial sum — includes rotor/stator bodies)

### Joint Axis Convention (CRITICAL)

Verified from the URDF `<axis xyz="...">` tags (pattern repeats identically for all four legs):

```
hip_joint   → axis = (1, 0, 0) → ABDUCTION (side-to-side lateral splay)
              Evidence: defaults FL=+0.1, FR=-0.1 (left/right mirror)
              CPG W column 0 — keep small (~±2°)

thigh_joint → axis = (0, 1, 0) → FLEXION (forward-backward swing) ← PRIMARY WALKING JOINT
              Evidence: defaults all positive (+0.8 front, +1.0 rear)
              CPG W column 1 — dominant for locomotion (~±12-15°)

calf_joint  → axis = (0, 1, 0) → KNEE BEND (pitch)
              Evidence: defaults all -1.5 rad (-86°)
              CPG W column 2 — foot clearance during swing (~±9-12°)
```

### Default Joint Positions (asymmetric front/rear)

```
Front thighs: +0.8 rad (+45.8°)     ← less forward lean
Rear thighs:  +1.0 rad (+57.3°)     ← more forward lean
All calves:   -1.5 rad (-85.9°)
Front hips:   +0.1 / -0.1 rad       ← slight outward splay
Rear hips:    +0.1 / -0.1 rad
```

The front/rear thigh asymmetry (0.8 vs 1.0) means the same CPG offset produces different per-leg behavior. **Hypothesis (verify empirically):** rear legs tend to stay planted longer under identical oscillator input.

### Actuator Configuration (DCMotorCfg)

```python
# In ~/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py
DCMotorCfg(
    effort_limit=280.0,       # Permissive ceiling (URDF per-joint: 91 hip / 93 thigh / 140 calf N·m)
    saturation_effort=280.0,
    velocity_limit=21.0,      # Average of URDF per-joint (19.7 / 23.3 / 15.6 rad/s)
    stiffness=200.0,          # Scaled for ~50 kg robot (cf. H1 uses 200 at ~47 kg)
    damping=5.0,
)
```

**WARNING:** This guide previously generated A1-style values (effort=23.7, velocity=30, stiffness=25, damping=0.5). Those are for the **12 kg A1**, not the **50 kg B1** — at those gains the PD controller cannot overcome B1's mass and joints will not track position targets. Always use the values above.

### Contact Sensor

- Foot contact pattern: **`.*_foot$`** (not `.*_calf$`).
- The URDF defines both `.*_calf` (lower leg bone) and `.*_foot` (foot pad) links. Only `.*_foot` reliably detects ground contact.
- `activate_contact_sensors=True` must be set in `spawn=sim_utils.UsdFileCfg(...)`.

### File Locations

```
~/IsaacLab/
├── source/isaaclab_assets/
│   ├── data/Robots/Unitree/B1/
│   │   ├── b1.usd                    # Top-level USD stage
│   │   ├── configuration/            # Layered sub-USDs (base, physics, sensor)
│   │   └── config.yaml               # URDF converter metadata
│   └── isaaclab_assets/robots/
│       └── unitree.py                # UNITREE_B1_CFG defined here
└── scripts/tools/
    └── convert_urdf.py               # URDF→USD converter

~/Downloads/
└── unitree_ros/robots/b1_description/
    ├── meshes/                       # B1 geometry files
    └── xacro/
        ├── b1.urdf                   # Original URDF
        └── b1_fixed.urdf             # Fixed mesh paths (file:// instead of package://)
```

---

## References

- Unitree ROS: https://github.com/unitreerobotics/unitree_ros
- Isaac Lab Docs: https://isaac-sim.github.io/IsaacLab/
- URDF Converter: `~/IsaacLab/scripts/tools/convert_urdf.py`

---

## Changelog

**2026-04-12:** Initial documentation
- B1 URDF conversion completed
- Configuration added to Isaac Lab 0.36.3
- Verified on Ubuntu 22.04, Isaac Sim 4.5

**2026-04-18:** Reconciled with actual environment
- Added Development Environment section (hardware / software stack)
- Replaced placeholder actuator values (A1: 23.7 N·m / 30 rad/s / stiffness 25) with B1-correct values (280 N·m / 21 rad/s / stiffness 200 / damping 5)
- Corrected mass estimate: ~50 kg datasheet (was incorrectly listed as ~12 kg, an A1 value)
- Documented layered USD output (`configuration/` subfolder, `config.yaml`) and updated Step 4 to copy full converter output
- Added B1 Robot Specifics section: verified joint axis convention, default joint positions, contact-sensor link naming (`.*_foot$` not `.*_calf$`)