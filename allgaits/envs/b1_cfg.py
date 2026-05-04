"""AllGaits-local override of UNITREE_B1_CFG with stiffer actuators.

The shared `isaaclab_assets.robots.unitree.UNITREE_B1_CFG` defaults to
stiffness=200 N·m/rad, damping=5, which is too low for B1 (~62 kg).
During bypass-CPG locomotion tests the body sagged ~9 cm below the
commanded height and drifted backward at -0.27 m/s because the legs
couldn't hold the stance pose.

Rather than editing the shared config (which would silently change
behaviour in every other project on this machine — e.g. cpg-drl-transition
also imports UNITREE_B1_CFG), we copy it and raise the PD gains here.
This keeps the override scoped to AllGaits.

Stiffness tuning history (mean_vx in bypass trot at μ=1.5, ω=2 Hz):
    200 → -0.27 m/s (severe sag, 9 cm)
    400 → -0.044 m/s (6x reduction, still ~6 cm sag)
    600 → expected ~+0.1 m/s (target forward)
"""

from __future__ import annotations

import copy

from isaaclab.actuators import DCMotorCfg
from isaaclab_assets.robots.unitree import UNITREE_B1_CFG


def _make_b1_allgaits_cfg():
    """Return a deep-copy of UNITREE_B1_CFG with AllGaits-tuned actuators."""
    cfg = copy.deepcopy(UNITREE_B1_CFG)
    cfg.actuators = {
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=280.0,
            saturation_effort=280.0,
            velocity_limit=21.0,
            stiffness=600.0,   # Raised 200→600 for B1's 62 kg body; prevents stance sag
            damping=15.0,      # Scaled proportionally (3× stiffness → 3× damping)
            friction=0.0,
        ),
    }
    return cfg


UNITREE_B1_ALLGAITS_CFG = _make_b1_allgaits_cfg()
"""B1 ArticulationCfg used by the AllGaits env. Identical to UNITREE_B1_CFG
except for raised PD stiffness (200→600) and damping (5→15)."""
