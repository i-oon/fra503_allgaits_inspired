"""Gym task registration for Isaac Lab.

Three registered tasks mirror the phased training plan:

    Isaac-AllGaits-B1-Trot-v0       Phase A — single gait (trot)
    Isaac-AllGaits-B1-3Gait-v0      Phase B — {walk, trot, pace}
    Isaac-AllGaits-B1-Full-v0       Phase C — all 9 gaits

Each task is the same env class/config with a different `active_gaits` tuple.
"""

from __future__ import annotations

import gymnasium as gym

from allgaits.envs.allgaits_env import AllGaitsEnv
from allgaits.envs.allgaits_env_cfg import (
    AllGaitsEnvCfg,
    PHASE_A_GAITS,
    PHASE_B_GAITS,
    PHASE_C_GAITS,
)
from allgaits.training.ppo_cfg import AllGaitsPpoRunnerCfg


def _env_cfg_for_gaits(gaits):
    def make_cfg():
        cfg = AllGaitsEnvCfg()
        cfg.active_gaits = gaits
        return cfg
    return make_cfg


gym.register(
    id="Isaac-AllGaits-B1-Trot-v0",
    entry_point="allgaits.envs.allgaits_env:AllGaitsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _env_cfg_for_gaits(PHASE_A_GAITS),
        "rsl_rl_cfg_entry_point": AllGaitsPpoRunnerCfg,
    },
)

gym.register(
    id="Isaac-AllGaits-B1-3Gait-v0",
    entry_point="allgaits.envs.allgaits_env:AllGaitsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _env_cfg_for_gaits(PHASE_B_GAITS),
        "rsl_rl_cfg_entry_point": AllGaitsPpoRunnerCfg,
    },
)

gym.register(
    id="Isaac-AllGaits-B1-Full-v0",
    entry_point="allgaits.envs.allgaits_env:AllGaitsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _env_cfg_for_gaits(PHASE_C_GAITS),
        "rsl_rl_cfg_entry_point": AllGaitsPpoRunnerCfg,
    },
)
