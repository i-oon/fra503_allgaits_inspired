"""PPO (RSL-RL) config matching CPG-RL Table I (Bellegarda & Ijspeert 2022 RAL).

AllGaits explicitly inherits these PPO hyperparameters ("same as [36]") —
see reference paper at `references/CPG-RL: ...pdf` Table I.

Notes on batch sizing:
    batch_size          = num_envs × num_steps_per_env   = 4096 × 24 = 98304
    mini_batch_size     = num_envs × num_steps / K       = 4096 × 6  = 24576
                          (K = num_mini_batches = 4)
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class AllGaitsPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    # CPG-RL Table I: batch 4096×24 with 5 epochs, 4 mini-batches
    num_steps_per_env: int = 24
    max_iterations: int = 5000          # override via CLI for longer runs
    save_interval: int = 50
    experiment_name: str = "allgaits_b1"
    # Running-mean/std observation normalizer. Our obs includes joint_vel
    # (up to ±21 rad/s) and CPG theta_dot (can exceed 60 rad/s), which are
    # much larger than normalized signals like projected_gravity (±1) —
    # without normalization the value function struggles to fit uniformly.
    empirical_normalization: bool = True

    # --- Actor-Critic network ---
    # MLP [512, 256, 128] with ELU activation, matching CPG-RL Table I.
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        # Start small so the Gaussian stays in the linear region of the
        # action decoder (|policy out| ≤ 1 maps to the physical range
        # without clipping). init=1.0 put 32% of samples in the clip dead
        # zone from step 0, which fed the noise-std runaway.
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # --- PPO algorithm ---
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,                  # CPG-RL Table I
        # Lower than CPG-RL's 0.01 — paired with a stronger action-rate
        # penalty below. At 0.01 the entropy bonus overpowered our
        # penalty and the Gaussian std ran up to 4+, producing 96%
        # clamp-saturated actions (bang-bang) that learn slowly.
        entropy_coef=0.001,
        num_learning_epochs=5,           # CPG-RL Table I
        num_mini_batches=4,              # gives mini_batch = batch / 4 = 24576
        learning_rate=1.0e-3,
        schedule="adaptive",             # adaptive LR per CPG-RL
        gamma=0.99,                      # CPG-RL Table I
        lam=0.95,                        # GAE λ, CPG-RL Table I
        desired_kl=0.01,                 # KL target for adaptive LR, CPG-RL Table I
        max_grad_norm=1.0,
    )
