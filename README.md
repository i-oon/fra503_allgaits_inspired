# fra503_allgaits_inspired

Replicating **AllGaits** (Bellegarda, Shafiee, Ijspeert — EPFL, 2024) on the **Unitree B1** in Isaac Lab. FRA503 course project.

---

## Why: the discrete gait transition problem

Animals don't use one fixed gait. A horse walks at low speed, trots at medium, and gallops at high speed — and it switches between these **discrete** patterns smoothly, in response to speed, terrain, and fatigue. Each gait is a distinct inter-limb phase relationship (walk, trot, bound, gallop…), not a continuous blend, and the switch itself is a control problem: **which** gait, **when** to switch, and **how** to land in the new pattern without falling.

This matters for quadruped robots because:

- **No single gait is optimal everywhere.** Trot is stable at medium speed, wasteful at low speed (walk is more efficient), and unstable at high speed (bound/gallop win). A robot stuck in one gait gives up energy, speed, and stability across most of its operating range.
- **Transitions are a separate problem from the gaits themselves.** A controller that walks well and trots well does not automatically switch between them. Naïve switching causes stumbles and falls.
- **Existing solutions are fragmented:**
  - **MPC** — needs per-gait parameter tuning and hand-crafted transition heuristics.
  - **Multi-policy DRL** — one policy per gait, stitched by a meta-controller. More moving parts, more sim-to-real risk.
  - **End-to-end DRL with generic rewards** — almost always collapses to **trot only**. Standard reward terms (angular-velocity penalties, symmetry) make trot a deep local minimum; other gaits never emerge.
  - **Imitation from mocap** — needs reference trajectories for every gait *and* every transition, which don't exist at scale.

**The open question:** can a single controller produce all natural gaits **and** transition between any pair, without per-gait reward engineering or demonstrations?

## Enter AllGaits

**AllGaits** answers yes — and the twist is that **gait transitions aren't really the paper's headline**. Its stated pitch is "one policy that learns all 9 quadrupedal gaits from a single generic reward." But the mechanism — a PPO policy that modulates 4 coupled oscillators (a Central Pattern Generator), where **the gait is selected by a coupling matrix Φ passed in at runtime** — makes transitions come *for free*:

- **Switch gaits mid-run?** Swap Φ. Training re-sampled Φ every 3 s, so the policy has already learned to re-stabilize across arbitrary gait changes.
- **Want a gait that doesn't exist in nature?** Hand it a novel Φ. The paper shows artificial 3-in-phase + 1-out-of-phase gaits that were never trained.
- **Lose a leg?** It keeps going, despite never training on leg failures.

Put differently: AllGaits collapses *gait* and *transition* into the same problem — *"track velocity under whatever inter-limb coupling the user requests"* — and that reframing dissolves the transition problem almost incidentally. The discrete switching that prior work handled with heuristics or meta-controllers is absorbed into the training distribution.

This is the idea we want to validate on a different, larger robot.

## This repo

Port the AllGaits framework to the Unitree B1 in Isaac Lab 0.36.3 / Isaac Sim 4.5. **Sim-only**, ~3–4 week FRA503 deliverable.

1. **Robot setup** — B1 URDF → USD, `UNITREE_B1_CFG` in Isaac Lab. See [SETUP.md](SETUP.md). ✅
2. **CPG + pattern-formation layer** ✅
   - **Hopf CPG** ([allgaits/cpg/hopf.py](allgaits/cpg/hopf.py)) — batched, GPU-ready, forward-Euler at 1 kHz per paper eqs 1–2. Convergence factor `a = 150` inherited from CPG-RL (Bellegarda 2022 RAL §III-A).
   - **Coupling matrices** for all 9 gaits ([allgaits/cpg/coupling.py](allgaits/cpg/coupling.py)) — Φ derived from Fig. 3 contact timings, plus `batch_coupling_matrices()` for per-env heterogeneous gait sampling.
   - **B1 leg FK/IK** ([allgaits/kinematics/b1.py](allgaits/kinematics/b1.py)) — 3-DOF quadruped leg with abduction-hip, thigh, calf. URDF-verified geometry (L1=L2=0.35 m, hip-abduction arm 0.12675 m). Also exposes `B1_COM_OFFSET_X = -0.018 m`: B1's overall COM sits ~1.8 cm behind the geometric center of the body (mass asymmetry from 0.8 rad front-thigh vs 1.0 rad rear-thigh defaults). Mirrors AllGaits' Go1 observation (§III-A) that COM-behind-hips motivates negative `x_off`.
   - **Pattern formation** ([allgaits/cpg/pattern.py](allgaits/cpg/pattern.py)) — paper eqs 3–4: CPG state `(r, θ)` + style params `(h, g_c, g_p, x_off, d_step)` → foot target → IK → joint position targets. Exposes `B1_STYLE_PARAM_RANGES` scaled 1.5× from the paper's Go1 ranges (h∈[0.27, 0.52], g_c∈[0.03, 0.18], x_off∈[-0.12, 0.04] — biased negative to align with B1's COM).
3. **Isaac Lab env** ([allgaits/envs/](allgaits/envs/)) ✅ *mechanically working; see debugging log below*
   - `AllGaitsEnvCfg` ([allgaits_env_cfg.py](allgaits/envs/allgaits_env_cfg.py)) — physics 1 kHz, policy 100 Hz (decimation 10), 20 s episodes, 4096 envs. Action ∈ ℝ⁸ linear-scaled + clamped to μ∈[1,2], ω∈[0,8] Hz. Observation 64-D matching paper §II-B.
   - `AllGaitsEnv` ([allgaits_env.py](allgaits/envs/allgaits_env.py)) — CPG integrated at 1 kHz inside `_apply_action`; pattern formation + IK produces joint PD targets; per-env Φ sampled from `active_gaits` pool. Periodic resampling (`v*` every 5 s, `Φ` every 3 s) per paper §II-C.
   - Reward (final): `lin_vel_x_track (6·dt) + lin_vel_x_direct (2·dt) + heading (1.5·dt) + cpg_active (0.5·dt, capped 3 Hz) − cpg_runaway (2.0·dt, quadratic above 3 Hz) − lin_vel_yz (2·dt) − ω_xyz (0.35·dt) − power (0.001·dt) − action_rate (0.025·dt)`. See debugging log §8–9 for the reward-tuning history.
   - Diagnostic logging: per-term reward, locomotion state (vel cmd/actual, height, tilt), action stats (raw mag, clip fraction, μ/ω means, rate), termination breakdown (fall_rate, timeout_rate, peak force). All 19 metrics printed per iteration AND logged to TensorBoard under `Episode/*`.
   - Spawn z-offset +0.08 m added in reset to prevent sub-ground foot settling; termination threshold 50 N on base/thigh (absorbs reset-settling transients).
   - **Local B1 override** ([b1_cfg.py](allgaits/envs/b1_cfg.py)) — `UNITREE_B1_ALLGAITS_CFG` with stiffness=600, damping=15 (up from shared default 200/5). Required because B1 at 62 kg sags ~9 cm under the default gains; local override avoids side-effects on `cpg-drl-transition`.
   - **Smoke test**: `python scripts/test_env.py --headless --num_envs 16 --steps 100` — requires Isaac Sim.
4. **PPO training** (RSL-RL 2.2.4) ✅ *Phase B complete — 3 gaits working*
   - [allgaits/training/ppo_cfg.py](allgaits/training/ppo_cfg.py) — `AllGaitsPpoRunnerCfg`: batch 4096×24, 5 epochs, 4 mini-batches, clip 0.2, **entropy 0.001** (down from CPG-RL Table I's 0.01), γ=0.99, λ=0.95, KL target 0.01, MLP [512, 256, 128] ELU, `init_noise_std=0.5`, `empirical_normalization=True`.
   - [allgaits/tasks/__init__.py](allgaits/tasks/__init__.py) — three registered gym tasks (one per phase): `Isaac-AllGaits-B1-Trot-v0`, `…-3Gait-v0`, `…-Full-v0`. Same env, different `active_gaits`.
   - [scripts/train.py](scripts/train.py) — entry point. `python scripts/train.py --task Isaac-AllGaits-B1-3Gait-v0 --num_envs 4096 --headless --max_iterations 6000`. Saves `final_model.pt` alias after training.
   - [scripts/play.py](scripts/play.py) — per-env diagnostic table; supports `--load_run` (exact name or regex), `--model` (step number or `final_model`), `--stochastic`, `--bypass_policy`, `--fix_h/g_c/x_off` style overrides, `--gait` coupling-matrix override. Prints PhysX-vs-FK ground-truth at step 1.
   - Logs: `logs/rsl_rl/allgaits_b1/` (TensorBoard: `tensorboard --logdir logs/rsl_rl/allgaits_b1`)
5. **Sim evaluation** ✅ *3 gaits validated in play*

### Demo results — checkpoint `logs/rsl_rl/allgaits_b1/phase_b_3gaits/final_model.pt`

Trained on Phase B (walk/trot/pace), 6000 iterations, 4096 envs. Style params fixed at h=0.35, g_c=0.08, x_off=−0.04 for stable play.

| Gait | cmd (m/s) | mean vx (m/s) | tracking | falls |
|------|-----------|---------------|----------|-------|
| Trot | 0.80 | +0.558 | 70% | 0/4 |
| Walk | 0.40 | +0.417 | 104% | 0/4 |
| Pace | 0.40 | +0.295 | 74% | 1/4 |

**Known limitation**: systematic rightward yaw drift across all gaits — the robot trots/walks forward in its own frame but rotates ~180° over 20 s. Caused by front/rear thigh asymmetry (0.8 vs 1.0 rad defaults) creating unequal step lengths; heading reward at 1.5 is insufficient to fully counteract it.

```bash
# trot
python scripts/play.py --load_run phase_b_3gaits --model final_model \
    --gait trot --vel_x 0.8 --num_envs 4 --fix_h 0.35 --fix_g_c 0.08 --fix_x_off -0.04

# walk
python scripts/play.py --load_run phase_b_3gaits --model final_model \
    --gait walk --vel_x 0.4 --num_envs 4 --fix_h 0.35 --fix_g_c 0.08 --fix_x_off -0.04

# pace
python scripts/play.py --load_run phase_b_3gaits --model final_model \
    --gait pace --vel_x 0.4 --num_envs 4 --fix_h 0.35 --fix_g_c 0.08 --fix_x_off -0.04
```

### Phased milestone plan

Paper-fidelity code, phased config validation (no code changes between phases — just restrict the Φ pool at runtime):

| Week | Phase | Φ pool | Goal | Status |
|---|---|---|---|---|
| 1 | Infra | — | CPG + IK + env scaffold, all unit-tested | ✅ |
| 2 | **A** | `{trot}` only | Confirm PPO loop converges to B1 trot at commanded velocity | ✅ |
| 3 | **B** | `{walk, trot, pace}` | 3 gaits working at target velocity | ✅ trot 70%, walk 104%, pace 74% |
| 4 | **C** | all 9 + transitions | Full AllGaits result | ⏳ out of scope for deadline |

### Tests

[tests/](tests/) — **58/58 passing** (~2 s). Run from repo root:

```bash
conda activate env_isaaclab
pip install -e .
python -m pytest tests/
```

Coverage so far:
- **[tests/test_coupling.py](tests/test_coupling.py)** (14 tests) — Φ antisymmetry, zero diagonal, per-gait phase patterns (trot diagonals, pace laterals, bound pairs, pronk in-phase), batched heterogeneous gait matrices.
- **[tests/test_cpg_hopf.py](tests/test_cpg_hopf.py)** (14 tests) — state shape, `r → μ` limit-cycle convergence, phase advance under ω, **gait phase-lock within 5° after 3 s simulation** for trot/pace/bound/pronk/walk, partial-env reset, CUDA placement.
- **[tests/test_kinematics.py](tests/test_kinematics.py)** (15 tests) — URDF-verified geometry, hip-joint positions, **B1 COM is 1.8 cm behind geometric center**, FK at zero/default poses, **IK→FK round-trip on 128 reachable sagittal points** (both left & right legs), left/right mirror symmetry, graceful clamping at workspace boundary (no NaN), (N, 4)-batched 4-leg FK.
- **[tests/test_pattern.py](tests/test_pattern.py)** (15 tests) — paper eqs 3–4 at rest/swing/stance, amplitude→step-length scaling, `x_off` shift, **full pipeline round-trip `CPG → foot → IK → FK` recovers target**, derived joints stay inside B1 URDF joint limits, `B1_STYLE_PARAM_RANGES` covers required keys, h range brackets 0.42 m, x_off range biased negative and includes `B1_COM_OFFSET_X`.

## Debugging log — what went wrong and how we diagnosed it

Phase A took five training runs and a lot of bypass-mode diagnostics to get the mechanism correct. Recording the issues here so the same traps don't catch us again in Phase B/C.

### 1. Contact-sensor indexing returned the wrong bodies (silent)
- **Symptom**: `feet↓ = 0` in every env at every step, even though the robot was clearly on the ground. Termination metric `max_base_thigh_force_N` also reported near-zero.
- **Cause**: I wired `self._foot_body_ids = self._robot.find_bodies(".*_foot$")`, but `net_forces_w_history` is indexed by the **contact sensor's** body list, not the articulation's. The two can have different orders.
- **Fix**: use `self._contact_sensor.find_bodies(".*_foot$")` for any indexing into `data.net_forces_w_history`. Anchor patterns with `$` to avoid matching suffix variants.
- **Impact**: trained with broken foot-contact observations for two full runs; the policy had no gait-timing signal.

### 2. Default pose spawns feet 7.7 cm below ground
- **Symptom**: In play mode, every env terminated within 3–5 steps (~25 resets per 100 steps). In training it was masked by `init_at_random_ep_len=True` staggering resets across time.
- **Cause**: `UNITREE_B1_CFG` spawns trunk at z=0.42 m, but FK says foot is 0.497 m below hip at default joint angles. So feet spawn 7.7 cm underground → PhysX settling transient → thigh contact force spikes above 10 N → termination.
- **Fix**: add `+0.08 m` to the spawn z in `_reset_idx`; raise `base_contact_threshold_n` from 10 N → 50 N (a real fall peaks at 300+ N on a 62 kg robot, so 50 N is safely above settling noise).

### 3. Action-noise std running away to 22 (bang-bang control)
- **Symptom**: `Policy/mean_noise_std` grew from init 1.0 to 22 over 3000 iters. `action/raw_clipped_frac = 0.96` — 96% of actions were in the tanh-saturation dead zone.
- **Cause**: The env's action decoder used `tanh` (then later `clamp`); either creates a zero-gradient region above |policy_output| ≈ 1. Once noise std grew past that, the entropy bonus in PPO kept pushing it higher with no penalty (policy effectively getting free entropy reward).
- **Fix sequence**:
  1. Switched to **linear scale + clamp** (`(action * scale + mid).clamp(min, max)`) — didn't fix the dead zone, just moved it.
  2. Added **action-rate penalty** `−‖Δa‖² · w · dt` to reward — creates a direct gradient cost on action noise.
  3. **Tuning**: tested `w=0.01` (noise_std capped at 2.0 but learning plateaued), `w=0.003` (noise_std ran to 4.1), finally `w=0.025` paired with **`entropy_coef: 0.01 → 0.001`**.
  4. Analytic equilibrium: `σ² ≈ entropy_coef / (4·w·dt)` — with `0.001 / (4·0.025·0.01) = 1`, predicting σ ≈ 1.0. **Observed σ = 0.25** (even better than predicted).
- **Impact**: two full training runs wasted on bang-bang policies before diagnostics exposed the saturation.

### 4. Terminal vs TensorBoard routing of custom metrics
- **Symptom**: `self.extras["log"] = {...}` appeared in TensorBoard under `Episode/...` but **not** in the RSL-RL iteration-end print block.
- **Cause**: isaaclab_rl's `RslRlVecEnvWrapper` passes `extras` through unchanged, but RSL-RL's `OnPolicyRunner.log()` looks for `infos["episode"]`, not `infos["log"]`.
- **Fix**: populate `self.extras["episode"] = {...}` directly.

### 5. Actuator stiffness of 200 N·m/rad is undersized for B1 (62 kg)
- **Symptom**: In bypass mode (μ=1.5, ω=2 Hz), body sagged 9 cm below commanded height, tilted 8° nose-up, and drifted backward at **−0.27 m/s** instead of trotting forward.
- **Cause**: PD stiffness 200 N·m/rad (inherited from Go1/A1 configs) can't support B1's 62 kg body at the commanded stance height. Legs bend under load, body sags, rear legs (which require more extension at the commanded pose) sag more than front → body pitches nose-up → gravity drags it backward.
- **Fix progression** (bypass backward-drift measurement, lower is better):
  - Stiffness=200: −0.27 m/s (severe sag)
  - Stiffness=400: −0.044 m/s (body sag partially fixed)
  - Stiffness=600: **−0.15 m/s** with body at 0.428 m (matching commanded 0.42), tilt 0.055 (3°) — stance is now stable and properly supported.
- **Critical**: edited the **shared** `isaaclab_assets.robots.unitree.UNITREE_B1_CFG` initially, which would have silently affected `cpg-drl-transition` too. Reverted and moved override to a local **[allgaits/envs/b1_cfg.py](allgaits/envs/b1_cfg.py)** that deep-copies `UNITREE_B1_CFG` and raises the gains. **Lesson: never mutate shared Isaac Lab asset configs from a project.**

### 6. Residual −0.15 m/s backward drift under "ideal" bypass
- **Symptom**: With stance fully supported (stiffness 600, body at 0.428, tilt 3°), bypass trot at μ=1.5/ω=2 Hz still drifts backward at 0.15 m/s.
- **Analysis attempts**: FK was verified exactly against PhysX-reported body positions (foot-in-hip-frame matches to 4 decimal places). The paper's `x_foot = x_off − d_step·(r−1)·cos(θ)` equation was flipped to `+` and also tested — gave −0.37 m/s (worse). Both gaits are structurally correct (diagonal pairs in-phase at 180°), feet↓ cycling ~2.6.
- **Current hypothesis**: asymmetric default thigh angles (0.8 rad front vs 1.0 rad rear) create a residual pitch/shear during stance transitions that the constant-μ,ω bypass can't compensate for. Swing-leg reaction forces at 2 Hz contribute but can't fully explain the magnitude.
- **Decision**: accept the imperfect bypass baseline and move to PPO training. The trained policy adapts μ, ω per-observation and has enough action bandwidth (μ∈[1,2], ω∈[0,8] Hz) to overcome a 0.15 m/s baseline by commanding stronger push-off. **Bypass is a diagnostic, not the operating mode.**

### 7. Isaac Sim hangs at startup
- **Symptom**: Play/train scripts print up to "app ready" then hang indefinitely (dark Isaac Sim window).
- **Cause**: zombie python processes from a prior run holding the GPU (`nvidia-smi` showed 7.7 GB used and 97% utilization by two stale python processes).
- **Fix**: `pgrep -f "python.*play\|python.*train\|isaac\|kit" | xargs -r kill -9` before each run. **Always check `nvidia-smi`** before starting a new Isaac Sim instance; the 4070 Ti SUPER (16 GB) cannot host two 4096-env training runs concurrently.

### 8. ω=0 collapse for non-trot gaits (ang_vel penalty too high)
- **Symptom**: After Phase C retrains with `ang_vel` penalty raised to 2.0, walk and pace policies converged to ω=0 — all legs frozen, robot standing still or falling immediately. Trot also began failing.
- **Cause**: `ang_vel_xyz` penalty at 2.0 is 20× the original value (0.1). Every gait except trot produces some oscillation-induced yaw during normal footfall. At 2.0 weight, the penalty for a 0.5 rad/s yaw transient (0.005/step) exceeds the locomotion gain, so the policy's optimal strategy is ω=0 + standstill. Trot is less prone because its diagonal symmetry minimises net yaw, but even it eventually succumbs.
- **Fix**: reduce `REWARD_ANG_VEL_XYZ_PENALTY` to **0.35** — enough to suppress sustained spinning (heading reward handles the directional gradient) without killing gait oscillations.

### 9. CPG active reward blew ω to 6–8 Hz (Phase B v5 backward motion)
- **Symptom**: Phase B retrain with `rew_cpg_active=0.25` clipped at 4 Hz reached 93% trot tracking at training time but went **backward** at 6–8 Hz during play for all three gaits (trot, walk, pace).
- **Cause**: The 4 Hz clip removes the *reward incentive* above 4 Hz but does not penalise it. PPO found a policy where high ω maximised some other gradient and the clip had no corrective force. At 6–8 Hz, foot tangential velocity during stance exceeds the friction limit → foot slides backward → ground reaction force reverses → backward net motion.
- **Fix**: replace the one-sided clip with a two-sided shape: **reward ω linearly up to 3 Hz** (natural stride range), then apply a **quadratic penalty above 3 Hz** (`rew_cpg_runaway = 2.0 × (ω − 3)²`). The penalty creates a hard gradient ceiling that prevents runaway while keeping the incentive to leave ω=0.

### 10. Checkpoint discovery bug (alphabetical sort picked wrong run and wrong step)
- **Symptom**: `play.py --load_run ".*"` loaded `all_gaits_v2/model_950.pt` instead of `all_gaits_v1/model_5999.pt` — the worst checkpoint from the wrong run.
- **Cause**: `_find_latest_checkpoint` sorted run directories alphabetically (`v2 > v1`) and checkpoint filenames as strings (`"950" > "5999"` because `"9" > "5"`).
- **Fix**: sort runs by `os.path.getmtime()` (newest last), sort checkpoints by `int(re.search(r"(\d+)", f).group(1))`. Also added exact-name matching so timestamp directories like `2026-05-03_21-15-39` can be passed directly without regex escaping.

---

> Paper: [references/AllGaits: Learning All Quadruped Gaits and Transitions.pdf](references/AllGaits:%20Learning%20All%20Quadruped%20Gaits%20and%20Transitions.pdf) · arXiv 2411.04787 · Nov 2024
