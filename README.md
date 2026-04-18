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
2. **CPG + pattern-formation layer**
   - ✅ **Hopf CPG** ([allgaits/cpg/hopf.py](allgaits/cpg/hopf.py)) — batched, GPU-ready, forward-Euler at 1 kHz per paper eqs 1–2.
   - ✅ **Coupling matrices** for all 9 gaits ([allgaits/cpg/coupling.py](allgaits/cpg/coupling.py)) — Φ derived from Fig. 3 contact timings, plus `batch_coupling_matrices()` for per-env heterogeneous gait sampling.
   - ⏳ **Pattern formation + 3-DOF IK** — paper eqs 3–4 using B1 kinematics (L1=L2=0.35 m, hip-abduction arm 0.12675 m).
3. **Isaac Lab env** — observation/action spaces per paper §II.B, generic velocity-tracking + power reward.
4. **PPO training** (RSL-RL 2.2.4) — resample Φ every 3 s, v* every 5 s, style params per reset. Domain randomization per Table II.
5. **Sim evaluation** — reproduce COT-vs-velocity curves per gait, transition tests, leg-failure robustness.

### Phased milestone plan

Paper-fidelity code, phased config validation (no code changes between phases — just restrict the Φ pool at runtime):

| Week | Phase | Φ pool | Goal |
|---|---|---|---|
| 1 | Infra | — | CPG + IK + env scaffold, all unit-tested |
| 2 | **A** | `{trot}` only | Confirm PPO loop converges to B1 trot at commanded velocity |
| 3 | **B** | `{walk, trot, pace}` | Mid-run Φ swap produces clean gait transitions |
| 4 | **C** | all 9 + transitions | Full AllGaits result, overnight training |

### Tests

[tests/](tests/) — 28/28 passing. Run from repo root:

```bash
conda activate env_isaaclab
pip install -e .
python -m pytest tests/
```

Coverage so far:
- **[tests/test_coupling.py](tests/test_coupling.py)** (14 tests) — Φ antisymmetry, zero diagonal, per-gait phase patterns (trot diagonals, pace laterals, bound pairs, pronk in-phase), batched heterogeneous gait matrices.
- **[tests/test_cpg_hopf.py](tests/test_cpg_hopf.py)** (14 tests) — state shape, `r → μ` limit-cycle convergence, phase advance under ω, **gait phase-lock within 5° after 3 s simulation** for trot/pace/bound/pronk/walk, partial-env reset, CUDA placement.

> Paper: [references/AllGaits: Learning All Quadruped Gaits and Transitions.pdf](references/AllGaits:%20Learning%20All%20Quadruped%20Gaits%20and%20Transitions.pdf) · arXiv 2411.04787 · Nov 2024
