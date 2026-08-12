# VICReg Failure Bundle — Compilation Index

**What this bundle is:** the complete, self-contained record of the
*"World Model + VICReg + SIGReg"* (arm 2) experiment on PushT — the **entire
repository code tree, file-for-file** (see `MANIFEST.md` for the exhaustive
172-file list), together with the failure analysis and every recorded result.

**Headline:** training converged beautifully (123,858 steps, no collapse, probe
R² rising) and then the model planned *worse* than the weaker baseline:
**Open-loop 4.67 ± 1.15, MPC 25.33 ± 2.31** vs **13.33 / 44.67** for SIGReg
alone. The implementation was audited correct; the failure is a
coefficient-regime + geometry-overwrite problem, documented in depth below.

---

## 1. Read this first

| # | document | what it contains |
|---|---|---|
| 1 | `FAILURE_LOG_VICREG_ARM2.md` | **the failure log itself**: metadata, verbatim run command, loss-composition tables, latent-health record, eval record, implementation audit, 4-part root-cause chain (§6), in-depth worked scenario (§7), ruled-out hypotheses (§8), follow-up queue (§9), artifact index |
| 2 | `PROGRESS_VICREG.md` | experiment manifest: run matrix, comparability contract, pre-registered kill criteria, pod queue, recorded results table |
| 3 | `PROGRESS_SIGREG_E2E.md` | full history of the SIGReg e2e arms whose numbers serve as the comparison references (13.33/44.67 and 30/40 grounded) |
| 4 | `REPRODUCTION.md` + `POD_SETUP_LOG.md` | exact protocol: pod B200 setup, data layout, training/eval reproduction rules |
| 5 | `SHORT_BUDGET_PILOTS.md` | pilot-first methodology this program follows (1h pilots gate full budgets) |
| 6 | `MANIFEST.md` | exhaustive file-by-file manifest of everything in this bundle, with sizes — proof the full code tree is included |

## 2. All results obtained

### 2.1 PushT planning results (identical protocol, `reproduce_table1.py`; GD open-loop `mode=last`, MPC `mode=staged`, α=1, seeds 100/200/300, 50 samples, goal_H=25)

| arm | Open-loop | MPC | MPC÷OL rescue |
|---|---|---|---|
| paper ✓ frozen baseline (straighten, sg=True) | 77.33 ± 6.18 | 85.33 ± 4.99 | 1.10× |
| e2e + SIGReg (ungrounded) | 13.33 ± 1.15 | 44.67 ± 10.26 | 3.35× |
| e2e + SIGReg + grounding | 30.00 ± 7.21 | 40.00 ± 10.39 | 1.33× |
| **arm 2: e2e + VICReg(25/1) + SIGReg** | **4.67 ± 1.15** (seeds 4/4/6) | **25.33 ± 2.31** (seeds 24/28/24) | **5.42×** |

Arm-2 plan times: gd 80.1 s · gd_mpc 1970.8 s (mean per seed).
The rescue factor climbs monotonically with metric damage (1.10 → 3.35 → 5.42):
the worse the latent geometry, the more closed-loop re-planning must compensate.

### 2.2 Arm-2 training record (telemetry digest, 620 intervals)

- Budget: 123,858 steps reached cleanly (epoch 2 of 3), 13.88 h wall, git `a9d08c1`.
- `z_loss` (prediction): 0.3125 → 0.0175 (6k) → **0.0018** (end).
- VICReg scaled term: 0.484 (step 200) → 0.154 (6k) → **0.146** (end) —
  the cov component is **56% of the total loss and ≈81× z_loss at the end**.
- `z_vicreg_std_loss`: 0.0132 → 8.6e-5 → 1.2e-5 (hinge satisfied early).
- `z_vicreg_cov_loss`: 0.154 → 0.146 (low flat equilibrium, never vanishes).
- SIGReg: 4.12 → 1.098, at the batch-32 null floor 1.058.
- Visual `probe_r2`: 0.157 → 0.339 (peak 0.494) — no collapse, information intact.
- `latent_eff_rank_frac` 0.873 → 0.875; decoder recon 0.128 → 0.0015.
- `agg_probe_r2`: −0.59 (step 600) then pinned −1.0000 (held-out ridge clamp on
  an underdetermined 64-row/128-dim fit — explained in the failure log §6.5).
- One transient spike at step 77,286 (std 15.1σ, cov 45.2σ), self-recovered.
- Summariser verdict: **OK — healthy run**.

## 3. Why it failed (summary of `FAILURE_LOG_VICREG_ARM2.md` §6–7)

1. **Loss-composition inversion.** The VICReg anchors (25/1) were calibrated
   against an invariance term that stays O(0.1–1). Our prediction term has no
   such floor — a strong predictor drives it to 0.0018 — so from step ~6k on
   the covariance term *is* the objective: 56% of total loss, ≈81× prediction.
2. **Coefficient-regime mismatch.** Anchors calibrated for D=2048, batch 2048,
   applied to D=8 projected tokens, batch 32: each of the 56 off-diagonal
   covariance entries is a large fraction of the term, inflating its relative
   weight orders of magnitude beyond the intended regime.
3. **Object mismatch.** VICReg whitens the raw visual token field
   (b,t,196,8) — the exact geometry GD planning walks on — while SIGReg
   (`apply_to=agg`) only sees aggregated latents and therefore *cannot see or
   compensate* the token-level overwrite.
4. **Consequence.** Unit-variance inflation + decorrelation erodes the local
   metric that maps latent distance to state distance. Open-loop GD (25-step
   trust) collapses to 4.67%; MPC's 5-step re-planning limits error
   accumulation, rescuing to 25.33%. Every conventional health gate (probe R²,
   rank, std) stayed green because they measure *information*, not *metric*.

Worked scenarios with concrete numbers: failure log §7 (std-hinge sledgehammer
at init vs silence at 6k; persistent O(0.14) whitening gradient vs O(0.002)
prediction gradient; a step-100k gradient step narrated; the planner's-eye
view at eval time).

## 4. Code map — where each piece of the story lives

| area | files | role in this experiment |
|---|---|---|
| objective | `models/visual_world_model.py` | `vcreg_std_loss` / `vcreg_cov_loss` (paper-faithful formulas), SIGReg block, loss composition (≈ L471–482, L731–745) |
| regularisers | `models/sigreg.py`, `models/diagnostics.py` | characteristic-function term; probe R² with [−1,1] clamp (explains the −1.0000) |
| encoders | `models/dino.py`, `models/encoder/resnet.py`, `models/encoder/vit.py` | DINO-channel encoder used in all arms |
| training | `train.py`, `training_log.py`, `summarize_training_log.py`, `iteration_budget.py`, `run_naming.py` | telemetry JSONL, digest, 123,858-step budget, folder naming |
| planning | `planning/gd.py`, `planning/mpc.py`, `planning/objectives.py`, `planning/evaluator.py`, `planning/base_planner.py`, `planning/cem.py`, `plan.py` | the GD/MPC protocols that exposed the failure |
| eval protocol | `reproduce_table1.py`, `summarize_run.py`, `collect_results.py`, `aggregate_results.py`, `evaluate.sh`, `eval_pusht_3seeds.sh`, `redo_pusht_paperexact.sh`, `run_final_protocol.sh` | identical protocol across all arms; timing sidecars |
| environments | `env/pusht/`, `env/wall/`, `env/pointmaze/`, `env/venv.py`, `env/serial_vector_env.py` | PushT is the failing environment |
| datasets | `datasets/*.py`, `preprocessor.py` | trajectory datasets per env |
| diagnostics | `experiments/metric_alignment.py` (+ `curvature_incentive.py`, `planning_jacobian.py`, `planning_landscape.py`, `rollout_drift.py`, `verify_encoder_trains.py`, `verify_stop_grad.py`, `violation_of_expectation.py`) | `rho_local` / `nn_state_ratio` — the metric-alignment probe whose absence let the failure hide (follow-up §9.4) |
| tests | `tests/test_vicreg_sigreg.py` (+ 10 more) | fidelity suite: formulas vs official VICReg algebra, whitening in float64, numerical cross-checks — all green |
| config | `conf/**/*.yaml` | hydra config groups incl. `conf/train.yaml` |
| metrics | `metrics/image_metrics.py`, `metrics/lpipsPyTorch/` | reconstruction metrics |

## 5. What is deliberately NOT in the bundle

- `.git/`, `__pycache__/`, caches, `.ipynb_checkpoints/` — noise.
- `research_papers/` — third-party reference implementations and archives
  (multi-GB), not produced by this experiment.
- `env/deformable_env/` — multi-MB CAD meshes unrelated to PushT/VICReg.
- giant local training logs from *older* arms (`train_channel_*.log`,
  `train_pusht_channel_*.log`, ~30 MB each) and large binaries
  (`2603.12231v2.pdf`, `*.tar.gz`, `temporal_straightening_original.zip`,
  `TEACHING_BUNDLE.zip`).
- Pod-only artifacts (referenced, not bundled): the arm-2 telemetry JSONL
  (`<run>/telemetry/train_20260811_185139.jsonl`), training log
  `train_vicreg_full_20260811_185133.log`, eval log
  `eval_vicreg_20260812_141157.log`, `results/*.json` and checkpoints under
  `/workspace/arun/temporal_straightening_with_vicreg/checkpoints/test/`.
  All numbers from them are transcribed in §2 above and in the failure log.

## 6. Decisions carried forward

1. Frozen-baseline retrain → eval (queue item 1).
2. **Weak-coefficient pilot** `training.vcreg_std_coeff=1
   training.vcreg_cov_coeff=0.04` (8k steps ≈ 1 h) — isolates the mechanism;
   success = planning within noise of the SIGReg-only arm.
3. Arm-3 grounded (`ground_proprio=1.0`, strong anchors) if (2) confirms.
4. Add metric-alignment diagnostics (`rho_local`, `nn_state_ratio`) to
   training-time telemetry so metric damage is visible during training.
5. Paper framing: arm 2 is a clean, well-evidenced negative result.

Git provenance: training code `a9d08c1`; docs `d9f47ef`, `6a9597c`, `8c135e6`,
`2b3b487`. Remote: `https://github.com/Subaru-5999/temporal_straightening_with_vicreg`.
