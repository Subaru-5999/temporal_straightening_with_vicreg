# Progress: VICReg phase — World Model + VICReg + SIGReg

Record of the phase that adds the VICReg term (Bardes, Ponce & LeCun, ICLR 2022,
arXiv:2105.04906) to the LeWM-style objective of
`research_papers/ICLR_Submission_Plan.md`. Operational pod docs:
`POD_SETUP_LOG.md`, `REPRODUCTION.md`; predecessor record: `PROGRESS_SIGREG_E2E.md`.

Every run below is launched only through the scripts listed, on the B200 MIG
`1g.45gb` slice (one job at a time; the scripts hard-gate this), at
`/workspace/arun/temporal_straightening_with_vicreg`.

---

## 1. Objective

The plan's unified objective, plus the extra VICReg term:

```
L = L_pred
  + lambda_var * VICReg-variance(Z)     relu(1 - sqrt(Var(z_i) + 1e-4)) per dim
  + lambda_cov * VICReg-covariance(Z)   off-diag(cov(Z))^2 / D
  + lambda_SIG * SIGReg(Z)              sketched isotropic Gaussianity (LeWM)
  [+ lambda_curv * L_curv               temporal straightening, optional stack]
```

Division of labor: the prediction loss plays VICReg's invariance term; VICReg
pins the second-order structure explicitly; SIGReg pins the full distribution
through the characteristic function; curvature (when on) straightens latent
trajectories. Implemented in `models/visual_world_model.py`
(`vcreg_std_loss` / `vcreg_cov_loss`, `vcreg_apply_to="visual"` = VICReg acts on
exactly the tokens SIGReg sees). Fidelity vs the official reference
implementation is pinned by `tests/test_vicreg_sigreg.py` (independent
`torch.std` / `torch.cov` / whitening cross-checks).

Hyperparameter anchors (zero free knobs beyond the two papers):

| knob | value | anchor |
|---|---|---|
| `lambda_var` | 25 | VICReg paper (batch 2048 / 2048-d; sweep down at batch 32) |
| `lambda_cov` | 1 | VICReg paper, same caveat |
| `lambda_SIG` | 0.1, M=1024 | LeWM sec. 3 |
| `lambda_curv` | 0.1 (`aggcos1e-1`) | Straightening App. B.6 |
| budget | 123,858 steps | Straightening App. A.3 (PushT 2 epochs x 61,929) |

---

## 2. Run matrix (PushT) — each arm in its own folder

Folder names come from `run_naming.variant_tag`; a different objective is a
different directory, so `train.py` auto-resume can never mix arms. **Always
read the folder back from the log; never assume it.**

| # | arm | script | run folder under `checkpoints/test/` | budget | status |
|---|---|---|---|---|---|
| 1 | frozen baseline, straightening ✓ (paper cell) | `train_pusht_on_paperiters.sh` | `pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05` | 123,858 | **retrain queued** (old pod ckpt lost) |
| 2 | VICReg + SIGReg, e2e | `train_pusht_vicreg_sigreg.sh` | `pusht_False_agg32_projchannel_dim8_hw14_sgFalse_lr1e-05_sig1e-1_e2e_vic_s2e1_c1e0` | 8k pilot -> 123,858 | **trained 2026-08-11/12 @ a9d08c1, eval queued** |
| 3 | VICReg + SIGReg + grounding, e2e | same + `training.ground_proprio=1.0` | `..._sig1e-1_e2e_gp1e0_vic_s2e1_c1e0` | 123,858 | queued |
| 4 | ablation VICReg-only | `SIGREG=0 bash train_pusht_vicreg_sigreg.sh` | `..._sgFalse_lr1e-05_e2e_vic_s2e1_c1e0` | pilot | optional |
| 5 | ablation SIGReg-only | `VCREG=0 bash train_pusht_vicreg_sigreg.sh` | `..._sig1e-1_e2e` (exists from SIGReg phase) | done | see PROGRESS_SIGREG_E2E.md |

Baseline script passes every contract knob explicitly (`stop_grad=True
freeze_backbone=True sigreg=False vcreg=False ground_proprio=0
backbone_lr=null`), so the recorded command line alone defines the baseline.

## 3. Comparability contract (baseline vs VICReg arms)

Only the objective/trainability may differ. Must match (verified from both
runs' telemetry configs): `straighten` per arm, `encoder_lr=1e-5`,
`predictor_lr=5e-4`, `action_encoder_lr=5e-4`, `batch_size=32`, `num_hist=3`,
`num_pred=1`, `frameskip=5`, `max_iterations=123858`, `epochs=3`,
`encoder=dino_channel`, `training.seed=0`.

Eval protocol — identical for all arms, no exceptions: `reproduce_table1.py
<run_name>` (applies internally: open-loop `mode=last`, MPC `mode=staged`,
`alpha=1`, seeds 100/200/300, 50 samples, `goal_H=25`). Never hand-rolled
`plan.py` calls that can drift between arms.

## 4. Traceability chain (what makes every run auditable)

1. **Launch**: timestamped log `train_<script>_<ts>.log` + `.train_pid`; MIG
   pre-flight gate refuses a second concurrent job.
2. **Command line**: every contract knob explicit in the log header, plus
   `git commit = <hash>` echoed at launch.
3. **Run dir**: hydra-resolved from the objective via `variant_tag`
   (`_vic_s<var>_c<cov>` suffix); read back with
   `grep -m1 -oE "checkpoints/test/[^ ]*" <log>`.
4. **Telemetry**: `<run_dir>/telemetry/*.jsonl` records the full config incl.
   `vcreg*`, `ground_proprio`, `git_commit` (train.py `_git_commit()`), every
   loss term, collapse metrics (`val_probe_r2`, `val_latent_eff_rank`),
   gradient/weight norms. Digest: `summarize_training_log.py`.
5. **Eval**: `plan_outputs_gd*/<run>_*/logs.json`, basename-scoped;
   `reproduce_table1.py` writes a per-run timing sidecar.

## 5. Reference numbers (PushT, 3 planning seeds)

| arm | Open-loop | MPC |
|---|---|---|
| paper ✓ frozen baseline | 77.33 ± 6.18 | 85.33 ± 4.99 |
| e2e + SIGReg (ungrounded) | 13.33 ± 1.15 | 44.67 ± 10.26 |
| e2e + SIGReg + grounding | 30.00 ± 7.21 | 40.00 ± 10.39 |

VICReg arms must be compared against these at the identical protocol (§3).

### Arm 2 training summary (123,858 steps, 13.88 h, B200 MIG, git a9d08c1)

- `z_loss` 0.3125 -> 0.0018; decoder recon 0.128 -> 0.0015; still falling at the cap.
- `sigreg_loss` 4.12 -> 1.098 (null floor 1.058); `z_vicreg_std_loss` ~1e-5 from
  step 6k (hinge satisfied, no crushing); `z_vicreg_cov_loss` 0.154 -> 0.146 flat.
- One transient spike at step 77,286 (std 15 sigma / cov 45 sigma of the window);
  adjacent windows on trend, no action.
- Collapse verdict OK: visual `probe_r2` 0.157 -> 0.339 (peak 0.494), eff-rank
  frac 0.875, `latent_std` 0.26.
- `agg_probe_r2` pinned at the -1 clamp for the whole run: held-out ridge probe
  (64 train rows vs 128 agg dims) is underdetermined; agg path is not linearly
  state-readable by construction and planning does not need it to be. Known
  property of the agg bottleneck, not a VICReg regression.
- Curvature without a straightening term: `agg_curvature_cos` -0.29 -> +0.30,
  visual +0.43 — compare against the aggcos1e-1 baseline after its retrain.

## 6. Pod queue (strictly serial, one MIG slice)

1. `git pull` -> verify `git log --oneline -1` matches this commit; CPU gate
   `python -m pytest tests/test_vicreg_sigreg.py -q`.
2. Baseline retrain, arm 1 (~14 h).
3. VICReg 8k pilot, arm 2 (~1 h). Kill criterion: `z_vicreg_std_loss` pinned
   ~1 while `val_probe_r2` falls -> relaunch `VCREG_STD=1 VCREG_COV=0.04`.
4. Full arm 2 and/or arm 3 (~14 h each).
5. `reproduce_table1.py` on baseline + VICReg arms; record numbers here.
