# FAILURE LOG — Arm 2: "World Model + VICReg + SIGReg" (PushT)

**Status: TRAINING HEALTHY / PLANNING REGRESSED.** The combined objective trained
exactly as specified and produced a non-collapsed, well-regularised latent — and
then planned *worse* than SIGReg alone on both protocols. This log records the
full evidence chain, the root-cause analysis, an in-depth worked scenario of the
failure mechanism, the hypotheses that were ruled out, and the follow-up queue.

---

## 0. Metadata

| field | value |
|---|---|
| run folder | `checkpoints/test/pusht_False_agg32_projchannel_dim8_hw14_sgFalse_lr1e-05_sig1e-1_e2e_vic_s2e1_c1e0` |
| objective | `L = L_pred + 25·VICReg-std(Z_vis) + 1·VICReg-cov(Z_vis) + 0.1·SIGReg(Z_agg)` |
| training git commit | `a9d08c1` (recorded in telemetry config) |
| hardware | B200 MIG `1g.45gb` slice, torch 2.7.0+cu128, bf16 |
| budget | 123,858 steps (paper-exact), stopped mid-epoch 2 of 3 |
| wall clock | 2026-08-11 18:51:39 → 2026-08-12 08:44:50 (13.88 h, 2.23–2.49 it/s) |
| telemetry | `<run>/telemetry/train_20260811_185139.jsonl`, 620 intervals, digest via `summarize_training_log.py` |
| eval | `reproduce_table1.py <run>` — GD open-loop `mode=last`, MPC `mode=staged`, α=1, seeds 100/200/300, 50 samples, goal_H=25 |
| **result** | **OL 4.67 ± 1.15 (seeds 4/4/6) · MPC 25.33 ± 2.31 (seeds 24/28/24)** |
| plan time | gd 80.1 s · gd_mpc 1970.8 s (mean per seed) |
| doc commits | training summary `d9f47ef`, eval result `6a9597c`, audit corrections `8c135e6` |

### References (identical protocol)

| arm | Open-loop | MPC |
|---|---|---|
| paper ✓ frozen baseline (straighten, sg=True) | 77.33 ± 6.18 | 85.33 ± 4.99 |
| e2e + SIGReg (ungrounded) | 13.33 ± 1.15 | 44.67 ± 10.26 |
| e2e + SIGReg + grounding | 30.00 ± 7.21 | 40.00 ± 10.39 |
| **this run: e2e + VICReg(25/1) + SIGReg** | **4.67 ± 1.15** | **25.33 ± 2.31** |

Adding VICReg at the paper anchors lowered success on *both* protocols versus
SIGReg alone: OL by ~65% (13.33 → 4.67), MPC by ~43% (44.67 → 25.33).

---

## 1. What was intended

Hypothesis (phase start): SIGReg pins the *full* latent distribution through the
characteristic function but only sees the aggregated trajectory latent; VICReg
(Bardes, Ponce & LeCun, ICLR 2022, arXiv:2105.04906) supplies an *explicit*
second-order guard (per-dim variance hinge + pairwise covariance) directly on
the visual token field, stabilising end-to-end training where SIGReg-only e2e
arms under-planned. Anchors taken verbatim from the VICReg paper
(`λ_var=25, λ_cov=1`) and LeWM (`λ_SIG=0.1, M=1024`); zero free knobs.

Comparability contract with the baseline arm: identical
`encoder_lr=1e-5`, `predictor_lr=5e-4`, batch 32, `num_hist=3`, `num_pred=1`,
frameskip 5, budget 123,858, `encoder=dino_channel`, seed 0; only the objective
and trainability differ (`stop_grad=False`, `freeze_backbone=False` here).

Pre-registered kill/fallback criteria (PROGRESS_VICREG.md §6): collapse
(`val_probe_r2 → 0`), or `z_vicreg_std_loss` pinned ~1 while probe falls →
relaunch with `std=1, cov=0.04`. **Neither fired.** The failure mode that
materialised was different, and is the subject of §6.

---

## 2. What was run (verbatim)

```bash
setsid nohup python train.py --config-name train.yaml \
  env=pusht encoder=dino_channel \
  training.vcreg=True training.vcreg_std_coeff=25 training.vcreg_cov_coeff=1 \
  training.vcreg_apply_to=visual training.straighten=False \
  training.sigreg=True training.sigreg_coeff=0.1 training.sigreg_num_proj=1024 \
  training.sigreg_apply_to=agg training.stop_grad=False training.freeze_backbone=False \
  training.encoder_lr=1e-5 training.backbone_lr=1e-5 training.epochs=3 \
  training.max_iterations=123858 env.num_workers=4 \
  ckpt_base_path=/workspace/arun/temporal_straightening_with_vicreg/checkpoints \
  > train_vicreg_full_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

Launch verification (all correct):
`VCReg enabled: True, apply_to=visual, std_coeff=25.0, cov_coeff=1.0` ·
`SIGReg enabled: coeff=0.1, num_proj=1024, knots=17, apply_to=agg` ·
`Combined objective active: World Model + VICReg (std=25.0, cov=1.0, on visual)
+ SIGReg (coeff=0.1)` · `Encoder base_model is TRAINABLE (end-to-end)` ·
`Stop-grad enabled: False` · `Iteration budget: effective total steps: 123,858,
cap binds first: stopping mid-epoch 2 of 3` · trunk 22,056,576 params @ 1e-5.

Termination (correct): `Iteration budget reached: global_iter=123858 /
max_iterations=123858 (epoch 2, batch 61928 of 61929)`; final checkpoint saved;
epoch-2 train 0.2613 / val 0.2926.

---

## 3. Training record (telemetry digest, interval means)

### 3.1 Loss composition over the run

| step | total | z_loss (pred) | VICReg scaled | of which cov | of which std | SIGReg scaled | VICReg share |
|---|---|---|---|---|---|---|---|
| 200 | 1.337 | 0.3125 | 0.484 | 0.154 | 0.0132 | 0.412 | 36% |
| 6,000 | 0.308 | 0.0175 | 0.154 | 0.152 | 8.6e-5 | 0.133 | 50% |
| 24,000 | 0.280 | 0.0060 | 0.152 | 0.151 | 2.7e-5 | 0.120 | 54% |
| 60,000 | 0.265 | 0.0029 | 0.148 | 0.148 | 2.3e-5 | 0.112 | 56% |
| 96,000 | 0.260 | 0.0022 | 0.147 | 0.147 | 1.9e-5 | 0.109 | 57% |
| 123,858 | 0.259 | **0.0018** | **0.146** | 0.146 | 1.2e-5 | 0.110 | **56% (≈81× z_loss)** |

Decoder recon fell 0.128 → 0.0015 in the same window. The prediction term
converges two orders of magnitude further than the cov term ever does; from
~step 6k onward the **covariance term is the single largest component of the
objective and the dominant source of encoder gradient**.

### 3.2 Latent health (no collapse)

- visual `probe_r2`: 0.157 → 0.339 (peak 0.494) — state readability *improved*.
- visual `latent_eff_rank_frac`: 0.873 → 0.875; `latent_std`: 0.39 → 0.26.
- agg `latent_std` 0.94–0.97; `agg_latent_abs_mean` ≈ 0.77 vs Gaussian 0.798.
- `sigreg_loss` 4.12 → 1.098, i.e. converged onto its batch-32 null floor 1.058.
- `z_vicreg_std_loss` ≈ 1e-5 from step 6k: the variance hinge is *satisfied*
  (every token channel at σ ≥ 1), not crushing the representation.
- `z_vicreg_cov_loss` 0.154 → 0.146: a low, flat, non-zero equilibrium.
- Summariser verdict: **OK — "Prediction loss fell while probe R² held or
  improved: this is what a healthy run looks like."**

### 3.3 Gradients / events

- `grad/encoder.trunk/ratio`: 3.11 (step 200, init transient) → 1.4e-2 (6k) →
  8.4e-4 (end): bounded, decaying, no explosion/vanishing.
- One transient spike at step 77,286 (`z_vicreg_std_loss` 15.1σ,
  `z_vicreg_cov_loss` 45.2σ of the window); adjacent windows on trend.
  Single-batch artifact; no intervention, no recurrence.
- `agg_probe_r2`: −0.59 at step 600, then **pinned at exactly −1.0000 from
  step 6,600 to the end** (see §6.5).
- Curvature without any straightening term: `agg_curvature_cos` −0.29 → +0.30,
  visual `curvature_cos` −0.20 → +0.43 (trajectories self-straighten somewhat).

---

## 4. Eval record

```
RESULT  pusht_..._sig1e-1_e2e_vic_s2e1_c1e0
  Open-loop  ours 4.67+/-1.15     seeds: 4, 4, 6      (of 50)
  MPC        ours 25.33+/-2.31    seeds: 24, 28, 24    (of 50)
  plan time  gd 80.1s | gd_mpc 1970.8s
```

Quantitative pattern — **MPC rescue factor = MPC ÷ OL success**:

| arm | OL | MPC | rescue factor |
|---|---|---|---|
| frozen baseline | 77.33 | 85.33 | 1.10× |
| e2e + SIGReg | 13.33 | 44.67 | 3.35× |
| **e2e + VICReg + SIGReg** | **4.67** | **25.33** | **5.42×** |

The worse the arm's latent geometry, the more it depends on closed-loop
re-planning. The baseline's metric is good enough that re-planning adds 10%;
VICReg's metric is so degraded that open-loop execution is near chance and MPC
salvages only a quarter of the trials. The tight MPC spread (±2.31 vs SIGReg's
±10.26) says *consistently* mediocre geometry, not instability.

---

## 5. Implementation audit — the code is correct

Full read of `models/visual_world_model.py` plus the fidelity suite:

1. **Formulas paper-faithful.** `vcreg_std_loss`: `mean(relu(1 − sqrt(var+1e-4)))`
   per dim over `reshape(-1, D)`; `vcreg_cov_loss`: centred covariance with the
   (n−1) divisor, squared off-diagonals summed ÷ D. Identical algebra to the
   official implementation; cross-checked independently in
   `tests/test_vicreg_sigreg.py` (`torch.std`, `torch.cov`, float64 whitening:
   both losses ≈ 0 on whitened unit-variance inputs, std hinge one-sided).
2. **Integration clean.** Term computed on the *live* `visual_only(z)`
   (b, t, 196, 8 → N = 32·4·196 = 25,088 samples, D = 8), gradients flow to the
   trainable trunk/projector, added to the total exactly once with 25/1; SIGReg
   added separately on `_agg_tokens(visual_only(z))`. No detach, no double
   count, no wrong tensor, coefficients exactly as launched (log line §2).
3. **One documentation error found and fixed (commit `8c135e6`):** the code
   comment and PROGRESS_VICREG §1 claimed VICReg "acts on exactly the tokens
   SIGReg sees". False: SIGReg with `apply_to=agg` sees the *aggregated*
   latents; VICReg sees the raw token field. The two terms shape **different
   objects** — and that object mismatch is load-bearing for the failure (§6.3).

Conclusion: the run executed the intended mathematics correctly. The failure is
a **design/regime failure**, not an implementation bug.

---

## 6. Root-cause analysis

### 6.1 Loss-composition inversion: the regulariser became the objective

In the VICReg paper the *invariance* term (their counterpart of our `z_loss`)
stays O(0.1–1) throughout training — a hard self-supervised task — so 25·std +
1·cov remain balanced *regularisers*. In our world model the prediction term is
not bounded below by task difficulty: a powerful predictor driving a
trainable encoder pushes `z_loss` to 0.0018. Any fixed-coefficient regulariser
then necessarily dominates the late objective. Measured: from step ~6k to the
end, the cov term is 50–81× the prediction term and >50% of the total loss.
The trunk's slow updates (lr 1e-5) were therefore steered, for ~118k steps,
predominantly by **whitening pressure**, with dynamics fidelity a side
constraint. The encoder ended training as a good whitener whose world model
happens to predict well in-sample.

### 6.2 Coefficient-regime mismatch (D = 8, not 2048)

The anchors 25/1 were calibrated for D = 2048 with batch 2048. Here D = 8:
the cov loss has only D(D−1) = 56 off-diagonal entries, each a *large* fraction
of the term, and the std hinge acts on just 8 channels of the projected token
field. The relative weight of the second-order terms versus the (vanishing)
prediction term is therefore orders of magnitude larger than in the regime the
anchors were tuned for. The VICReg paper itself sweeps these coefficients with
batch size; we deliberately launched at the anchors and pre-registered the
fallback `std=1, cov=0.04` for exactly this eventuality. This run *is* that
eventuality.

### 6.3 Object mismatch: whitening the field SIGReg cannot see

VICReg whitens the raw token field; SIGReg only observes the aggregated latent
`h(tokens)`. Token-level distortion is therefore **invisible to the
distributional term** and cannot be compensated by it. Channel-whitening and
σ ≥ 1 inflation reorganise the token geometry: directions that carried small
but spatially meaningful signal are inflated to unit variance alongside noise
directions. GD planning walks the *local metric* of exactly this field
(`rho_local` in `experiments/metric_alignment.py` is what gradient descent
exploits); whitening erodes the channel correlations that make latent distances
track state distances. Hence §4's pattern: open-loop GD (which trusts the
metric for 25 executed steps) collapses to 4.67, while MPC (which re-queries
the model every 5 steps) survives at 25.33.

### 6.4 Why healthy training metrics did not warn us

All collapse gates watch *information* (probe R², rank, std) — and information
was fine: the field still carries state (visual probe 0.34). What died is the
**metric structure** that linear probes do not measure: a representation can be
informative and still have a local geometry that gradient descent cannot walk.
The one telemetry signal that did show it was `agg_probe_r2` (§6.5), plus the
post-hoc MPC-rescue-factor ladder (§4). Lesson recorded: for planning papers,
training health must include a *metric-alignment* diagnostic
(`experiments/metric_alignment.py`: `rho_local`, `nn_state_ratio`), not only
information probes.

### 6.5 `agg_probe_r2 = −1.0000` explained (clamp, not −∞)

`models/diagnostics.probe_r2` is a *held-out* ridge probe clamped to [−1, 1].
On the 128-dim agg latent it fits 129 coefficients from 64 interleaved train
rows (32×4 val batch ÷ 2) — an underdetermined fit that scores below the
held-out mean unless the linear signal is strong, hence the pinned clamp from
step 6.6k on. Interpretation: the agg path is not *linearly* state-readable —
a known property of the agg bottleneck (planning does not require linear
readability), consistent with §6.3's geometry degradation but not itself a
failure signal. The visual probe (well-posed: 196·8 features pooled over many
samples) is the information gate, and it stayed green.

---

## 7. In-depth worked scenario

Numbers below marked *(illustrative)* are a concrete instantiation of the
measured regime, for intuition; all unmarked numbers are from the telemetry.

### 7.1 The std hinge at launch vs at step 6k

At init, a token channel with std s = 0.3 contributes
`relu(1 − sqrt(0.09 + 1e-4)) = 1 − 0.3002 ≈ 0.70`; with λ_var = 25 that is a
~17.5 gradient pressure per deficient channel — the hinge is a sledgehammer
early (measured: std term 0.0132·25 ≈ 0.33 of the step-200 loss). By step 6k
every channel satisfies σ ≥ 1 (measured std loss 8.6e-5) and the term goes
quiet. *Effect*: all 8 channels of the token field are permanently held at
unit variance — including channels whose natural, informative scale was 0.3.
The field's anisotropy (which directions matter) is erased at second order.

### 7.2 The cov term's permanent pressure

Two channels with correlation ρ and unit variance contribute ρ² to the squared
off-diagonal sum; ÷ D = 8 that is ρ²/8 per pair. The measured equilibrium
0.146 implies an rms off-diagonal covariance ≈ sqrt(0.146·8/56) ≈ 0.14 — the
field *is* largely decorrelated, and that is precisely the damage: the
decorrelation is maintained by a **persistent gradient** of O(0.14) per step
against a prediction gradient of O(0.002). *(Illustrative)* Over the final
60k steps at lr 1e-5, the trunk's cumulative displacement direction is set by
the whitener: measured `delta/encoder.trunk` stays ≈ 0.009–0.010 per probe
interval at the end of training — the trunk never stops moving, and what it is
moving *for* is the cov term.

### 7.3 A gradient step at step 100,000 (narrative)

One batch arrives. The predictor, already excellent, produces z_loss ≈ 0.002:
its backprop into the trunk is a whisper — "nudge the dynamics, a little,
here". The cov term evaluates to ≈ 0.147: its backprop is a shout — "these
eight channels still correlate at the 0.1 level; rotate the field". The std
hinge adds "and keep every channel at σ ≥ 1". The Adam update at lr 1e-5 is
therefore ≈ the whitening direction plus noise from the whisper. Multiply by
24k remaining steps: the token field converges to a geometry whose second-order
statistics are canonical (σ = 1, ρ ≈ 0.14 rms) and whose *task-relevant local
metric* has been steadily overwritten. SIGReg, watching only `h(tokens)`,
reports a perfectly Gaussian aggregate (agg abs-mean 0.772 vs 0.798) and cannot
see the overwrite. Training looks exemplary in every logged scalar.

### 7.4 The planner's-eye view at eval time

Open-loop GD initialises 25 actions at zero and descends
`‖z_pred(a) − z_goal‖²` through the latent. *(Illustrative)* In the baseline's
field, a 1-unit state displacement produces a latent displacement whose
direction is stable across the neighbourhood, so 100 Adam steps at lr 0.1 walk
coherently toward the goal (77% success). In the whitened field, neighbouring
states map to tokens whose distances are dominated by inflated, decorrelated
noise directions: the descent direction decorrelates from the true state
gradient within a few steps, the plan oscillates, and 25 executed steps carry
the block nowhere (4.67%). MPC re-solves every 5 executed steps against the
*actual* new observation, discarding the diverged tail of each bad plan —
hence 25.33%: five-step horizons are short enough that the metric's local
error does not fully accumulate. The 5.42× rescue factor is the size of the
metric damage.

### 7.5 Counterfactuals

- **std=1, cov=0.04 (pre-registered fallback):** the cov shout drops ~25×,
  below the prediction whisper's cumulative influence only late; the metric
  overwrite should largely vanish. If planning recovers toward 13.33/44.67,
  §6.1–6.2 are confirmed as the mechanism (cheap 1h pilot decides).
- **Grounding (arm 3):** the proprio anchor re-injects metric information at
  the token level (it helped SIGReg OL 13.33 → 30.00); it may partially resist
  the whitening even at strong anchors. Tests a repair, not the mechanism.

---

## 8. Ruled-out hypotheses

| hypothesis | evidence against | verdict |
|---|---|---|
| implementation bug | §5 audit + 16 fidelity tests; log lines match config | ruled out |
| representational collapse | probe R² ↑, rank 0.875, std healthy, verdict OK | ruled out |
| optimisation instability | smooth losses, bounded grads, spike recovered | ruled out |
| VICReg crushing variance | std loss ≈ 1e-5 (satisfied), not pinned ~1 | ruled out |
| under-training | paper-exact budget; losses still falling at cap | ruled out |
| eval protocol drift | identical `reproduce_table1.py` protocol as all arms | ruled out |
| seed luck | 3 seeds, tight ±1.15/±2.31 spreads, monotone rescue ladder | ruled out |
| **coefficient-regime dominance + metric overwrite** | §3.1 composition inversion, §4 rescue ladder, §6.3 object mismatch | **accepted** |

---

## 9. Decisions and follow-up queue

1. **Baseline retrain** (comparison cell) — full command in PROGRESS_VICREG §6;
   folder `pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`.
2. **Weak-VICReg pilot** (`training.vcreg_std_coeff=1
   training.vcreg_cov_coeff=0.04`, else identical, 8k steps ≈ 1 h): isolates
   §6.1/6.2. Success criterion: planning within noise of the SIGReg-only arm.
3. **Arm 3 grounded** (`training.ground_proprio=1.0`, strong anchors, full
   budget): tests the §7.5 repair branch if the pilot confirms the mechanism.
4. **Diagnostic upgrade (future runs):** add a metric-alignment probe
   (`rho_local`, `nn_state_ratio` from `experiments/metric_alignment.py`) to
   the training-time diagnostics so metric damage is visible *during* training,
   not only at eval.
5. Paper framing: arm 2 is a clean negative result — SIGReg's distributional
   match already supplies the regularisation the e2e arm needs; a strong
   second-order prior on the token field is redundant-at-best, and at
   out-of-regime coefficients it overwrites the planning metric while every
   conventional health metric stays green.

---

## Appendix A — exact commands

Training: §2. Eval:

```bash
setsid nohup python reproduce_table1.py \
  pusht_False_agg32_projchannel_dim8_hw14_sgFalse_lr1e-05_sig1e-1_e2e_vic_s2e1_c1e0 \
  > eval_vicreg_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

Digest: `python summarize_training_log.py <run>/telemetry/train_20260811_185139.jsonl`.

## Appendix B — artifact index

- telemetry: `<run>/telemetry/train_20260811_185139.jsonl`
- training log: `train_vicreg_full_20260811_185133.log`
- eval log: `eval_vicreg_20260812_141157.log`
- results: `results/pusht_False_..._vic_s2e1_c1e0.json`, `.timing.json`,
  `results/table1_reproduction.{md,csv}`
- manifests: `PROGRESS_VICREG.md` (§2 matrix, §5 results), `PROGRESS_SIGREG_E2E.md`
- code: `models/visual_world_model.py` (vcreg block L471–482, L731–745),
  `models/diagnostics.py` (probe clamp L96–121), `tests/test_vicreg_sigreg.py`
- commits: `a9d08c1` (trained code), `d9f47ef` (training summary),
  `6a9597c` (eval result), `8c135e6` (audit corrections)
