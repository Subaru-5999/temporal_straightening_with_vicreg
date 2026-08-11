# ICLR Teaching Guide — Straightened Latent World Models (Complete Implementation Package)

> **Purpose.** This document is a self-contained teaching package. A reader (human or a
> smaller LLM acting as tutor) should be able to teach, from it alone, every concept the
> project uses and every component actually implemented in this codebase, in the order a
> learner needs them. Every claim is anchored to a file (and line where useful) so the
> tutor can point at real code, not abstractions.
>
> **Companion plan.** The research plan this implements: *"Straightened Latent World
> Models: Combining Gaussian Regularization and Temporal Straightening for Stable
> End-to-End Planning"* — take LeWorldModel's two-term objective (prediction + SIGReg),
> add temporal straightening (curvature), and make **fully end-to-end** JEPA training
> (no frozen DINOv2) support fast **gradient-based** planning.

---

## 0. How to teach from this guide (for the tutoring LLM)

1. Teach in the order of §13 (the lesson plan). Each lesson cites the section to read.
2. Never present a loss term without stating (a) what degree of freedom it pins,
   (b) what failure it prevents, (c) what it is blind to. This triad is the spine of
   the whole project.
3. When a learner asks "why does X exist?", answer with the *failure that forced it*.
   The project's history (§7) is a chain of failures → diagnostics → fixes; teach it as
   that chain, not as a feature list.
4. Prefer the analogies in §11 when a concept is abstract (moving meter, map/roads,
   note-taker). Then always return to the exact code anchor.
5. Every number quoted in §9 is a measured result; teach results *with* their seed
   counts and error bars, and with the structural reading (OL→MPC gap), not just the
   headline value.

---

## 1. Project in one paragraph

A latent world model encodes images into a latent space (encoder), predicts future
latents from past latents and actions (predictor), and plans by optimizing action
sequences so that rolled-out latents reach the goal latent (planner). The paper
*Temporal Straightening for Latent Planning* (arXiv 2603.12231) showed that adding a
curvature regularizer — penalizing turns in latent trajectories — makes Euclidean
latent distance approximate geodesic distance and makes **gradient-based** planning
converge; but it kept the DINOv2 backbone frozen. LeWorldModel (LeWM) showed that
SIGReg — a sketched isotropic-Gaussian regularizer — lets a JEPA train **without**
stop-gradient or frozen encoders by preventing collapse. This project **combines** the
two into one end-to-end objective, discovers that the combination is necessary but
**insufficient** (the trainable encoder silently deletes task content — the pusher —
from the visual latent), and repairs it with (i) a **proprio grounding** term that
anchors content, and (ii) **counterfactual** terms (cf_curv, act_sens) that regularize
the off-data predictor geometry the planner actually uses.

---

## 2. Architecture (what exists, where)

Model class: `models/visual_world_model.py` (`VWorldModel`). Components:

| Component | Role | Code |
|---|---|---|
| Visual encoder | DINOv2 ViT backbone (optionally trainable) + lightweight CNN `ChannelProjector` + optional aggregation head (`agg`) | `models/dino.py` |
| Proprio encoder | MLP encoding ground-truth proprio (agent/block pos, vel) into channels appended to every patch token | `models/proprio.py` |
| Action encoder | Maps raw action to embedding tiled into the latent | `models/visual_world_model.py::encode_act` |
| Predictor | ViT with temporal causal attention; predicts next latent from K past latents+actions | `models/vit.py` |
| Decoder (optional) | VQ/transposed-conv decoder for visualization only; never shapes the planner | `models/decoder/transposed_conv.py`, `models/vqvae.py` |

**The concatenated latent `z`.** Each patch token carries, concatenated along the
channel dim (`concat_dim=1`): `[visual | proprio | action]`. The prediction loss is
taken on `[visual | proprio]` (action excluded), and the planning objective reads the
visual and proprio channels separately (`planning/objectives.py`). This concatenation
is the root of the shedding loophole (§7): because the loss is on the *concatenated*
tensor, the visual tokens are never *required* to encode what the proprio channels
already provide (`models/visual_world_model.py` L194-216 comment).

---

## 3. Data flow: training vs planning

### 3.1 Training (teacher-forced; trajectories come from the dataset)

Each sample is a window of **real consecutive observations** (PushT: `frameskip=5`,
`num_hist=3`, `num_pred=1` → 4 frames; `conf/train.yaml` L161-175). One `encode()` call
maps the whole window at once:

```
z = encode(obs, act)          # (b, 4, patches, dim)  ← the latent trajectory
z_src = z[:, :3]              # history → predictor input
z_tgt = z[:, 1:]              # future  → prediction target
z_pred = predict(z_src)       # one-step-ahead predictions
MSE(z_pred, sg?(z_tgt))       # L_pred
```

(`forward()` L652-706). **No autoregression during training**: old predictions are
discarded; both MSE sides live in the *current* encoder's coordinate system. The
curvature loss is computed on the *encoded observed window*, predictor not involved
(L732-736). This is the answer to "where does the trajectory for cosine similarity
come from?": **from the dataset**, via one fresh encode per step.

### 3.2 Planning (autoregressive; trajectories come from the predictor)

`rollout()` (L793-818): encode the initial frames once, then chain the predictor's own
outputs, re-inserting candidate action embeddings each step (`replace_actions_from_z`,
L782-790). The planner optimizes the action tensor so the rolled-out latent reaches the
goal latent. At eval time the encoder is fixed, so there is no cross-chart mismatch;
the risk here is **compounding predictor error and off-data geometry** (§7.5).

---

## 4. The training objective, term by term

The unified objective (implemented verbatim, comment at L721):

    L = L_pred + λ_SIG · SIGReg(Z) + λ_curv · L_curv        (+ our extra terms below)

### 4.1 L_pred — prediction (self-consistency)

`L_pred = ‖_{t+1} − sg?(z_{t+1})‖²` (L687-706; `stop_grad` flag chooses `sg`).

- **What it trains:** predictor (hard), encoder input side (and target side too when
  `stop_grad=False`, our e2e setting following LeWM).
- **What it pins:** nothing external. It is a *self-consistency* requirement: "the
  predictor's forecast, in today's coordinates, matches today's encoding of reality."
- **The trap:** self-consistency admits degenerate fixed points — a constant encoder
  (collapse) and a content-shedding encoder both score well. Every real requirement
  must be imported by other terms.
- **The moving-meter view:** the encoder is a trainable coordinate chart over a *fixed*
  external manifold (the pixels/actions). The MSE is chart-relative; at each step it is
  internally consistent, and across steps the chart drifts slowly (LR 1e-5). The loss
  curve is therefore measured with a slowly moving ruler — trust external probes (§8),
  not the raw loss.

### 4.2 SIGReg — distributional anti-collapse (models/sigreg.py)

Sketched Isotropic Gaussian Regularization (LeJEPA/LeWM). Method: project latents onto
M=1024 fresh random unit directions each step; on each 1-D projection compute the
Epps–Pulley statistic — the squared distance between the empirical characteristic
function and the standard normal's, `φ(t)=exp(−t²/2)`, integrated with a Gaussian window
by trapezoid rule over `knots=17` nodes in `[0, 3]`; average over projections and time
(`SIGReg.forward`, L88-130). By Cramér–Wold, matching all 1-D marginals matches the
joint. Input layout `(T, B, D)` via `to_time_major` (L133-145); patch tokens fold into
the batch axis. Always upcast to float32 under bf16 training (L100-109).

- **Pins:** the *distribution* — hence the **scale**. A shrinking encoder is no longer
  a free way to minimize L_pred; a constant encoder (all mass at one point) is
  maximally penalized because the statistic is batch-wise per timestep.
- **Prevents:** collapse. Falsified necessity in `experiments/verify_stop_grad.py`
  gates: unfrozen + stop_grad, no SIGReg → collapse (loss fell 13× while probe R² went
  0.85 → 0.00); unfrozen + SIGReg → no collapse.
- **Blind to:** *content*. A Gaussian distribution can carry or lack the agent; both
  satisfy SIGReg. Also blind to direction (that is curvature's job). The code comment
  states the division: "SIGReg pins the distribution (and hence the scale); the
  curvature term pins the direction" (L721-724).
- **Hyperparameters:** `sigreg_coeff=0.1` (λ_SIG), `num_proj=1024`, `knots=17`,
  `sigreg_apply_to=agg` (pooled trajectory representation) — `conf/train.yaml` L59-69.

### 4.3 L_curv — temporal straightening (direction)

Velocities on the encoded observed window: `v_t = z_{t+1} − z_t`;
`L_curv = 1 − cos(v_t, v_{t+1})`, averaged, with a `step_thresh` NaN guard for the
collapse edge case (`_cos_curvature` L469-487; `total_curvature` L494+ requires ≥3
frames). Modes: `cos` (per-patch/flat/avg-pool variants) and `aggcos` (aggregate tokens
with the learned `agg` head first); `curv_on ∈ {features, velocity}` chooses whether
aggregation happens before or after differencing (conf comment L50-58). Applied to
visual latents only (L732-736), matching the paper ("straightening loss is only applied
on visual latents").

- **Pins:** trajectory **direction** (turning angles). Scale-invariant — so it gives
  *no* collapse barrier (verify_stop_grad T5: shrinking latents leaves curvature
  unchanged; at exact collapse it NaNs).
- **Why it matters for planning (paper's claim, §4 of arXiv 2603.12231):** straight
  trajectories ⇒ Euclidean distance ≈ geodesic distance (distance heatmaps match
  A*-steps, paper Fig. 6) ⇒ the planning objective over actions is better conditioned
  (quadratic with H ≻ 0 in the linear case; smoother action-space landscapes, Fig. 4)
  ⇒ gradient descent converges where it previously got lost.
- **Blind to:** content and scale. And — the subtle finding of
  `experiments/curvature_incentive.py` — straightening can *reward* forgetting the
  pusher, because the pusher is the fastest, most direction-changing object: removing
  it makes trajectories straighter. A symptom promoted to a cause; it indicts the
  objective, not the optimizer.
- **Hyperparameter:** `straighten=aggcos1e-1` (λ_curv = 0.1).

### 4.4 Proprio grounding — the content anchor (our contribution)

`ground_proprio=1.0` adds `L_ground = ‖W·visual_tokens − proprio‖²` with a **linear**
read-out head on the visual tokens only, target = the model's own proprio *input*
(an existing ground-truth signal, not a new label) (`proprio_grounding_loss`,
L217-260 init + L741-747 use).

- **Why linear on purpose:** the planning cost is Euclidean on those tokens; content
  that is only non-linearly decodable would not make the cost sensitive to the pusher.
- **Why it exists:** the shedding failure (§7.2). It closes the loophole "visual tokens
  may offload the agent onto the proprio channels."
- **Result:** agent probe R² recovered to 0.991/0.993; open-loop 13.33 → 30.00; the
  OL→MPC gap collapsed +31 → +10 (baseline +8) — the structural signature of repair.
- `ground_proprio_dims`: which state dims to ground; `[0,1]` (agent position) vs all
  four is an ablation axis (conf L92-100: position-only hurts straightness, cos 0.327
  vs 0.589 — grounding unpredictable high-frequency content raises curvature).

### 4.5 Counterfactual terms — off-data geometry (our contribution)

The planner rolls out **action sequences never seen in training**; nothing in
§4.1–4.4 guarantees those rollouts are straight or even action-sensitive. Two terms
(`_counterfactual_terms`, L543-604; applied in `train.py` L1005-1023):

- **cf_curv (0.1):** curvature penalty on predictor-chained rollouts under
  counterfactual actions from a *fixed, detached* initial latent (`cf_H=4`,
  `cf_mode=cos`). Straightens the off-data rollout map — the space the planner lives in.
- **act_sens (0.1):** hinge enforcing a minimum latent displacement between two
  counterfactual action arms, relative to batch spread (`act_sens_margin=0.1`).
  Prevents the planner's cost from becoming action-insensitive (the pusher-blind cost's
  disease, generalized).
- **Gradients reach only predictor + action encoder** (initial latent encoded under
  no_grad) — exactly the first-order rollout map the planner differentiates.
- **Engineering:** applied as a **separate forward/backward after the main optimizer
  step**, after `del z_out, ...` frees the main graph — otherwise the arms' backward
  recompute stacks on the live graph and OOMs (train.py L1005-1018; comment at
  visual_world_model L749-753). Memory history: activation checkpointing alone was
  insufficient; half-batch arms (`cf_batch_frac=0.5`) + peak separation completed the
  fix.

---

## 5. Optimization structure (train.py)

- Separate optimizers: encoder (backbone LR `null` ≡ same group logic at `encoder_lr=1e-5`),
  predictor, action encoder, decoder. `freeze_backbone=False` for e2e.
- Main step: L_pred + SIGReg + curvature + grounding in one graph.
- Second step: counterfactual loss, stepping **only** predictor + action encoder
  (exact, since its gradients touch only those).
- Iteration budget pinned paper-exact: PushT 61,929 iters/epoch × 2 = 123,858
  (`iteration_budget.py`, conf L40-44) so variants compare at identical update budgets.
- `verify_encoder_trains.py` proves the trunk actually moves (weight delta > 0) —
  `requires_grad=True` alone is not proof (optimizer-group ordering bugs are real).

---

## 6. Planning stack (planning/)

- **BasePlanner** (`base_planner.py`): shared encode/eval plumbing.
- **GDPlanner** (`gd.py`): actions are `requires_grad` parameters; loop of
  `rollout → objective → backward → adam step → cosine-annealed LR → small Gaussian
  action_noise`; optional early stop when all evals succeed (L98-148). This is the
  planner straightening is *for*.
- **CEMPlanner** (`cem.py`): sampling-based cross-entropy baseline (mu/sigma updates
  over top-k). The speed comparison: GD 76.2 s vs CEM ~660 s at equal success → **8.7×**.
- **MPC** (`mpc.py`): replan every `n_taken_actions` env steps, re-encoding a real
  observation each cycle — the "crutch" that masks representation defects (§7.3).
- **Objectives** (`objectives.py`): `loss_visual + alpha·loss_proprio` on last frame
  (`last`), all frames geometrically weighted (`all`), or `staged`. The `normalize`
  flag divides each channel by its own eval-batch spread: without it, on the e2e run
  `alpha=1` behaved like `alpha_eff=0.0042` (scales 0.001089 vs 0.2598) — the proprio
  term numerically inert, the cost 99.58% weighted on the channel that forgot the
  pusher (docstring L28-48).
- **Evaluator** (`evaluator.py`): executes candidate actions in the real env; PushT
  success = `pos_diff < 20` (over agent_x, agent_y, block_x, block_y) **and**
  `angle_diff < π/9` (`env/pusht/pusht_wrapper.py:57`) — the agent is *half* the
  position score, which is why a pusher-blind cost loses half the task.

---

## 7. Failure history → diagnostics → fixes (the narrative spine)

### 7.1 Failure 0 — collapse without SIGReg
Unfrozen encoder + stop_grad alone collapses (verify_stop_grad T3: loss ↓13×, probe
0.85→0.00). Fix: SIGReg (§4.2). Lesson: stop-grad is not an anti-collapse mechanism for
trainable trunks.

### 7.2 Failure 1 — the pusher-blind latent (shedding)
With SIGReg, no collapse — but the visual latent deleted the agent: probe R² 0.943 →
−0.011 (agent_x/y) while the block *improved* 0.945 → 0.979 (`metric_alignment.py`;
code comment L194-216). Selective, not general: a content operation. Cause: the
concatenated-loss redundancy loophole; dropping the agent lowers L_pred; SIGReg
(distribution) and curvature (direction) neither forbid it — and curvature may even
reward it (`curvature_incentive.py`). Invisible in training metrics; visible only in
content probes. Consequence: open-loop 13.33%, GD and CEM both converge to the same bad
optimum (cost's optimum in the wrong place; ground-truth actions score 1.0). Fix:
proprio grounding (§4.4).

### 7.3 Structural read — the OL→MPC gap
Ungrounded: OL 13.33 vs MPC 44.67 (+31). MPC re-observes every frameskip, so it never
needs the agent to be *planned*; open-loop does. Paper baseline gap is +8. After
grounding: +10. The gap is the representation-defect signature, independent of absolute
success (`PROGRESS_SIGREG_E2E.md` L805-829).

### 7.4 Failure 2 — probes good, planning still bad (off-data geometry)
Grounded model matches pristine DINOv2 on every static probe yet plans ~20-30% vs
baseline 77%. Static decodability ≠ planning utility. The governing quantities are
*action-sensitivity* metrics: visual `snr` (rollout separation over noise) and `beat`
(fraction of alternative action sequences scoring better than the true one) — ordering
7.33 < 13.33 < 20.67 tracks snr 1.09 < 2.91 < 3.21 and beat 0.140 > 0.065 > 0.019
monotonically across three models (`PROGRESS_SIGREG_E2E.md` L510-558). Fix direction:
counterfactual terms (§4.5).

### 7.5 Rollout drift decomposition (`rollout_drift.py`)
Per horizon k: `rollout_mse` (autoregressive, as planning uses), `teacher_mse`
(one-step from real window; all training optimizes), `spread` (distance between
genuinely different states), `drift = rollout/spread`, `compound = rollout/teacher`.
Small drift at H=5 (0.135 → 0.110 final) rules out "rollout is garbage"; the defect is
geometric/semantic, not accuracy.

---

## 8. Diagnostic toolkit (models/diagnostics.py + experiments/)

- **probe_r2** (L96-140): *held-out* ridge linear probe, interleaved split. Held-out
  because in-sample R² on a 128-dim latent from a 128-row batch is exactly 1.0 while
  representing nothing. Linear because the planning cost is linear-Euclidean. ~0 or
  negative = "no usable linear information" = collapse/shedding signature.
- **latent_std** (L65-77): does z still vary across inputs? The collapse indicator.
- **effective_rank** (L47-61): participation ratio `(Σλ)²/Σλ²` ∈ [1,d]; 1 = all variance
  in one direction.
- **curvature_cos** (L81-92): mean cos(v_t, v_{t+1}); 1.0 = perfectly straight.
- **snr / beat / drift** (`rollout_drift.py`): action-sensitivity and usability metrics
  (§7.4-7.5).
- **planning_landscape.py**: 2-D slices of J(a) through the GD solution (toward
  ground-truth actions vs random orthogonal); round = well-conditioned, needle/ridge =
  geometry fighting the optimizer.
- **planning_jacobian.py**: condition-number proxies of the planning Jacobian/Hessian
  before/after straightening (plan §3).
- **violation_of_expectation.py**: a physics-aware model must rank
  err(true) ≪ err(static), err(reverse), err(foreign); ratios ≈ 1 mean texture
  memorization, not physics.
- **Core doctrine** (diagnostics docstring L3-7): *a falling prediction loss is NOT
  evidence training is working.* Minimum telemetry for e2e runs: std, eff_rank, probe
  R², curvature, snr/beat.

---

## 9. Results ledger (PushT, GD, α=1, seeds 100/200/300, 50 samples, goal_H=25)

| Variant | Open-loop | MPC | OL→MPC gap |
|---|---|---|---|
| e2e + SIGReg + curv (ungrounded) | 13.33 ± 1.15 | 44.67 ± 10.26 | +31 |
| e2e + SIGReg + curv + grounding | 30.00 ± 7.21 | 40.00 ± 10.39 | +10 |
| paper frozen baseline | 77.33 ± 6.18 | 85.33 ± 4.99 | +8 |

Readings: grounding's entire gain is open-loop (where pusher-blindness bites); MPC is
statistically unchanged (−0.55σ) because its crutch is no longer load-bearing; parity
with the frozen baseline was never available on PushT/H=5 (pristine DINOv2 probes ~0.94
on every scored dimension — frozen features are saturated). Wall-clock: GD 76.2 s,
GD-MPC 1545.8 s, CEM ~660 s → GD 8.7× faster than CEM at equal success. Final grounded
model planning signal: snr 3.49 (k=5), beat 0.029, drift 0.110; long-horizon k=10 snr
2.14, k=15 snr 1.85 — straightness survives beyond the protocol horizon.

---

## 10. Config & reproduction

- Canonical launch (pod): `python train.py --config-name train.yaml env=pusht
  encoder=dino_channel training.straighten=aggcos1e-1 training.sigreg=True
  training.sigreg_coeff=0.1 training.sigreg_apply_to=agg training.stop_grad=False
  training.freeze_backbone=False training.encoder_lr=1e-5 training.backbone_lr=null
  training.ground_proprio=1.0 training.cf_curv=0.1 training.cf_H=4 training.cf_mode=cos
  training.act_sens=0.1 training.act_sens_margin=0.1 training.epochs=3
  training.max_iterations=123858 env.num_workers=4`
- Evaluation: `reproduce_table1.py` — pure-Python driver running the paper's exact GD
  protocol per run (open-loop `plan_gd`, MPC `plan_gd_mpc`), 3 seeds, no result mixing
  (guards against run-name prefix collisions that once pooled two models into one row;
  `tests/test_rebuild_master_sidecars.py`, `tests/test_run_naming.py`).
- Tests guard the fragile invariants: `test_sigreg` (statistic scale/numerics),
  `test_proprio_grounding`, `test_tailored_losses` (cf terms), `test_objective_normalize`,
  `test_curvature_and_diagnostics`, `test_iteration_budget`, `test_eval_protocol`.

---

## 11. Concept glossary & analogies

- **JEPA / latent world model**: predict future *representations*, not pixels; plan in
  latent space.
- **Temporal straightening**: penalize turns (`1−cos` of consecutive latent velocities)
  so latent paths are straight ⇒ Euclidean ≈ geodesic ⇒ GD planning converges.
- **SIGReg**: batch-wise isotropic-Gaussian constraint via sketched Epps–Pulley; the
  anti-collapse scale pin.
- **Collapse**: encoder → constant; L_pred = 0 trivially. **Shedding**: encoder deletes
  one content (agent) while staying healthy everywhere else; the silent failure.
- **Content-blindness**: a constraint on distribution or direction cannot see *which*
  information is carried. SIGReg and L_curv are both content-blind.
- **Three degrees of freedom**: scale (SIGReg) / direction (curvature) / content
  (nothing, unless grounded). The unanchored DoF is spent by L_pred.
- **Moving meter / chart drift**: trainable encoder = re-definable ruler; MSE is
  chart-relative self-consistency over a fixed external manifold (pixels/actions).
  Anchors: DINOv2 init (continuity), SIGReg's fixed Gaussian, grounding's fixed
  artifact, small LR.
- **Teacher forcing vs autoregression**: training reads re-encoded real frames;
  planning chains predictions. Two regimes, two failure modes.
- **OL→MPC gap**: structural probe of representation health; MPC's re-observation is a
  crutch that masks pusher-blindness.
- **snr / beat**: does the cost respond to the *right* actions, and do wrong actions
  lose? The metrics that predict planning success when static probes don't.
- **Analogies to use**: (1) note-taker told only "predict the next play" never writes
  the referee down (shedding); (2) cartographer legislated to draw observed roads
  straight (training), planner later drawing hypothetical routes on the finished map
  (planning); (3) a meter whose markings must be statistically perfect may still
  measure nothing about the referee (SIGReg's content-blindness).

---

## 12. Mapping to the ICLR plan's five contributions

1. **Unified objective** → §4.1-4.3 (`visual_world_model.py` L721-736), 3 scalars
   (λ_SIG, λ_curv, + grounding/cf as the e2e repair kit).
2. **First fully e2e JEPA with GD planning** → §5 (`freeze_backbone=False`,
   `verify_encoder_trains.py`) + §9 table + GD-vs-CEM wall-clock.
3. **SIGReg ⊥ straightening complementarity** → §4.2/4.3 division (scale vs direction),
   verify_stop_grad gates, effective-rank/Gaussianity telemetry (straightening raised
   eff_rank 2.25 → 3.40 without breaking Gaussianity).
4. **GD faster & better than CEM on same model** → §6/§9 (8.7×).
5. **Clean ablations/analyses** → §8 (curvature, landscapes, Jacobian proxies, probing,
   VoE) + grounding-dims ablation + λ sensitivity axes.

Plus the project's *deeper* empirical claim, earned the hard way: **straightening +
SIGReg are necessary but insufficient for e2e planning without content anchoring and
off-data regularization** — the result a frozen-encoder paper could never see.

---

## 13. Lesson plan (teaching order)

1. **L1 Latent world models & JEPA** (§1-3): encoder/predictor/planner; teacher forcing
   vs rollout. Code: `forward()`, `rollout()`.
2. **L2 The prediction loss and its traps** (§4.1): self-consistency, collapse, moving
   meter. Code: L687-706; experiment: verify_stop_grad T3.
3. **L3 SIGReg** (§4.2): sketching, Epps–Pulley, why scale-pin stops collapse; gates.
4. **L4 Straightening** (§4.3): velocities, cosine, scale-invariance; why GD needs it
   (paper §4, Figs 4-6); curvature_incentive caveat.
5. **L5 The shedding failure** (§7.2 + §4.4): concatenated-latent loophole, probe R²,
   grounding fix; OL→MPC gap (§7.3).
6. **L6 Off-data geometry** (§7.4-7.5 + §4.5): snr/beat, drift; cf_curv/act_sens; the
   separate-backward engineering.
7. **L7 Planning & evaluation** (§6, §10): GD/CEM/MPC mechanics, objective normalize,
   success criterion, reproduce_table1 protocol.
8. **L8 Results & narrative** (§9, §12): what moved, what didn't, and the paper story.

Each lesson ends with: *state the triad* — what the term pins / what failure it
prevents / what it is blind to.
