# Teaching Bundle — manifest

Companion archive for `ICLR_TEACHING_GUIDE.md`. Feed the guide first; the tiers below
control how much code context the tutoring model receives.

## Tier 0 — always
- ICLR_TEACHING_GUIDE.md          the curriculum (teach in §13 order, triad per lesson)

## Tier 1 — core code (lessons L1–L7)
- models/visual_world_model.py    the full objective, grounding, counterfactual, rollout
- models/sigreg.py                SIGReg (sketched Epps–Pulley anti-collapse)
- models/diagnostics.py           probe R2 / eff_rank / curvature metrics
- planning/gd.py                  gradient planner
- planning/objectives.py          planning cost + normalize story
- planning/mpc.py, planning/cem.py, planning/evaluator.py, planning/base_planner.py
- conf/train.yaml                 hyperparameters with comments
- env/pusht/pusht_wrapper.py      success criterion (agent is half the score)

## Tier 2 — failure history (lessons L5–L8)
- experiments/verify_stop_grad.py     collapse falsification (gates 1–3)
- experiments/verify_encoder_trains.py backbone-moves proof
- experiments/metric_alignment.py     shedding evidence (probe table)
- experiments/curvature_incentive.py  straightening rewards forgetting
- experiments/rollout_drift.py        drift / snr / beat
- PROGRESS_SIGREG_E2E.md              results ledger + narrative
- train.py                            (lessons cite ~L990–1030: cf separate backward)

## Tier 3 — reference (per lesson)
- _paper.txt                        original paper text (objective: L230–330; arch: L420–480)
- reproduce_table1.py               eval protocol (GD open-loop + GD-MPC, 3 seeds)
- experiments/planning_landscape.py, experiments/planning_jacobian.py,
  experiments/violation_of_expectation.py   analyses (L8)
- models/dino.py, models/vit.py, models/proprio.py   architecture details (L1)

Do NOT include research_papers/ (original paper code / LeWM): it is the reference the
project deviates from; including it invites teaching the wrong variant.
