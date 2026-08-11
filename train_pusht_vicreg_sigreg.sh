#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# train_pusht_vicreg_sigreg.sh
#
# The "World Model + VICReg + SIGReg" objective on PushT:
#
#   L = L_pred
#     + lambda_var * VICReg-variance(Z)      relu(1 - std(z_i)) per dim
#     + lambda_cov * VICReg-covariance(Z)    off-diag(cov(z))^2 / d
#     + lambda_SIG * SIGReg(Z)               sketched isotropic Gaussianity
#
# i.e. the prediction loss plays the role of VICReg's invariance term, the
# VICReg terms pin the second-order structure explicitly, and SIGReg pins the
# full distribution (Bardes et al., VICReg, ICLR 2022, arXiv:2105.04906;
# LeWorldModel / LeJEPA, research_papers/le-wm). Temporal straightening is OFF
# by default here (this variant isolates the two distributional regularisers);
# set STRAIGHTEN=aggcos1e-1 to stack it on top.
#
# Anchors for every hyperparameter:
#   lambda_var   = 25         VICReg paper (batch 2048, 2048-d). At our batch 32
#                             / small latent this can dominate L_pred -- sweep
#                             down (VCREG_STD=1) if val_probe_r2 falls.
#   lambda_cov   = 1          VICReg paper, same caveat.
#   lambda_SIG   = 0.1        LeWM sec. 3 ("M = 1024 projections, lambda = 0.1").
#   vcreg_apply_to = visual   VICReg acts on exactly the tokens SIGReg sees.
#   stop_grad    = False      LeWM: no stop-gradient heuristics.
#   freeze_backbone = False   end-to-end; VICReg+SIGReg together suppress the
#                             collapse the frozen trunk used to hide.
#   epochs/iters   = 123,858  Straightening App. A.3 PushT budget (61,929 x 2).
#
# Usage:
#   bash train_pusht_vicreg_sigreg.sh
#   VCREG_STD=1 VCREG_COV=0.04 bash train_pusht_vicreg_sigreg.sh   # scaled-down sweep
#   SIGREG=0     bash train_pusht_vicreg_sigreg.sh                 # VICReg-only ablation
#   VCREG=0      bash train_pusht_vicreg_sigreg.sh                 # SIGReg-only ablation
#   STRAIGHTEN=aggcos1e-1 bash train_pusht_vicreg_sigreg.sh        # + curvature
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

# --- Blackwell B200 / MIG env recipe ---
unset CUDA_VISIBLE_DEVICES                  # MIG UUID breaks mujoco-py's int() parse
export DATASET_DIR="${DATASET_DIR:-/workspace/arun/data}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WANDB_MODE="${WANDB_MODE:-offline}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync   # MIG NVML allocator assert
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

MAX_ITERS="${MAX_ITERS:-123858}"            # 61,929 x 2 == paper's PushT budget
VCREG="${VCREG:-1}"                         # 0 -> SIGReg-only ablation
VCREG_STD="${VCREG_STD:-25}"                # VICReg lambda_var
VCREG_COV="${VCREG_COV:-1}"                 # VICReg lambda_cov
VCREG_APPLY="${VCREG_APPLY:-visual}"        # visual == same tokens SIGReg sees
SIGREG="${SIGREG:-1}"                       # 0 -> VICReg-only ablation
SIGREG_COEFF="${SIGREG_COEFF:-0.1}"
STRAIGHTEN="${STRAIGHTEN:-False}"           # stack curvature on top if desired
FREEZE="${FREEZE:-False}"                   # False == end-to-end
ENCODER_LR="${ENCODER_LR:-1e-5}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
CKPT_BASE="${CKPT_BASE:-$PWD/checkpoints}"
LOG="train_pusht_vicreg_sigreg_$(date +%Y%m%d_%H%M%S).log"

VCREG_FLAG=$([ "${VCREG}" = "0" ] && echo "False" || echo "True")
[ "${VCREG}" = "0" ] && { VCREG_STD=0; VCREG_COV=0; }
SIGREG_FLAG=$([ "${SIGREG}" = "0" ] && echo "False" || echo "True")
[ "${SIGREG}" = "0" ] && SIGREG_COEFF=0.0

echo "DATASET_DIR   = ${DATASET_DIR}"
echo "objective     = L_pred + ${VCREG_STD}*VICReg-var + ${VCREG_COV}*VICReg-cov (${VCREG_APPLY}) + ${SIGREG_COEFF}*SIGReg | straighten=${STRAIGHTEN}"
echo "freeze_backbone = ${FREEZE}   encoder_lr=${ENCODER_LR}  backbone_lr=${BACKBONE_LR}"
echo "max_iterations  = ${MAX_ITERS}"
echo "log             = ${LOG}"
echo

# --- pre-flight: the 45 GB MIG slice holds exactly one job ---
BUSY=""
for pat in "[t]rain.py --config-name" "[p]lan.py --config-name" "[r]eproduce_table1.py"; do
  hits=$(pgrep -af "$pat" || true)
  [ -n "$hits" ] && BUSY="${BUSY}${hits}"$'\n'
done
if [ -n "${BUSY}" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "REFUSING TO START: the MIG slice already has a job." >&2
  echo "${BUSY}" >&2
  echo "Chain on its PID instead, or set FORCE=1 if you know the slice is free." >&2
  exit 1
fi

echo "== stray python processes (kill -9 anything left over) =="
ps -eo pid,etime,rss,cmd | grep -i python | grep -v grep || echo "  none"
echo "== MIG memory (want a few MiB used, not ~41 GB) =="
nvidia-smi | sed -n '/MIG dev/,/Processes/p' || true
echo

setsid nohup python train.py --config-name train.yaml \
  env=pusht \
  encoder=dino_channel \
  training.vcreg="${VCREG_FLAG}" \
  training.vcreg_std_coeff="${VCREG_STD}" \
  training.vcreg_cov_coeff="${VCREG_COV}" \
  training.vcreg_apply_to="${VCREG_APPLY}" \
  training.straighten="${STRAIGHTEN}" \
  training.sigreg="${SIGREG_FLAG}" \
  training.sigreg_coeff="${SIGREG_COEFF}" \
  training.sigreg_num_proj=1024 \
  training.sigreg_apply_to=agg \
  training.stop_grad=False \
  training.freeze_backbone="${FREEZE}" \
  training.encoder_lr="${ENCODER_LR}" \
  training.backbone_lr="${BACKBONE_LR}" \
  training.epochs=3 \
  training.max_iterations="${MAX_ITERS}" \
  env.num_workers=4 \
  ckpt_base_path="${CKPT_BASE}" \
  "$@" \
  > "${LOG}" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > .train_pid
cat <<EOF

Started PID ${PID} (saved to .train_pid). Detached: survives disconnects.

  Watch:            tail -f ${LOG}
  Objective check:  grep -E "VICReg enabled|SIGReg enabled|Combined objective|base_model is" ${LOG}
  Budget check:     grep -A6 "Iteration budget" ${LOG}

  *** THE metrics to watch -- collapse, logged once per epoch: ***
    grep -oE "val_probe_r2[^,]*|val_latent_eff_rank[^,]*|val_agg_probe_r2[^,]*" ${LOG}
  If val_probe_r2 heads to 0 the encoder has collapsed and the run is dead.
  With VICReg too strong (25/1 at batch 32) the variance hinge can also crush
  the representation: watch loss/z_vicreg_std_loss in the telemetry and scale
  VCREG_STD down if it stays pinned at ~1 while val_probe_r2 falls.

Run directory: derived by hydra from the objective (the _vic_s<var>_c<cov>
suffix), so read it back from the log, do not assume it:
  grep -m1 -oE "${CKPT_BASE}/test/[^ ]*" "${LOG}"
EOF
