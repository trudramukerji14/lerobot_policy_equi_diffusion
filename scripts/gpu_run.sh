#!/usr/bin/env bash
# Launch training inside a detached tmux session so it survives SSH disconnects.
#
#   bash scripts/gpu_run.sh                      # baseline run with defaults below
#   STEPS=20 BATCH_SIZE=8 JOB=gpu_smoke bash scripts/gpu_run.sh   # cheap 20-step GPU test
#   AUTO_SHUTDOWN=1 bash scripts/gpu_run.sh      # power off the machine when training ends
#
# Watch:   tmux attach -t train      (detach with Ctrl-b then d)
# Log:     tail -f outputs/<JOB>.log
set -euo pipefail

JOB="${JOB:-baseline}"
STEPS="${STEPS:-27500}"          # paper: 50000/n epochs = 1000 epochs for 50 demos; 3521 frames / 128 = 27.5 steps/epoch
EVAL_FREQ="${EVAL_FREQ:-550}"    # paper: 50 evals over training (every 20 epochs)
EVAL_EPISODES="${EVAL_EPISODES:-10}"  # not stated in the paper; trade-off vs. eval time
LOG_FREQ="${LOG_FREQ:-50}"       # loss line every N steps (use 10 for short tests)
# Native 480x480 images OOM a 23GB GPU at batch 128. Resize + random crop (ratio 0.9, like the
# robomimic 84 -> 76 crop). Must be identical for baseline and EquiDiff. RESIZE="" disables.
RESIZE="${RESIZE:-128 128}"
CROP_RATIO="${CROP_RATIO:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-128}"  # paper: 128
SAVE_FREQ="${SAVE_FREQ:-2000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
POLICY="${POLICY:-diffusion}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"

mkdir -p outputs
LOG="outputs/${JOB}.log"

CMD="uv run python train.py \
  --repo-id lerobot/metaworld_mt50 --episodes 2000 2049 \
  --drop-features observation.environment_state \
  --policy ${POLICY} --horizon 16 --n-action-steps 8 \
  --device cuda --steps ${STEPS} --batch-size ${BATCH_SIZE} \
  --save-freq ${SAVE_FREQ} --num-workers ${NUM_WORKERS} --log-freq ${LOG_FREQ} \
  ${RESIZE:+--resize-shape ${RESIZE} --crop-ratio ${CROP_RATIO}} \
  --env-task metaworld-push-v3 --env-eval-freq ${EVAL_FREQ} --eval-episodes ${EVAL_EPISODES} \
  --job-name ${JOB} --overwrite 2>&1 | tee ${LOG}"

if [ "${AUTO_SHUTDOWN}" = "1" ]; then
  # Shut down whether training succeeded or failed, so a crash doesn't keep billing.
  CMD="${CMD}; sudo shutdown -h now || shutdown -h now"
fi

tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train "${CMD}"

echo "Started tmux session 'train' (job=${JOB}, steps=${STEPS}, batch=${BATCH_SIZE})."
echo "Attach: tmux attach -t train    Log: tail -f ${LOG}"
echo "Checkpoints: outputs/${JOB}_${POLICY}_lerobot_metaworld_mt50/checkpoints/"
[ "${AUTO_SHUTDOWN}" = "1" ] && echo "AUTO_SHUTDOWN=1: the machine will power off when training ends."
