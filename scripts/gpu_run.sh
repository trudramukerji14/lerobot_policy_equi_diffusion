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
STEPS="${STEPS:-30000}"          # TODO: set from the paper once the epoch count is verified
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
  --save-freq ${SAVE_FREQ} --num-workers ${NUM_WORKERS} \
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
