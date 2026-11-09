#!/usr/bin/env bash
# Evaluate a trained policy in the MetaWorld sim.
# Usage: scripts/eval.sh <pretrained_model_dir> [n_episodes] [device] [task]
set -euo pipefail

POLICY_PATH="${1:?usage: eval.sh <pretrained_model_dir> [n_episodes] [device] [task]}"
N_EPISODES="${2:-50}"
DEVICE="${3:-cpu}"
TASK="${4:-metaworld-push-v3}"
OUT_DIR="${OUT_DIR:-outputs/eval_$(basename "$(dirname "$(dirname "$(dirname "$POLICY_PATH")")")")}"

MUJOCO_GL="${MUJOCO_GL:-glfw}" uv run lerobot-eval \
  --policy.path="$POLICY_PATH" \
  --policy.device="$DEVICE" \
  --env.type=metaworld \
  --env.task="$TASK" \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size=1 \
  --output_dir="$OUT_DIR"
