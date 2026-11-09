#!/usr/bin/env bash
# One-time setup on a fresh Linux GPU instance. Run from the repo root:
#   bash scripts/setup.sh
set -euo pipefail

# System libs: tmux for detached runs; EGL/GL for headless MuJoCo rendering; ffmpeg for video.
if command -v apt-get >/dev/null 2>&1; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update -y
  $SUDO apt-get install -y tmux git ffmpeg libegl1 libgl1 libglib2.0-0 libosmesa6
fi

# uv
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync

echo "--- checks ---"
uv run python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
uv run python -c "import lerobot, mujoco; print('lerobot', lerobot.__version__, '| mujoco', mujoco.__version__)"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "NOTE: HF_TOKEN is not set. Downloads work but are rate-limited: export HF_TOKEN=..."
fi
echo "Setup done. Next: bash scripts/gpu_run.sh"
