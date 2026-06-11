#!/usr/bin/env bash
# CUDA PC (WSL2 / Linux) 用セットアップ。Depth Anything 3 を CUDA で動かす。
# 使い方:  CUDA_CHANNEL=cu124 ./setup-cuda.sh    （PC の CUDA に合わせて変更）
set -euo pipefail
cd "$(dirname "$0")"

CUDA_CHANNEL="${CUDA_CHANNEL:-cu124}"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CUDA 版 torch / torchvision
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"

# 共通依存 + DA3(CUDA) 依存
pip install -r requirements.txt
pip install -r requirements-cuda.txt

python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
PY

echo "完了。 DEPTH_BACKEND=auto なら CUDA 検出時に DA3 が選択されます。"
echo "起動: ./run.sh   （http://localhost:8000, /api/health で backend を確認）"
