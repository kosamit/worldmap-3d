#!/usr/bin/env bash
# venv を作成し依存をインストールする (初回のみ)。
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "セットアップ完了。 ./run.sh で起動できます。"
