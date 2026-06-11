#!/usr/bin/env bash
# WorldMap 3D バックエンドを起動する。初回は ./setup.sh で venv を作成しておくこと。
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
PORT="${1:-8000}"
echo "WorldMap 3D backend → http://localhost:${PORT}  (docs: /docs)"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --reload
