#!/usr/bin/env bash
# docs/ を Markdown のままブラウザで閲覧する（docsify, ビルド不要・拡張子そのまま）。
# 使い方:
#   ./docs/serve.sh           # http://localhost:8088
#   ./docs/serve.sh 9000      # ポート指定
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8088}"
echo "WorldMap 3D Docs → http://localhost:${PORT}  (Ctrl+C で終了)"
exec python3 -m http.server "${PORT}"
