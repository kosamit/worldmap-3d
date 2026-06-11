#!/usr/bin/env bash
# フェーズ1 のテストページを静的サーバーで配信する。
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8765}"
echo "WorldMap 3D walk test → http://localhost:${PORT}/"
exec python3 -m http.server "${PORT}"
