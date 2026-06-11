#!/usr/bin/env bash
# Mac から CUDA PC(WSL2) のバックエンドを localhost:8000 へトンネルする。
# 前提: 両機が Tailscale で接続済み。PC 側で ./run.sh が起動中。
#
# 使い方:
#   ./connect.sh <user@pc-tailscale-name>
#   または  WORLDMAP_PC=<user@host> ./connect.sh
#
# autossh があれば自動再接続を使う（推奨）。無ければ素の ssh。
set -euo pipefail

HOST="${1:-${WORLDMAP_PC:-}}"
if [ -z "$HOST" ]; then
  echo "使い方: ./connect.sh <user@pc-tailscale-name>   (または WORLDMAP_PC を設定)"
  exit 1
fi

echo "tunnel: Mac localhost:8000 -> ${HOST} の localhost:8000  (Ctrl+C で終了)"

if command -v autossh >/dev/null 2>&1; then
  exec autossh -M 0 -N \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L 8000:localhost:8000 "$HOST"
else
  echo "(autossh 未導入: 切断時は再実行してください。 brew install autossh で自動再接続)"
  exec ssh -N \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L 8000:localhost:8000 "$HOST"
fi
