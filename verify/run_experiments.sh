#!/usr/bin/env bash
# 品質比較の一括実行: 地点ごとに DA3推論を1回キャッシュ → 各バリアントを別プロセスで
# 生成（メモリ隔離） → orbit.html でGIF化 → HTMLギャラリーに集約。
# 使い方: bash run_experiments.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/backend/.venv/bin/python"
SCR="$ROOT/backend/scripts"
SCENES="$ROOT/backend/data/scenes"
TMP="${TMPDIR:-/tmp}/da3_exp"
mkdir -p "$TMP"

LOCS=("outdoor:35.6595:139.7005" "indoor:35.659350:139.700962")
VARIANTS=(baseline aggressive depth_smooth taubin cloud_clean tsdf poisson)

for loc in "${LOCS[@]}"; do
  IFS=':' read -r name lat lng <<< "$loc"
  npz="$TMP/pred_$name.npz"
  echo "=== [$name] inference (cache) ==="
  "$PY" "$SCR/cache_pred.py" "$lat" "$lng" "$npz" || { echo "cache failed for $name"; continue; }
  ids=()
  for v in "${VARIANTS[@]}"; do
    sid="exp_${name}_${v}"
    echo "--- [$name] build $v ---"
    "$PY" "$SCR/build_variant.py" "$npz" "$v" "$sid" "$name / $v" || echo "build $v FAILED"
    [ -f "$SCENES/$sid/meta.json" ] && ids+=("$sid")
  done
  # manifest.json を meta.json から組み立て
  manifest="$TMP/manifest_$name.json"
  "$PY" - "$SCENES" "$manifest" "${ids[@]}" <<'PYEOF'
import json, sys, pathlib
scenes, out = pathlib.Path(sys.argv[1]), sys.argv[2]
items = []
for sid in sys.argv[3:]:
    m = json.loads((scenes/sid/"meta.json").read_text())
    items.append(m)
pathlib.Path(out).write_text(json.dumps(items, ensure_ascii=False))
print("manifest:", out, len(items), "items")
PYEOF
  echo "=== [$name] render gallery ==="
  node "$ROOT/verify/gallery.mjs" "$manifest" "$ROOT/verify/gallery_$name"
done
echo "ALL DONE. ギャラリー: verify/gallery_*/index.html"
