# 3D化 検証ハーネス

フロントUIを Playwright で操作して **3D化を実行 → 生成メッシュを全方位レンダリング
→ ターンテーブルGIF と summary.json を出力**します。Claude が繰り返し回して、
生成の成否と 3D の見た目を毎回確認するための道具です。

## 準備（初回のみ）

```bash
cd verify
npm install
npx playwright install chromium
```

ffmpeg があれば GIF も生成します（無ければフレーム PNG のみ）。

## 実行

フロント(:3000) と バックエンド(:8000) を起動した状態で:

```bash
node verify3d.mjs 35.659350 139.700962            # 高精度3D化(マルチビュー)
node verify3d.mjs 35.659350 139.700962 --mode simple   # 簡易3D化
node verify3d.mjs <lat> <lng> --frames 12 --el 20 --out ./out
```

## 出力（`out/`、git管理外）

- `turntable.gif` … 生成メッシュをぐるっと回したGIF（3Dの立体構造が分かる）
- `frame_00.png … ` … 各アングルの静止画
- `summary.json` … `{ ok, status, vertexCount, viewpoints, imagesUsed, radius, gif, errors }`

終了コードは成功 0 / 失敗 1。

## 注意

- バックエンドが `--reload` 起動だと、ソース編集で再起動し**実行中の生成ジョブが落ちます**。
  検証中はコードを編集しないか、`--reload` なしで起動してください。
- ヘッドレスはソフトウェアWebGL(swiftshader)です。**3Dシーン内をポインターロックで
  歩く操作はクラッシュ**するため、本ハーネスは「カメラを外周から回す(orbit)」方式で
  立体を可視化します（`frontend/public/orbit.html`）。実際の歩行は実GPUのブラウザで。
