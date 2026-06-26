# 引き継ぎノート (HANDOFF)

別マシン（CUDA PC）で続きを実装するための引き継ぎ。最終更新: 2026-06-26。

## 1. 現在地

- ブランチ: `feat/map-route-3d-nextjs`（push 済み）
- 最新コミット: `38102c6 feat: 左パネルに「3D化」ボタンを追加`
- 作業ツリー: クリーン（`.playwright-mcp/` と検証用 `*.png` は未追跡・無視でOK）

### 起動方法
- バックエンド: `cd backend && ./run.sh`（FastAPI, :8000）。初回は `./setup.sh`（CUDAは `./setup-cuda.sh` と `requirements-cuda.txt`）。
- フロント: `cd frontend && npm install && npm run dev`（Next.js, dev は **webpack** エンジン固定。`:3000`空きなら3000、塞がってれば3001等）。
- フロントの「バックエンドURL」入力で接続先を変更可能（既定 `http://localhost:8000`）。

## 2. いま動くもの（写真パノラマ方式に刷新済み）

フロントは Next.js(App Router/TS) + React Three Fiber に全面刷新済み。深度メッシュ方式は「3D化」ボタンとして残存。

- **地図(左パネル)**: Google Map（`@vis.gl/react-google-maps`）。Places検索、1クリックで降り立つ、現在地ピンは**向き矢印**（見回しに追従）。
- **歩行(右)**: Street View 写真を全天球 equirectangular に再投影して球体表示。マウスドラッグで見回し、**コンパス**表示、W=見ている方向の隣へ/S=戻る/画面の矢印で隣接ノードへ。隣接は `StreetViewService` の link をオンデマンドで辿る。
- **「3D化」ボタン**: 現在地を深度メッシュ化（`/api/reconstruct/panorama`）→ `SceneViewer`（WASD歩行）に切替。「パノラマに戻る」で復帰。

### 主なファイル
- backend: `app/main.py`(API), `app/equirect.py`(全天球再投影・並列取得), `app/streetview.py`(Static API, pano_id対応, cube faces), `app/depth.py`(DAv2/DA3振り分け), `app/depth_da3.py`(DA3, CUDA専用), `app/reconstruct.py`(視差→メッシュ), `app/panorama.py`(多視点合成), `app/tour.py`/`app/route.py`(旧方式・現在フロント未使用), `app/storage.py`, `app/geo.py`
- frontend: `app/page.tsx`(状態機械/UI), `components/MapPicker.tsx`(地図/検索/方向ピン), `components/TourViewer.tsx`(全天球球体/コンパス/矢印), `components/SceneViewer.tsx`(深度メッシュWASD歩行), `lib/api.ts`, `lib/geo.ts`

### 主なエンドポイント
- `GET /api/health` … `{device, backend}`（今のMacは `mps`/`dav2`）
- `GET /api/config` … `{maps_api_key, has_maps_key, max_route_points}`
- `POST /api/streetview/cube` … pano_id→equirect生成(pano_idでキャッシュ, `/panos/<id>/equirect.jpg`)
- `POST /api/reconstruct/panorama` … lat/lng→深度メッシュglb（「3D化」が使用）
- 旧: `/api/reconstruct`, `/api/reconstruct/streetview`, `/api/reconstruct/route`, `/api/route/tour`（現フロント未使用）

## 3. 環境の重要点（DAv2 vs DA3）

- `app/depth.py` の `active_backend()`: `DEPTH_BACKEND=auto|dav2|da3`。auto は「CUDAあり＋DA3導入済み」なら `da3`、無ければ `dav2`。
- **DA3 は CUDA 専用**（`app/depth_da3.py`、`from depth_anything_3.api import DepthAnything3`、`.to("cuda")`）。Mac(MPS)では import 不可。
- いまローカルは `backend: dav2` / `device: mps`（= Depth Anything V2 Small）。**「3D化がうまくいかない」一因は DAv2-Small + mesh解像度256という軽量設定**。
- **GPU化＝即高精度ではない**（同一モデルなら精度ほぼ同じ＝速度向上）。ただしGPUで DA3 や大型モデル・高解像度が現実的になり、結果的に高精度になる。
- CUDA PC でやること: `requirements-cuda.txt`導入、`DEPTH_BACKEND=da3`、`DA3_MODEL`（例 `depth-anything/DA3METRIC-LARGE`）。`/api/health` が `"backend":"da3"` になれば有効。

## 4. 次タスク（未着手）: Depth/再構成パラメータを可変化

ユーザー要望:「depth anything3 のパラメータをいじれるようにしたい。今はうまくいってない」。
対象環境=**CUDA PC(DA3)**。可変にする項目（ユーザー選択済み）:
1. **深度モデル**（DAv2: Small/Base/Large、DA3: `DA3_MODEL` の metric/relative・サイズ）
2. **深度レンジ near/far**（`reconstruct.NEAR_M`/`FAR_M`）
3. **不連続しきい値**（`reconstruct.DISCONTINUITY_RATIO`）
4. **解像度/視点数**（mesh `max_width`、360°の `num_views`）

### 設計（推奨実装）
バックエンド: パラメータをリクエストで上書きできるよう各層に通す。
- `reconstruct.disparity_to_depth(disparity, disp_range, near_m=NEAR_M, far_m=FAR_M)`
- `reconstruct.reconstruct_mesh(..., near_m, far_m, discontinuity_ratio, max_width)`
- `panorama.build_panorama(images, fov_deg, max_width, near_m, far_m, discontinuity_ratio)`
- `depth.estimate_depth(image, model=None)`：DAv2 は `get_pipeline(model)` を**モデルIDごとにキャッシュ**（dict化）、DA3 は `depth_da3._load(model_id)` を**IDごとにキャッシュ**。
- `main.py` `/api/reconstruct/panorama` に Form 追加: `near`, `far`, `discontinuity`, `max_width`, `num_views`(既存), `depth_model`。各層へ伝播。
- `GET /api/config`（or `/api/health`）に `depth_backend` と **モデルプリセット一覧**を返す → UIのドロップダウン用。

フロント: 左パネルに折りたたみ「3D化 詳細設定」を追加。
- `depth_model`(select), `near`/`far`(number), `discontinuity`(number/slider), `max_width`(number), `num_views`(number)。
- `lib/api.ts` の `reconstructPanorama` 引数を拡張して上記を送る。
- バックエンド `active_backend` に応じて model プリセット（DAv2 系 / DA3 系）を出し分ける。

### 検証手順（CUDA PCで）
1. `/api/health` が `da3` を返すこと。
2. UI「詳細設定」で model=DA3METRIC-LARGE 等、near/far・discontinuity・max_width を変えて「3D化」→ 見た目とメッシュ品質を比較。
3. DAv2 でも同UIで Small→Base→Large が切り替わること（Macでも検証可、ただしMPSは遅い）。
4. パラメータ変更後にモデル再ロードが起きる点に注意（初回のみ重い）。

### 注意 / 既知の癖
- 深度メッシュ方式は原理的に写真パノラマより歪む。改善は「より良いモデル＋高解像度＋near/far/discontinuity調整」で。
- mesh の `PANORAMA_MAX_WIDTH=256`（`main.py`）が低解像度の主因。可変化＋既定引き上げを検討。
- `tour.py`/`route.py`/cube faces 等の旧コードは未使用。整理してよい。

## 5. 会話の続きの始め方（CUDA PC）
1. `git fetch && git checkout feat/map-route-3d-nextjs && git pull`
2. このファイル(`HANDOFF.md`)を Claude に読ませて「セクション4の次タスクを実装」と指示。
3. （任意）ECC の `/save-session` `/resume-session` も利用可（ただしセッション要約ファイルは別途移送が必要）。
