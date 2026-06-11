# WorldMap 3D

Google Map の 3D 版。Street View の画像から 3D データを生成し、擬似キャラクターで街の中を歩けるようにするプロジェクト。

## フェーズ

1. **フェーズ1 ✅** — ブラウザ上で WASD 歩行できる 3D テストページ
2. **フェーズ2 ✅** — Python バックエンドで 1枚画像 → Depth Anything V2 で深度推定 → 点群経由で歩けるメッシュ(glb)生成
3. **フェーズ3 ✅** — 生成 3D データを蓄積し、フロント(ビューワー)で読み込んで歩く

## ディレクトリ構成

```
worldmap-3d/
├── frontend/                # Three.js による 3D ウォークシーン
│   ├── index.html           # フェーズ1: テスト街を歩く
│   ├── viewer.html          # フェーズ3: 生成シーンを読み込み歩く + 3D化UI
│   ├── styles.css / viewer.css
│   └── src/
│       ├── main.js              # index 用エントリ
│       ├── viewer.js            # viewer 用エントリ（API連携 + 歩行）
│       ├── PlayerController.js   # WASD + マウスルック(PointerLock) + 重力/ジャンプ
│       ├── testCity.js          # テスト用の街シーン
│       ├── SceneLoader.js       # glb 読み込み(GLTFLoader)
│       └── api.js               # バックエンドAPIクライアント
└── backend/                 # Python (FastAPI + Depth Anything V2)
    ├── app/
    │   ├── main.py              # FastAPI エンドポイント
    │   ├── depth.py             # 深度推定 (transformers, MPS/CUDA/CPU 自動選択)
    │   ├── reconstruct.py       # 深度 → 点群 → グリッドメッシュ → glb
    │   ├── streetview.py        # Street View Static API 取得
    │   └── storage.py           # シーン(glb+meta.json)の蓄積・列挙
    ├── requirements.txt
    ├── setup.sh / run.sh
    └── data/scenes/<id>/        # 生成物 (gitignore)
```

## 起動方法

### バックエンド (フェーズ2/3)

```bash
cd backend
./setup.sh        # 初回のみ: venv 作成 + 依存インストール
./run.sh          # http://localhost:8000 (API docs: /docs)
```

Street View API を使う場合は `backend/.env` に key を置く（gitignore 済み）:

```
GOOGLE_MAPS_API_KEY=xxxxxxxx
```

### フロントエンド

```bash
cd frontend
./serve.sh        # http://localhost:8765/
```

- `index.html` … フェーズ1のテスト街を歩く
- `viewer.html` … バックエンドに画像/緯度経度を投げて3D化 → そのまま歩く

## API

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/api/health` | 稼働確認・推論デバイス |
| POST | `/api/reconstruct` | 画像アップロード(multipart `file`, `fov`) → 3D化 |
| POST | `/api/reconstruct/streetview` | `lat,lng,heading,pitch,fov` → Street View取得 → 3D化 |
| GET | `/api/scenes` | 蓄積シーン一覧 |
| GET | `/scenes/<id>/scene.glb` | 生成 glb (静的配信) |

## 操作方法（歩行）

| キー | 動作 |
|------|------|
| 画面クリック | 開始（ポインタロック） |
| W A S D / 矢印 | 移動 |
| マウス | 視点回転 |
| Shift | ダッシュ |
| Space | ジャンプ |
| Esc | ロック解除 |

## 技術メモ

- Three.js は CDN (importmap) から ES Modules で読み込み、ビルド工程なし。
- 深度推定は **Depth Anything V2 (Small)** を transformers 経由で実行。Apple Silicon の MPS / CUDA / CPU を自動選択。
  単一画像の相対深度から、ピンホール逆投影で点群を作り、深度グリッドを三角形化してメッシュ(glb)を生成する。
  オクルージョン境界（急な深度差）の面は分断して引き伸ばしを防ぐ。
- 1枚画像のため絶対スケールは不定。距離は [1m, 25m] に正規化した相対値。

### 今後の拡張候補

- **Depth Anything 3** (ByteDance, CUDA 環境向け): マルチビュー整合・カメラポーズ推定で、複数 Street View を1つの一貫した3D空間に合成可能。`xformers` 依存のため MPS では不安定 → CUDA マシン用意時に `depth.py` を差し替える。
- 複数視点(heading違い)の合成によるパノラマ全周の3D化。
- メッシュとの当たり判定（現状すり抜け可）。
- 隣接シーンの相対配置による「街」全体の連結。
