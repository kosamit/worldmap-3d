# リモートGPUバックエンド構成（Mac開発 / WSL2でDA3推論）

開発・フロントは Mac のまま、`http://localhost:8000` も変えずに、**重い推論(Depth Anything 3)だけ外部のCUDA PC(WSL2)** で動かす構成。

```
[Mac]  エディタ + frontend(:8765) + localhost:8000  ──SSH-L──▶  [WSL2] uvicorn(:8000) + DA3(CUDA)
              （見え方・コマンド・URLは現状維持）        Tailscale（家でも外でも同じホスト名）
```

ポイント:
- コードは **1リポジトリ両対応**。`depth.py` が CUDA を検出すると DA3、無ければ Mac の DAv2/MPS を選ぶ（`DEPTH_BACKEND=auto`）。
- 接続は **Tailscale**（NAT越え・別ネットワークでも同じMagicDNS名）。
- **SSHローカルフォワード**で Mac の `localhost:8000` をWSL2へ透過 → **フロント無変更**。

---

## 0. 前提
- CUDA PC: Windows + WSL2(Ubuntu)、NVIDIA ドライバ導入済み（`nvidia-smi` がWindowsで通る）。
- Mac: 開発機。

## 1. WSL2 を整える（PC側 / 一度だけ）
```bash
# Windows PowerShell
wsl --update
```
WSL2 内 Ubuntu で systemd を有効化（Tailscale 常駐に必要）:
```bash
sudo tee /etc/wsl.conf >/dev/null <<'EOF'
[boot]
systemd=true
EOF
```
Windows 側で `wsl --shutdown` してから WSL を開き直す。

WSL2 で CUDA が見えるか確認:
```bash
nvidia-smi   # GPU が表示されればOK（WSL2はWindowsのドライバを共有）
```

## 2. Tailscale（両機）
- Mac: Tailscale アプリを入れてログイン。
- WSL2:
  ```bash
  curl -fsSL https://tailscale.com/install.sh | sh
  sudo tailscale up --ssh        # --ssh で Tailscale SSH を有効化（鍵管理不要）
  tailscale status               # 自分のMagicDNS名（例: cuda-wsl）を確認
  ```
  同一アカウントでログインすれば、家でも外でも同じ名前（例 `cuda-wsl`）で繋がる。

## 3. バックエンドをPCに置いて起動（WSL2）
```bash
git clone <このリポジトリ> worldmap-3d
cd worldmap-3d/backend
CUDA_CHANNEL=cu124 ./setup-cuda.sh    # PC の CUDA に合わせて cu121/cu124 等
./run.sh                              # http://localhost:8000
```
別ターミナルで確認:
```bash
curl -s localhost:8000/api/health
# {"status":"ok","device":"cuda","backend":"da3"}  ← これが出れば DA3 稼働
```
Street View を使うなら `backend/.env` に `GOOGLE_MAPS_API_KEY=...` を置く（gitignore済み）。

## 4. Mac から繋ぐ（localhost:8000 を維持）
```bash
cd worldmap-3d
./connect.sh <user>@cuda-wsl     # 例: ./connect.sh kosamit@cuda-wsl
# Tailscale SSH 利用時は user は WSL2 のユーザー名
```
これで Mac の `localhost:8000` が WSL2 の uvicorn に透過。別ターミナルで:
```bash
curl -s localhost:8000/api/health   # backend":"da3" が返ればトンネル成功
cd frontend && ./serve.sh           # これまで通り http://localhost:8765/
```
フロントの「バックエンドURL」は `http://localhost:8000` のままでOK。

---

## 日々の開発（やり方は不変）
- **編集**: Mac のエディタで `backend/app/*.py` を編集（今まで通り）。
- **PCへ反映**: いずれか
  - 低頻度: PCで `git pull` → `./run.sh`（`--reload`なので保存検知で再起動）。
  - 高頻度: [mutagen](https://mutagen.io/) で Mac→PC 双方向同期（保存即反映、エディタ操作は不変）:
    ```bash
    # Mac
    brew install mutagen-io/mutagen/mutagen
    mutagen sync create --name=worldmap \
      ~/Gist/AutoClaude/worldmap-3d  <user>@cuda-wsl:~/worldmap-3d \
      --ignore=.venv --ignore=data --ignore=node_modules
    ```
- **接続**: `./connect.sh` を常駐（autossh 推奨で自動再接続）。

## 切り分け
| 症状 | 確認 |
|------|------|
| health が `backend":"dav2"` | PCで CUDA 未検出か DA3 未導入。`python -c "import torch;print(torch.cuda.is_available())"` / `pip show depth-anything-3` |
| Mac で `localhost:8000` 不通 | `./connect.sh` が生きているか、PCで `./run.sh` 起動中か |
| Tailscale で繋がらない | 両機 `tailscale status`、同一アカウントか |
| xformers 入らない | CUDA版torch先入れ→`pip install xformers`、CUDAバージョン整合を確認 |

## DA3 モデルの選択
`DA3_MODEL` 環境変数で切替（既定 `depth-anything/DA3METRIC-LARGE` = メートル絶対深度）。
VRAM が小さければ `DA3METRIC-LARGE`→`DA3-BASE` 等に。
```bash
DA3_MODEL=depth-anything/DA3-BASE ./run.sh
```
