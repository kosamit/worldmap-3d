# 新論文コンセプト：Street View から「歩ける意味的3D世界」を測量基準で構築する

調査日: 2026-06-28 / 目的: DA3 と S-NeRF の技術を日本語で整理し（特にカメラ推定・物体サイズ検出）、
本プロジェクト（意味コライダー層）と組み合わせた**新しい手法**として構想する。
関連: [town_3d_reconstruction_survey.md](town_3d_reconstruction_survey.md) / [semantic_scene_colliders.md](semantic_scene_colliders.md)
**関連研究の系譜は [tech_genealogy.md](tech_genealogy.md)（本論文の Related Work）にまとめた。**

---

## 0. 一行アイデア

> **既知のカメラ高さを“測量の標尺”にして絶対スケールを与え**、DA3 のレイマップでカメラ姿勢を、
> セマンティックセグメンテーションで床/壁/天井と物体を取り、**LiDARも与えられた3D箱も使わずに**
> 「歩ける・衝突する・意味の付いた」街路3D世界を、疎な Street View から作る。

S-NeRF が LiDAR と既知3D箱に依存していた部分を、**単眼＋カメラ高さの幾何**で置き換えるのが新規性。

---

## 1. Depth Anything 3 (DA3) 技術まとめ（日本語）

### 1.1 表現：depth map ＋ ray map
- 入力 N_v 枚 ℐ={𝐈_i}。各画像に **depth map 𝐃_i∈ℝ^{H×W}** と **ray map 𝐌∈ℝ^{H×W×6}**（画素ごとに光線の**原点3＋方向3**）を出力。
- **Dual-DPT Head**：共有 reassembly → 深度ブランチ／光線ブランチの2系統（共通特徴・異なる融合）。
- バックボーンは DINOv2 系＋**クロスビュー自己注意**（前段 L_s 層=画像内, 後段 L_g 層=クロスビュー, 既定 2:1）。

### 1.2 カメラ姿勢推定（★重要）
ray map から幾何的に復元：
- **カメラ中心** 𝐭_c = (1/HW)∑_{h,w} 𝐌(h,w,:3)（光線原点の平均）。
- **ホモグラフィ 𝐇=𝐊𝐑**：標準光線 𝐩 を対象光線方向へ写す 𝐇 を
  𝐇*=argmin_{||𝐇||=1} ∑||𝐇𝐩×𝐌(dir)|| で **DLT** により解き、**RQ分解**で 𝐊,𝐑 を得る。
- 代替の**軽量カメラヘッド**（カメラトークン→MLP）：FOV 𝐟∈ℝ², 回転 四元数 𝐪∈ℝ⁴, 並進 𝐭∈ℝ³ を直接予測（計算量~0.1%）。
- **ポーズ条件注入**：既知パラメータは camera token 𝐜_i=ℰ_c(𝐟,𝐪,𝐭) を全注意層へ。未知時は学習可能トークン 𝐜_l。

### 1.3 スケール／メートル復元・物体サイズ
- 学習時、全GT信号を**共通スケール因子**（再投影点群 𝐏 の平均 ℓ₂ ノルム）で正規化。
- **DA3-Metric**：正準カメラ空間 `深度 *= f_c/f`（f_c=300）で焦点距離差を吸収しメトリック化。
- → **物体の大きさ**はメトリック点群上で直接測れる（深度×画角）。

### 1.4 複数視点の整合・融合
- 点群は要素ごと演算 **𝐏=𝐭+𝐃(u,v)·𝐝**。
- **Umeyama＋RANSAC**でGT/視点間整列 → **TSDF**で密3D。

### 1.5 学習
- 合成のみで学習した**教師**→実画像の**生徒**（RANSAC least-squares で scale-shift 整列 𝐃^{T→M}=ŝ𝐃̃+t̂）。
- 損失 ℒ=ℒ_D+ℒ_M+ℒ_P+βℒ_C+αℒ_grad。

---

## 2. S-NeRF（Street Views NeRF）技術まとめ（日本語）

### 2.1 全体像
自動運転データ（疎・低重複・無限遠背景）向けの街路NeRF。**背景大規模シーン**と**前景の動く車**を同時に新視点合成。

### 2.2 カメラ姿勢推定・改善（★重要）
- SLAM/IMU の初期姿勢 P=[R,T] に**学習可能な改善オフセット ΔP=(ΔR,ΔT)** を足して誤差低減。
- 動く車は**仮想カメラ変換**で物体中心座標へ：**P̂_i = P_b P_i^{-1}**（P_b=物体位置）。物体を静止させ、カメラの相対運動だけ学習。
- **リプロジェクション信頼度**：𝐗_t=ψ(ψ⁻¹(𝐗_s,P_s),P_t) でワープし RGB/SSIM/VGG 類似度から信頼度。

### 2.3 深度教師信号
- 疎LiDARを **NLSPN（depth completion）** で稠密化（隣接フレーム累積）。
- **ジオメトリ信頼度** C_depth=γ(|d_t−d̂_t|/d_s)、**フロー信頼度** C_flow（光フロー一貫性）、閾値 τ=20% で外れ値除去。
- 学習可能加重 Ĉ=∑ω_iC_i（RGB/SSIM/VGG/深度/フロー）。

### 2.4 前景車両・物体サイズ（★重要）
- **3D検出器**で「3D bounding box＋中心」を取得 → **物体ごとのサイズ・姿勢が既知**。
- 仮想カメラ変換で物体中心NeRFを学習、インスタンスセグメンテーションで背景と分離。

### 2.5 背景の無限遠パラメータ化
- 改良シーン正規化 f(x)=x/r (||x||≤r), さもなくば (2−r/||x||)x/||x||（r=3m）。近距離詳細＋無限遠圧縮。

### 2.6 損失
- ℒ=ℒ_color+λ₁ℒ_depth+λ₂ℒ_smooth。深度は信頼度加重 ℒ_depth=∑Ĉ·|𝒟−𝒟̂|。背景 λ₁=0.2, 前景 λ₁=1。

---

## 3. カメラ推定：両者の比較と採用案

| | DA3 | S-NeRF | 新論文（採用案） |
|---|---|---|---|
| 起点 | なし（画像のみ・pose-free） | SLAM/IMU初期姿勢 | **既知の取得角(heading/pitch)** ＋必要ならDA3 ray-pose |
| 手法 | ray map→DLT→RQ分解 で K,R / 軽量MLPヘッド | ΔP 学習オフセット＋reprojection confidence | DA3 の ray-pose を初期値、**S-NeRFのΔP改善**で微調整 |
| 強み | センサ不要 | 既知軌跡で安定 | Street Viewは取得角が既知＝**強い初期値**が無料 |

→ **Street View では heading/pitch が既知**なので、S-NeRF の「初期姿勢＋ΔP改善」枠組みに**取得角を初期姿勢として注入**でき、DA3 の ray-pose を併用して堅牢化できる。

---

## 4. 物体サイズ検出：両者と新提案

| 方式 | サイズの出どころ | 依存 |
|---|---|---|
| DA3 | メトリック点群（深度×画角） | 深度モデル |
| S-NeRF | **与えられた3D bbox**（3D検出器） | データセットの3D箱・LiDAR |
| **新提案** | **接地点×画角×既知カメラ高さ** | セグメンテーションのみ（深度・LiDAR・3D箱 不要） |

**新提案の物体サイズ式（測量原理）**：
- セグメンテーションで物体インスタンスを取り、**接地点（最下画素）の視線**を**カメラ高さ H の床平面**に交差 → 物体までの距離 d。
- 角度範囲 (Δθ, Δφ) と d から **幅 ≈ Δθ·d、高さ ≈ tan(Δφ)·d**。
- → **単眼＋カメラ高さだけ**でメートル単位の箱/円柱コライダーが出る（[semantic_scene_colliders.md](semantic_scene_colliders.md) の戦略1で実装済み）。

---

## 5. 新論文の提案手法（組み合わせ）

**タイトル案**: *“Surveyed Streets: Metric, Semantic, and Walkable 3D from Monocular Street-View Panoramas without LiDAR or Given Boxes”*

### コア主張（新規性）
1. **測量基準スケール**：S-NeRF が LiDAR、DA3 が学習メトリックで得る絶対スケールを、**既知カメラ高さ（標尺）＋床平面RANSAC**で代替。センサ不要。
2. **意味コライダー層**：セグメンテーション（OneFormer/Cityscapes）→ 床/壁/天井の**平面**＋物体の**箱/円柱**を、**ゲーム即用のコライダー**として出力。再構成と相互作用を橋渡し。
3. **箱なし物体サイズ**：S-NeRF の「与えられた3D箱」を、**接地＋画角＋カメラ高さ**で推定（§4）。
4. **取得角を初期姿勢に**：Street View の既知 heading/pitch を S-NeRF 型 ΔP 改善の初期値に。DA3 ray-pose を併用。

### 全体パイプライン
```
疎な Street View パノラマ（既知 heading/pitch）
  ├─[姿勢] 取得角→初期姿勢 → (任意)DA3 ray-pose + S-NeRF型ΔP微調整
  ├─[意味] OneFormer/Mask2Former(Cityscapes) → 床/壁/天井/空/車/人/柱/植生
  ├─[スケール] 床RANSAC＋既知カメラ高さ H → メートル基準（測量）
  ├─[幾何・衝突] 床=平面, 壁=ファサード, 物体=接地点×画角×H で箱/円柱（深度・LiDAR不要）
  ├─[見た目] 背景=S-NeRF無限遠正規化 or 3DGS / 前景車=物体中心NeRF（仮想カメラ変換）
  └─[融合] 信頼度加重(S-NeRF)＋多視点整合(DA3: Umeyama/TSDF)
出力: 写真品質で歩け・衝突し・意味の付いた街路3D（コライダーJSON＋3DGS/NeRF）
```

### S-NeRF/DA3 から借りる具体部品
- **信頼度加重融合**（S-NeRF Ĉ）：複数パノの重なりを「別生成→合体」でなく**信頼度で融合**（過去の重なり問題の解）。
- **仮想カメラ変換**（S-NeRF P̂=P_bP^{-1}）：動的な車を物体中心で扱い、背景から分離。
- **ray-pose＋ΔP**：センサなしで頑健な姿勢。
- **無限遠正規化 f(x)**：空・遠景を破綻なく。

### 評価（構想）
- 幾何: 床平面性・壁直交性・物体寸法のメートル誤差（巻尺GTやGoogle実測比）。
- 衝突: 歩行可能率・貫通率。
- 見た目: PSNR/SSIM/LPIPS（新視点）。
- アブレーション: LiDAR無し vs S-NeRF、3D箱無し vs S-NeRF、カメラ高さ標尺の有無。

---

## 6. 本プロジェクトとの対応（実装状況）
- §4 の物体サイズ・床/壁コライダーは **`app/colliders.py` に実装済み**（戦略1, メートル単位）。
- カメラ高さ標尺は **`_ground_plane_cap`（panorama3d.py）** で実証済み。
- **取得角→初期姿勢の DA3-free 化＝実装済み**（2026-06-28）。`build_analytic_prediction`
  が既知 heading/pitch/fov からカメラ姿勢(K,extrinsics)を解析的に構成し、`main.py` の
  `method=="colliders"` 分岐は **DA3 推論を完全スキップ**。深度・LiDAR・3D箱を一切使わずに
  SegFormer(Cityscapes)＋床平面＋既知カメラ高さで床/壁/物体コライダーを生成する。
  → 本論文のコア主張1・3（測量基準スケール／箱なし物体サイズ）が単眼セグのみで成立。
- 次：S-NeRF型 信頼度融合での多パノ統合、3DGS化。

---

## 7. 出典
- Depth Anything 3: https://arxiv.org/abs/2511.10647 , https://depth-anything-3.github.io/
- S-NeRF: https://arxiv.org/abs/2303.00749 , https://ziyang-xie.github.io/s-nerf/ , https://ar5iv.labs.arxiv.org/html/2303.00749
- 関連（再構成/意味）: [town_3d_reconstruction_survey.md](town_3d_reconstruction_survey.md), [semantic_scene_colliders.md](semantic_scene_colliders.md)
