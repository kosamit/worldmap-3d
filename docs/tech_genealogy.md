# 技術系譜・関連研究（Related Work）

調査日: 2026-06-28 / 目的: これまでの調査（[town_3d_reconstruction_survey.md](town_3d_reconstruction_survey.md) /
[semantic_scene_colliders.md](semantic_scene_colliders.md)）で挙げた全技術の **派生の系譜** と **相互関係** を整理し、
新論文（[new_paper_concept.md](new_paper_concept.md)）の Related Work とする。

狙い: **複数地点の Street View をつなげて3D化**。**見た目は写実的（NeRF/3DGS）**、
**衝突はゲーム的でよい（プリミティブ・コライダー）** という設計選択を、系譜の中で位置づける。

---

## 0. 全体マップ：4つの系統が交差する

```
(A) 幾何・構造        (B) 見た目・描画        (C) 意味（2D→3D）     (D) レイアウト/事前分布
SfM/MVS → 学習pointmap  NeRF → 3DGS            seg → 3Dリフト         単画像/パノラマ レイアウト
   │                      │                      │                      │
   └──────────┬───────────┴──────────┬───────────┴───────────┬──────────┘
              ▼                       ▼                       ▼
        姿勢・点群               写実レンダリング         意味・物体・床壁
              └───────────── 本提案：交差点 ─────────────┘
        「写実な見た目 ＋ ゲーム的コライダー ＋ 意味 ＋ 測量スケール」
```

---

## 1. (A) 幾何・構造：SfM から「学習pointmap基盤モデル」へ

```
COLMAP (2016) ── 古典SfM+MVS：特徴点+RANSAC+バンドル調整。遅い・広ベースライン弱い
   │  （特徴点パイプラインを Transformer が end-to-end 置換）
   ▼
DUSt3R (2024) ── pose-free・pointmap（画素→3D点を共通座標で）。2枚から密復元。基盤モデルの起点
   ├─► MASt3R ── マッチング統合。LoFTR/SuperGlue 超え → MASt3R-SfM
   ├─► VGGT (2025) ── 多視点を1回のfeed-forwardで姿勢＋点群（反復最適化を排除）。COLMAP比+50%
   │       ├─► Depth Anything 3 (2025) ── ★現SOTA。depth+ray map、ray-poseでK,R、メトリック。VGGT超え
   │       └─► π³ (Pi3, ICLR2026) ── 基準ビュー無しの置換不変。入力順に頑健 → Pi3X(メトリック/条件注入)
   │               └─► KV-Tracker ── π³のKVをキャッシュしリアルタイム6-DoF追跡（モデル非依存）
   └─► MapAnything / MV-DUSt3R+ ── メトリック対応・数秒で疎視点復元
```
**関係の本質**: 「特徴点マッチ」→「pointmapを直接回帰」へ。DA3/Pi3 は **COLMAPの正統な後継** で、疎・広ベースラインの Street View に強い。本プロジェクトの DA3 はこの系統。

## 2. (B) 見た目・描画：NeRF から 3DGS、そして街路特化へ

```
NeRF (2020) ── 体積レンダリングで新視点合成。重い・有界
   ├─► Instant-NGP ── ハッシュ符号化で高速化
   ├─► Mip-NeRF / Mip-NeRF 360 (2022) / NeRF++ ── アンチエイリアス・無限遠(unbounded)
   │       └─[街路特化]─► Block-NeRF（街区分割・Waymo）, S-NeRF（LiDAR+動的車）,
   │                      StreetSurf（SDF表面）, MatrixCity（大規模データ）
   ▼ （陰関数MLP → 明示的ガウシアンへ。高速・編集可）
3D Gaussian Splatting (2023) ── 点群を異方性ガウシアンでスプラット。リアルタイム
   ├─[街路特化]─► Street Gaussians（動的都市）, StreetSurfGS（路面・LiDAR不要）,
   │              MetroGS（大規模・幾何精度）, RoGs（路面）, HUGS/HO-Gaussian
   └─[メッシュ化]─► SuGaR, DN-Splatter ── 表面整合→メッシュ（=コライダー素材）
```
**関係の本質**: NeRF(陰)→3DGS(明示)で **写実×高速** が実用域に。Street View の「見た目」はここ。**本提案の写実層**は 3DGS（または S-NeRF）を採用候補。

## 3. (C) 意味：2Dセグメンテーション → 3Dへリフト

```
FCN/DeepLab（CNN）
   │ （Transformer・統一化）
   ▼
SETR / SegFormer (2021) ── 効率的ViTセグメンテーション（Cityscapes 84% mIoU）
MaskFormer → Mask2Former (2022) ── マスク分類で semantic/instance/panoptic 統一
   └─► OneFormer (2023) ── 1モデルで全タスクSOTA（Cityscapes 68.5 PQ）
SAM (2023) ── プロンプト型万能セグメント
   └─► SAM2 (2024) ── 動画・メモリ注意で時間整合（動的物体の追跡除去）

[2D→3D リフト]
Semantic-NeRF (2021) ── NeRFに意味を載せる
   ├─► Panoptic Lifting (2023) ── 2Dパノプティックを多視点一貫に3Dへ（Hungarianで整合）
   ├─► Contrastive Lift (2023) ── コントラストでインスタンス統合
   └─► PLGS / PanopticSplatting (2024-25) ── 上記の 3DGS 版（高速）
```
**関係の本質**: 検出は seg→panoptic→universal(OneFormer)、動画はSAM→SAM2。3D化は Semantic-NeRF→Panoptic Lifting→3DGS版。**本提案の意味層**はここ（OneFormer/SegFormer＋必要なら3Dリフト）。

## 4. (D) レイアウト/事前分布：床・壁・天井の構造

```
LayoutNet (2018, パノラマ) → HorizonNet (2019, 1D境界) → DuLa-Net / OmniLayout
   └ Manhattan world 仮定（壁⊥床・壁⊥壁）で 床/壁/天井 を平面に
```
**関係の本質**: 屋内パノラマで確立。**街路をコリドー（床＋ファサード＋空）とみなせば流用可**。本提案の床/壁推定の事前分布。

---

## 5. 横断する関係（系統をまたぐ繋がり）

- **幾何(A)→描画(B)**: DUSt3R/DA3/VGGT の点群・姿勢は **3DGS/NeRFの初期化** に使える（COLMAP不要のGS）。例: Gesplat。
- **描画(B)→コライダー**: NeRF/3DGS の密度・表面を **marching cubes/TSDF/SuGaR** でメッシュ化 → 物理コライダー。
- **意味(C)+描画(B)**: Panoptic Lifting/PLGS が **見た目と意味を同じ3Dに**。物体ごとに切り出してコライダー化可能。
- **意味(C)+レイアウト(D)+測量**: seg の床/壁 ＋ Manhattan ＋ **既知カメラ高さ** で、深度なしに**メートルの床/壁/物体**（本提案の核）。
- **追跡(KV-Tracker)**: π³(A)の上に乗り、**歩きながら逐次に地図化・自己位置**＝オンライン化の核。

---

## 6. 本提案の位置づけ（系譜のどこに立つか）

**複数地点の Street View をつなげ、写実な見た目＋ゲーム的コライダー** を作る本提案は、4系統の**交差点**に立つ：

| 層 | 採用（系譜上の出自） | なぜ |
|---|---|---|
| 姿勢 | 既知 heading/pitch ＋ DA3 ray-pose(A) ＋ S-NeRF型ΔP | Street Viewは取得角が既知＝強い初期値 |
| 見た目（写実） | 3DGS / S-NeRF (B) | リアルタイム写実、街路特化が豊富 |
| 衝突（ゲーム的） | seg(C)＋レイアウト(D)＋**測量スケール** → 箱/円柱/平面 | 「写実な面メッシュ」でなく**粗いプリミティブ**で十分・堅牢 |
| 意味・物体 | OneFormer/SAM2 (C)、物体サイズ=接地×画角×カメラ高さ | LiDAR・与えられた3D箱に依存しない |
| 多地点融合 | DA3 Umeyama/TSDF(A) ＋ S-NeRF 信頼度加重 | 「別生成→合体」でなく信頼度融合で重なり解消 |

**新規性の言い換え**: 既存研究は「写実」(B)か「正確な密幾何」(A)を追うが、本提案は
**「写実な見た目」と「ゲーム的な粗いコライダー」を分離**し、後者を**意味＋レイアウト＋測量**で安価・堅牢に作る点が新しい。
ゲームでは衝突が完璧な面である必要はない（箱で十分）——この割り切りが、疎な Street View でも破綻しない鍵。

---

## 7. 年表（ざっくり）

| 年 | 幾何(A) | 描画(B) | 意味(C) | レイアウト(D) |
|---|---|---|---|---|
| 2016 | COLMAP | | | |
| 2018-19 | | | | LayoutNet, HorizonNet |
| 2020 | | NeRF | | |
| 2021 | | Instant-NGP | SegFormer, Semantic-NeRF | |
| 2022 | | Mip-NeRF360, Block-NeRF | Mask2Former | |
| 2023 | | 3DGS, S-NeRF, StreetSurf | OneFormer, SAM, Panoptic Lifting | |
| 2024 | DUSt3R, MASt3R | Street Gaussians, SuGaR | SAM2, PLGS | |
| 2025 | VGGT, **DA3**, MapAnything | StreetSurfGS, MetroGS | PanopticSplatting | |
| 2025-26 | **π³/Pi3X**, KV-Tracker | | | |

---

## 8. 出典（主要）
- COLMAP https://colmap.org/ ; DUSt3R/MASt3R/VGGT評価 https://arxiv.org/abs/2507.14798 ; MASt3R解説 https://learnopencv.com/mast3r-sfm-grounding-image-matching-3d/
- DA3 https://arxiv.org/abs/2511.10647 ; Pi3 https://arxiv.org/abs/2507.13347 ; KV-Tracker https://arxiv.org/abs/2512.22581 ; MapAnything https://arxiv.org/pdf/2509.13414
- NeRF→3DGS survey https://arxiv.org/pdf/2401.03890 ; S-NeRF https://arxiv.org/abs/2303.00749 ; StreetSurf https://arxiv.org/pdf/2306.04988 ; Block-NeRF https://the-decoder.com/google-maps-ai-technology-enables-street-view-3d/
- Street Gaussians https://arxiv.org/pdf/2401.01339 ; StreetSurfGS https://arxiv.org/html/2410.04354v1 ; MetroGS https://arxiv.org/pdf/2511.19172 ; SuGaR https://arxiv.org/abs/2311.12775
- SegFormer https://arxiv.org/pdf/2105.15203 ; Mask2Former https://arxiv.org/pdf/2112.01527 ; OneFormer https://arxiv.org/pdf/2211.06220 ; SAM2 https://arxiv.org/pdf/2503.12781
- Panoptic Lifting https://nihalsid.github.io/panoptic-lifting/ ; Contrastive Lift https://arxiv.org/pdf/2306.04633 ; PLGS https://arxiv.org/html/2410.17505
- レイアウト LayoutNet https://arxiv.org/pdf/1803.08999 ; HorizonNet https://arxiv.org/pdf/1901.03861 ; DuLa-Net https://arxiv.org/pdf/1811.11977
