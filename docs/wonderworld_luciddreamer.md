# WonderWorld / LucidDreamer 調査メモ（③ 1枚→歩けるシーン生成）

選択肢③「1枚の画像から生成AIで奥行きのあるシーンを丸ごと作る」系の2大手法を、
本プロジェクト（Street View → 歩ける3D, GPU=RTX 4070 Ti SUPER 16GB）への適用観点で調査。

調査日: 2026-06-27。出典は末尾。

---

## 結論（先に）

| 観点 | WonderWorld | LucidDreamer |
|---|---|---|
| **16GBで動くか** | ❌ **48GB必須**（公式明記）→不可 | △ コンシューマGPUで動作実績あり（16GBはやや厳しいが現実的） |
| シーン生成 | 対話的・高速（A6000で1シーン<10秒） | バッチ（数分） |
| 出力 | Fast Layered Gaussian Surfels (FLAGS) | **3D Gaussian Splatting (.ply)** |
| ライセンス | 不明記（要確認） | **CC-BY-NC-SA-4.0（非商用）** |
| 入力 | 1枚＋テキスト＋カメラ操作 | 1枚＋テキストプロンプト |
| 本プロジェクト適合 | VRAMで脱落 | 候補になりうる |

- **WonderWorld は 48GB GPU 必須**で、この 16GB マシンでは動きません（クラウドA6000/A100が要る）。
- **LucidDreamer は 16GB で動かせる可能性がある**唯一の現実解。ただし出力が
  **Gaussian Splat (.ply)** なので、フロントに**スプラット描画**を足す必要がある（今は glTFメッシュ）。
- **重大な性質**: ③系は両方とも「1枚から世界を**創作（ハルシネーション）**して広げる」。
  歩けるが、奥はAIの想像で**実際のStreet Viewの場所を忠実に再現するものではない**。
  これは「その場所をちゃんと再現したい」目的とは方向性が違う点に注意。

---

## WonderWorld（CVPR 2025, Stanford 等）

**概要**: 1枚の画像から、対話的に「連結した」3Dシーンを次々生成して歩き回れる。
ユーザーがテキストで内容を、カメラ移動で配置を指定すると低遅延で世界が広がる。

**技術**:
- 表現 = **Fast Layered Gaussian Surfels (FLAGS)**（層状＋サーフェルでジオメトリ初期化）
- **Guided depth diffusion**: 深度推定を部分条件づけして、新領域を既存と整合させる
- 構成要素: Marigold(深度) / Stable Diffusion 2 Inpainting(生成) / OneFormer(セグメンテーション) / RepViT-SAM

**要件（重要）**:
- **GPU VRAM 48GB 必須**（公式README明記）→ **RTX 4070 Ti SUPER 16GB では不可**
- PyTorch 2.4.0 / CUDA 12.4、カスタムCUDA: `depth-diff-gaussian-rasterization-min`, `simple-knn`
- A6000(48GB)で 1シーン < 10秒

**操作**: W/A/S/D 移動、I/J/K/L 視点、キャンバスクリックで操作開始。

**適合性**: 速くて高品質だが **VRAMで脱落**。やるならクラウド(A6000/A100 48GB)前提。

---

## LucidDreamer（"Domain-free Generation of 3D Gaussian Splatting Scenes"）

**概要**: 大規模拡散モデルを使い、ドメインを問わず1枚＋テキストから
高精細な **3D Gaussian Splatting** シーンを生成。屋内/屋外/絵画など何でも。

**技術（Dreaming ↔ Alignment の交互反復）**:
1. **Dreaming**: 既存点群を目標視点へ投影し、その投影を**ガイドにインペイント**して
   多視点整合な画像を生成
2. **Alignment**: 生成画像を深度推定で3Dへ持ち上げ、新しい点群として合成
3. 反復で点群を成長させ、最後に 3D Gaussian Splatting を最適化

**要件**:
- Python **3.9 固定**（Open3D の都合で3.10不可）、CUDA ≥11.4
- PyTorch **2.0.1** / torchvision 0.15.2、diffusers, peft, open3d
- ZoeDepth(timm==0.6.7), plyfile、カスタムCUDA: `depth-diff-gaussian-rasterization-min`, `simple-knn`
- VRAM: README明記なし。SD-inpaint + ZoeDepth + GS最適化で実測 ~16–24GB。
  HuggingFace に通常版/`mini`版デモあり → **16GBでも mini相当の設定なら可能性あり**
- **ライセンス CC-BY-NC-SA-4.0（非商用）** ← 商用利用に注意

**入出力**: 入力=1枚＋テキスト（任意でネガティブプロンプト）。出力=**`.ply`(Gaussian Splat)**。
SuperSplat や WebGL系スプラットビューワーで閲覧。

**既知の弱点**: 屋内はプロンプトを簡潔にした方が良い／物体の重複が出やすい
（ネガティブプロンプトで調整）／インペイント品質に依存。

---

## ③ を採用する場合のフロント影響（重要）

両者とも出力が **Gaussian Splat**。現在のビューワーは **glTF メッシュ**（three.js + WASD + BVH衝突）。
→ ③にするなら **スプラット描画パスを追加**する必要がある:
- 例: `@mkkellogg/gaussian-splats-3d`（three.js用GSビューワー）等を導入
- 衝突判定（WASDで歩く床/壁）はスプラットには無いので、別途近似（深度メッシュ併用 or 簡易当たり）

---

## ① / ② / ③ の位置づけ比較

| | ① 3DGS(複数写真) | ② 動画拡散 遮蔽補完 | ③ WonderWorld/LucidDreamer |
|---|---|---|---|
| 実際の場所の忠実度 | **高**（実写真を学習） | 高（実点群＋穴だけ生成） | **低**（奥はAIの創作） |
| 奥行き/視差 | 本物 | 実部は本物・穴は近似/生成 | 生成（それっぽい） |
| 隙間 | 出る | ほぼ無し | 無し（創作で連続） |
| 16GB | 可（最適化型は要時間） | **可**（軽量版は導入済で動作確認済） | WW=不可 / LD=ギリ |
| 出力 | Splat or Mesh | **Mesh（既存ビューワーそのまま）** | Splat（要ビューワー追加） |
| 実装状況 | 未 | **②lite 実装済・動作確認済** | 未（要DL・要GS描画） |

---

## 推奨

- **「その場所を忠実に歩きたい」が主目的なら ③ は不向き**（奥がAIの創作になり、実際の駅/街と別物になる）。
  すでに動いている **②lite（実点群＋穴だけ生成補完, 既存メッシュビューワーで歩ける）** を磨く方が目的に合う。
- **「1枚からそれっぽい歩ける世界を作る」体験自体が目的なら ③**。その場合:
  - WonderWorld は 48GB 必須 → **クラウドGPU前提**。
  - **LucidDreamer が 16GB ローカルの現実解**。ただし非商用ライセンス＋Gaussian Splat描画の追加が要る。
- 折衷: ②lite で実形状を固め、足りない奥だけ ③系の生成で補う、も将来あり得る。

---

## 出典
- WonderWorld 論文: https://arxiv.org/abs/2406.09394 — コード: https://github.com/KovenYu/WonderWorld
- LucidDreamer 論文: https://arxiv.org/abs/2311.13384 — コード: https://github.com/luciddreamer-cvlab/LucidDreamer
- LucidDreamer プロジェクト: https://luciddreamer-cvlab.github.io/
