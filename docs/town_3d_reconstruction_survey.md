# 街を画像から3D化する技術 — 調査まとめ

調査日: 2026-06-28 / 対象: Street View 風画像から「歩ける街の3D」を作る技術。
本プロジェクト（DA3 マルチビュー＋パノラマ生成3D）の文脈に紐づけて整理する。

---

## 0. 結論（先に）

- **Depth Anything 3 (DA3) は、実は COLMAP の「上位互換」で本分野の現SOTA。** 別技術に乗り換える話ではなく、**DA3を「多視点融合」として正しく使えていない**のが今の課題。
- DA3 は「**任意枚数の視点から、ポーズ有無を問わず、空間的に整合した深度・rayマップを出力 → 1つの点群に融合**」するために作られている。実測で VGGT を **ポーズ+35.7% / 幾何+23.6%** 上回り、大規模で **COLMAP（48h）より良い**。
- よって本命の方向は「**別々に球を作って重ねる → DA3本来の“1つに融合”へ**」。あなたの「別生成→合体は失敗」の直感は正しい。
- 見た目を写真品質で歩けるようにするなら **3D Gaussian Splatting (3DGS)** が実用本命。
- 補助技術として **Pi3/Pi3X**（DA3 の競合/兄弟）、**SAM2**（動的物体の除去・整合マスク）、**KV-Tracker**（リアルタイム姿勢追跡）が有用。

---

## 1. 古典の定番: COLMAP（SfM + MVS）

[COLMAP](https://colmap.org/) — 写真から3D化する標準パイプライン。
- **SfM (Structure-from-Motion)**: 特徴点(SIFT)→画像間マッチ→**カメラ位置姿勢＋疎点群**を復元（バンドル調整で再投影誤差最小化）。
- **MVS (Multi-View Stereo)**: 姿勢を使って **密な点群→メッシュ** 化（PatchMatch ステレオ → 融合 → Poisson/Delaunay）。
- 実績: ローマ中心部を 2.1万枚で復元など大規模対応。

**街路での弱点**
- 重複の少ない**広いベースライン**・**繰り返しテクスチャ（似た窓・壁）**・**動く物体**・**テクスチャレスな壁/路面**でマッチ破綻＆穴。
- 大規模で**非常に遅い**（48時間級）。早期の強フィルタで穴が増える。

→ Street View は「疎・広ベースライン・繰り返し」なので COLMAP は本質的に不利。

---

## 2. 学習系フィードフォワード再構成（＝COLMAPの現代版）

「特徴点+RANSAC」を Transformer が end-to-end で置換。**少数・低重複でも密に復元**でき、ポーズ不要が多い。

| モデル | 要点 | 参考 |
|---|---|---|
| **DUSt3R** | 2枚から pose-free 密点群。系統の起点 | — |
| **MASt3R** | DUSt3R＋高精度マッチング | — |
| **VGGT** | 多視点を同時に、ポーズ＋幾何を一括。COLMAP比 完成度+50% | [arXiv](https://arxiv.org/html/2503.11651v1) |
| **🏆 Depth Anything 3** | **VGGT超え・現SOTA**。整合depth/rayを融合→点群→3DGS/メッシュ。ポーズ有無不問 | [論文](https://arxiv.org/abs/2511.10647) / [サイト](https://depth-anything-3.github.io/) |
| **🆕 π³ (Pi3) / Pi3X** | **固定参照ビュー無し**の置換不変(permutation-equivariant)設計。入力順に頑健・高スケーラブル。ポーズ/深度/点群でSOTA。ICLR 2026 | [論文](https://arxiv.org/abs/2507.13347) / [GitHub](https://github.com/yyfz/Pi3) |
| MapAnything / MV-DUSt3R+ | メトリック対応・数秒で疎視点復元（CVPR2025） | [MapAnything](https://arxiv.org/pdf/2509.13414) / [MV-DUSt3R+](https://mv-dust3rp.github.io/) |

### π³ (Pi3) / Pi3X — 注目の追加調査
- **特徴**: 従来手法（DUSt3R/VGGT/DA3 の一部）が「最初のビューを基準フレーム」にするのに対し、**π³ は基準ビューを持たない置換不変アーキ**。affine不変なカメラ姿勢＋scale不変なローカル点群を予測 → **入力順序に頑健で崩れにくい**。
- **Pi3X（2025-12-28）**: グリッド状アーティファクト除去（滑らかな点群）・高精度な信頼度・**条件注入（カメラ姿勢/内部パラメータ/深度）**・**近似メトリックスケール**復元に対応。
- **本プロジェクトへの含意**: 我々の悩み「視点を跨ぐと崩れる/重なる」はまさに**基準ビュー依存とスケール不整合**が原因。π³ の「基準ビュー無し＋メトリック」は **DA3 の代替/比較候補**として有力。Pi3X の「条件注入」で**既知GPS姿勢・カメラ高さ**を入れれば、我々の測量アプローチと噛み合う。

---

## 3. 見た目を作る（歩ける写真品質）

点群/姿勢が出た後、写真品質で歩けるようにする描画技術。

**NeRF系（街路）**
- [Block-NeRF](https://the-decoder.com/google-maps-ai-technology-enables-street-view-3d/): Googleがサンフランシスコを街区分割NeRFで再構成（学習9–24h / TPU×32、重い）。
- [StreetSurf](https://arxiv.org/pdf/2306.04988), [S-NeRF](https://arxiv.org/pdf/2303.00749): 街路特化NeRF。

**3D Gaussian Splatting系（実用本命）** — 高速描画・編集可・点群から初期化しやすい:
- [Street Gaussians](https://arxiv.org/pdf/2401.01339)（動的都市）
- [StreetSurfGS](https://arxiv.org/html/2410.04354v1)（LiDAR不要・路面メッシュ）
- [MetroGS](https://arxiv.org/pdf/2511.19172)（大規模・幾何精度）
- [RoGs](https://arxiv.org/pdf/2405.14342)（路面特化）, HUGS / HO-Gaussian（自動運転の街路）
- 動的物体除去つき高品質街路再構成（[arXiv 2503.12001](https://arxiv.org/abs/2503.12001)）

---

## 4. 補助技術（追加調査）

### SAM2 — Segment Anything 2（動的物体の除去・整合マスク）
- **動画対応のセグメンテーション**。Memory Attention で**フレーム間を追跡**し、時間的に整合したマスク(masklet)を生成。クリック/ボックスでプロンプト可。
- **3D用途**: [SAM2Point](https://sam2point.github.io/) はゼロショットで3D（点群/LiDAR/屋内外）をセグメント。動画として3Dを扱う。
- **本プロジェクトへの含意**: 街路再構成の大敵「**人・車などの動的物体**」を、**視点を跨いで整合したマスクで除去**できる（今の単発YOLOより安定）。除去後に穴をLaMa/拡散で補完 → きれいな静的シーン。
- 参考: [SAM2 survey](https://arxiv.org/pdf/2503.12781), [Video SAM review](https://arxiv.org/pdf/2507.22792)

### KV-Tracker — Real-Time Pose Tracking with Transformers
- **π³ の上に構築**。キーフレームで π³（双方向アテンション）でシーンを地図化し、**global self-attention の Key-Value をキャッシュ**してシーン表現として保持 → 以降は**単一クエリフレームで再ローカライズ**。
- **リアルタイム6-DoF姿勢追跡＋オンライン再構成**を単眼RGB動画から。**最大15×高速化**、ドリフト/破滅的忘却なし。30FPS（〜50フレーム）。**モデル非依存**（他の多視点ネットにも適用可、再学習不要）。
- 参考: [サイト](https://marwan99.github.io/kv_tracker/) / [arXiv](https://arxiv.org/abs/2512.22581)
- **本プロジェクトへの含意**: 将来「歩きながら逐次に近傍パノを地図化・自己位置推定」する**オンライン化**の核になりうる。KVキャッシュは**DA3にも適用可能**（モデル非依存）なので、DA3地図を一度作れば軽量に追跡・拡張できる。

---

## 5. 本プロジェクトへの推奨パイプライン（難易度順）

**案A（最小・今すぐ）— DA3を“正しく融合”**
全視点を1回のDA3多視点推論に通し、整合 point/ray マップを **1つの点群に融合** → TSDF/Poisson で**単一面**化。GPSは**全体のスケール・位置の固定だけ**に使う（DA3ドリフト補正）。
→ 「別々の球を重ねる」現状を、DA3本来の「1つに融合」へ。重なり問題の根本解決。

**案B（中・本命の見た目）— DA3 + 3DGS**
案Aの点群＋DA3ポーズで **3DGSを初期化・最適化** → 写真品質で滑らかに歩ける。街路3Dの現実的SOTA。

**案C（比較検証）— Pi3X / MASt3R でポーズ・幾何を補強**
DA3が崩れる箇所を **π³X（基準ビュー無し＋条件注入でGPS/高さ注入）** や MASt3R で補強 → 案B へ。

**前処理（全案共通）— SAM2 で動的物体除去**
人・車を**視点跨ぎ整合マスク**で除去 → 穴埋め → 静的シーンを再構成。

**将来（オンライン化）— KV-Tracker**
歩行に合わせて近傍を逐次地図化＋自己位置推定（DA3にKVキャッシュ適用）。

---

## 6. 次の一手

おすすめは **案A → 案B**（まず融合を正す → 3DGSで見た目を上げる）。
動的物体が気になる現場では **SAM2 前処理**を先に入れると効果大。

深掘り候補（本文取得して実装粒度で要約可能）: DA3本体 / StreetSurfGS / MetroGS / Pi3X / KV-Tracker。

---

## 出典
- DA3: https://arxiv.org/abs/2511.10647 , https://depth-anything-3.github.io/
- COLMAP: https://colmap.org/
- VGGT/DUSt3R/MASt3R 評価: https://arxiv.org/abs/2507.14798 ; VGGT: https://arxiv.org/html/2503.11651v1
- MV-DUSt3R+: https://mv-dust3rp.github.io/ ; MapAnything: https://arxiv.org/pdf/2509.13414
- π³ (Pi3): https://arxiv.org/abs/2507.13347 , https://github.com/yyfz/Pi3
- SAM2: https://arxiv.org/pdf/2503.12781 , SAM2Point: https://sam2point.github.io/ , video review: https://arxiv.org/pdf/2507.22792
- KV-Tracker: https://marwan99.github.io/kv_tracker/ , https://arxiv.org/abs/2512.22581
- 3DGS街路: Street Gaussians https://arxiv.org/pdf/2401.01339 ; StreetSurfGS https://arxiv.org/html/2410.04354v1 ; MetroGS https://arxiv.org/pdf/2511.19172 ; RoGs https://arxiv.org/pdf/2405.14342
- NeRF街路: Block-NeRF https://the-decoder.com/google-maps-ai-technology-enables-street-view-3d/ ; StreetSurf https://arxiv.org/pdf/2306.04988 ; S-NeRF https://arxiv.org/pdf/2303.00749
