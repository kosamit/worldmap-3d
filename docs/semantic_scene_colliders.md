# 意味検出 → 3D衝突判定シーン（DA3任意・NeRF可） 調査メモ

調査日: 2026-06-28（逐次追記する生きた調査ノート）
方針（ユーザー指示）: **DA3は捨ててもよい**。画像集合から **床/壁/天井(空)** と **ビル/車/人/オブジェクト** を意味で検出し、そこに **3D衝突判定(Collider)** を置く。見た目は **NeRFが使えるならNeRF**。

関連: [town_3d_reconstruction_survey.md](town_3d_reconstruction_survey.md)（再構成技術の全体像）、[generative_walkable_3d.md](generative_walkable_3d.md)

---

## 0. 全体像（2つの geometry 戦略）

「3Dの位置」をどう得るかで2路線。**どちらもDA3不要**。

- **戦略1: レイアウト＋接地（軽量・深度モデル不要）** ← まず推奨
  パノラマ（equirect）に **レイアウト推定**で床/壁/天井の境界を出し、**既知カメラ高さ**で平面を解析的に確定（先日の床RANSACの発展）。物体は **2D検出 → 地面接地点**で3D配置。深度推定モデルが一切いらない。
- **戦略2: Semantic/Panoptic NeRF・3DGS（写真品質・重い）**
  多視点から NeRF/3DGS を学習（見た目）＋ **Panoptic Lifting** で2Dセグメンテーションを3Dへリフト（意味）。**メッシュ抽出**して衝突判定に。

共通の検出器: **セマンティック/パノプティック・セグメンテーション**（下記）。

---

## 1. 検出器：セマンティック／パノプティック・セグメンテーション

街路の画素を「床(road/sidewalk)・壁(building/wall)・空(sky)・ポール・標識・植生・車・人」に分類する中核。

| モデル | 要点 | 参考 |
|---|---|---|
| **OneFormer** | Cityscapes パノプティックで **SOTA(68.5 PQ)**。1モデルで semantic/instance/panoptic 統一。床壁天井＋物体を一括 | [GitHub](https://github.com/SHI-Labs/OneFormer) / [arXiv](https://arxiv.org/pdf/2211.06220) |
| **Mask2Former** | panoptic/instance/semantic 全部でSOTA級。COCO/ADE20K/Cityscapes | [arXiv](https://arxiv.org/pdf/2112.01527) / [site](https://mask2former.com/) |
| **SegFormer** | B5で **Cityscapes 84.0% mIoU**、軽量・高速（SETRの5×速・4×小） | [arXiv](https://arxiv.org/pdf/2105.15203) |
| **SAM2** | 動画整合マスクで**動的物体(車/人)を視点跨ぎ除去・追跡** | [survey](https://arxiv.org/pdf/2503.12781) |

**クラス対応（Cityscapes）**: road/sidewalk=床, building/wall/fence=壁, sky=空(天井扱い), pole/traffic-sign/traffic-light=柱, vegetation/terrain=植生, person/rider=人, car/truck/bus=車。
→ 「床・壁・天井(空)・ビル・車・人・物体」をそのまま満たす。

**役割**: パノプティックで「クラス＋インスタンス」が出るので、**インスタンス毎にコライダー**を作れる（この車・あの柱）。

---

## 2. 戦略1のコア：レイアウト推定（床/壁/天井） + 接地配置

深度モデル無しで「床・壁・天井」の3D平面を出す。

### 2-1. レイアウト推定（パノラマ→床/壁/天井境界）
- 屋内パノラマで実績のある **HorizonNet / LayoutNet / DuLa-Net**：equirectから **天井-壁・壁-壁・床-壁の境界**を1Dベクトルで予測 → 部屋の箱レイアウト。
- **Manhattan world仮定**（壁は床に直交・互いに直角）で平面フィット。非Manhattanへの拡張版もあり。
- 街路は「屋内の部屋」ではないが、**コリドー（床＋両脇の壁＝ファサード＋空）** とみなせば同じ枠組みが効く。我々は既に equirect を持っているので相性が良い。
- 参考: [HorizonNet](https://arxiv.org/pdf/1901.03861), [LayoutNet](https://arxiv.org/pdf/1803.08999), [DuLa-Net](https://arxiv.org/pdf/1811.11977), [非Cuboid](https://arxiv.org/pdf/2104.07986)

### 2-2. 解析的に平面化（既知カメラ高さ＝測量）
- **床**: セグメンテーションの road/sidewalk 画素 → 既知カメラ高さ(≈2.5m)の水平平面にレイ交差で配置（先日実装した `_ground_plane_cap` の発想）。
- **壁(ファサード)**: building 画素の **床との境界線(footprint)** から鉛直に立ち上げ。高さは天井境界 or 一定。
- **天井/空**: sky 画素 → 遠景ドーム or 無限遠（衝突なし）。
→ **深度推定ゼロ**で床/壁/天井の平面コライダーが出る。

### 2-3. 物体の接地配置（車・人・柱）
- 2D検出（OneFormer instance / YOLO）で物体の **2Dボックス下端（接地点）** を取得。
- その画素のレイを **床平面に交差** → 物体の3D足元位置。
- サイズはボックスの見かけ＋距離からスケール推定 → **箱/円柱コライダー**を設置。
- 動的物体は SAM2 で除去（背景化）も選べる。

→ これで「ビル/車/人/柱に3D衝突判定」が深度モデル無しで成立。

---

## 3. 戦略2のコア：Semantic/Panoptic NeRF・3DGS（写真品質）

見た目をNeRF/3DGSで作り、意味を3Dへリフトして一貫した3D意味シーンにする。

| 手法 | 要点 | 参考 |
|---|---|---|
| **Panoptic Lifting** | 2Dパノプティックマスクを**多視点一貫**に3D neural fieldへリフト（Hungarianで視点間インスタンスID整合） | [site](https://nihalsid.github.io/panoptic-lifting/) |
| **Contrastive Lift** | コントラスト学習でインスタンスを3D統合 | [arXiv](https://arxiv.org/pdf/2306.04633) |
| **PLGS / PanopticSplatting** | 上記の **3DGS版**（NeRFより高速・効率） | [PLGS](https://arxiv.org/html/2410.17505) / [PanopticSplatting](https://arxiv.org/pdf/2503.18073) |

→ 出力は「クラス・インスタンス付きの3D」。建物/車/人ごとに切り出してコライダー化できる。

### 3-1. コライダー（衝突メッシュ）抽出
NeRF/3DGSの体積表現 → ゲーム用コライダーへ。
- **SuGaR / DN-Splatter**: 3DGSを表面整合させ**メッシュ抽出** | [SuGaR](https://arxiv.org/abs/2311.12775), [DN-Splatter](https://openaccess.thecvf.com/content/WACV2025/papers/Turkulainen_DN-Splatter_Depth_and_Normal_Priors_for_Gaussian_Splatting_and_Meshing_WACV_2025_paper.pdf)
- **TSDF融合＋Marching Cubes / Poisson**: 密度グリッド→等値面→メッシュ（物理エンジン/コライダー向き）。細い構造は落ちやすい。

---

## 4. 提案アーキテクチャ（2層）

```
入力: Street View 画像集合（複数視点）

[見た目レイヤー]  パノラマ生成3D（今） → 将来 NeRF/3DGS
[意味コライダーレイヤー]（新規・本題）
  各視点/パノラマ
   → セグメンテーション(OneFormer/Mask2Former, Cityscapes)
   → 戦略1: レイアウト＋既知カメラ高さで 床/壁/天井 平面化
            物体(車/人/柱)は 2D検出→床接地点→箱/円柱
     （または戦略2: Panoptic NeRF/3DGS → メッシュ抽出）
   → クラス＋インスタンス＋形状(box/cylinder/plane)＋transform を JSON 出力
  フロント
   → 不可視コライダーを生成（既存 colliderRef で衝突）
   → 任意でクラス別 3Dアセット配置（木/車/標識 等）
```

**再利用できる既存資産**
- `primitives.py`: 平面→薄箱、クラスタ→箱/円柱フィット（形状部分）
- `segment.py`: YOLOセグメンテーション（動的物体）＋LaMa穴埋め
- `_ground_plane_cap`（panorama3d.py）: 既知カメラ高さの床平面（戦略1の床）
- ビューワー `colliderRef`: BVH当たり判定（既に衝突する）
- equirect パイプライン: レイアウト推定の入力にそのまま使える

**新規**
- セマンティック/パノプティック・セグメンテーション（OneFormer/Mask2Former/SegFormer）
- レイアウト推定（HorizonNet系, 戦略1）or Panoptic Lifting（戦略2）
- クラスタグ付きコライダーJSON出力
- フロントの不可視コライダー生成＋3Dアセット配置

---

## 5. 推奨ロードマップ

1. **MVP（戦略1・深度不要）**: SegFormer/OneFormerで road/building/pole/car/person を分類 → 床=既知高さ平面、壁=ファサード箱、柱=円柱、車/人=接地箱 → **クラスタグ付き不可視コライダーJSON** → フロントで衝突。
2. **見た目**: 当面パノラマ。良ければ 3DGS（[survey](town_3d_reconstruction_survey.md) 参照）。
3. **動的物体**: SAM2で視点跨ぎ除去 → きれいな静的背景。
4. **発展（戦略2）**: Panoptic Lifting/PLGS で多視点一貫の3D意味 → SuGaR等でコライダーメッシュ。
5. **3Dアセット配置**: クラス別に木/車/標識モデルを差し替え。

---

## 6. 出典
- OneFormer: https://github.com/SHI-Labs/OneFormer , https://arxiv.org/pdf/2211.06220
- Mask2Former: https://arxiv.org/pdf/2112.01527 , https://mask2former.com/
- SegFormer: https://arxiv.org/pdf/2105.15203
- SAM2: https://arxiv.org/pdf/2503.12781
- レイアウト: HorizonNet https://arxiv.org/pdf/1901.03861 ; LayoutNet https://arxiv.org/pdf/1803.08999 ; DuLa-Net https://arxiv.org/pdf/1811.11977 ; 非Cuboid https://arxiv.org/pdf/2104.07986
- Panoptic Lifting: https://nihalsid.github.io/panoptic-lifting/ ; Contrastive Lift https://arxiv.org/pdf/2306.04633 ; PLGS https://arxiv.org/html/2410.17505 ; PanopticSplatting https://arxiv.org/pdf/2503.18073
- メッシュ/コライダー抽出: SuGaR https://arxiv.org/abs/2311.12775 ; DN-Splatter https://openaccess.thecvf.com/content/WACV2025/papers/Turkulainen_DN-Splatter_Depth_and_Normal_Priors_for_Gaussian_Splatting_and_Meshing_WACV_2025_paper.pdf
