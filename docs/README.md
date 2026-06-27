# WorldMap 3D — ドキュメント

Street View 風の画像集合から「歩ける街の3D」を作るプロジェクトの設計・調査メモ。
このページは Markdown のままブラウザで閲覧できます（docsify）。

## 閲覧方法

```bash
./docs/serve.sh            # → http://localhost:8088
# 任意ポート: ./docs/serve.sh 9000
```

`.md` の拡張子はそのまま。ファイルを編集すれば、ブラウザ再読込で即反映されます。

## もくじ

### 調査（リサーチ）
- [街を画像から3D化 技術調査](town_3d_reconstruction_survey.md) — COLMAP / DA3 / VGGT / Pi3 / 3DGS / NeRF / SAM2 / KV-Tracker の全体像
- [意味検出 → 3D衝突判定シーン](semantic_scene_colliders.md) — 床/壁/天井・ビル/車/人を検出してColliderを置く（DA3任意・NeRF可）

### 設計メモ
- [生成・隙間なく歩ける3D](generative_walkable_3d.md)
- [WonderWorld / LucidDreamer](wonderworld_luciddreamer.md)

---

> メモ: 表示には docsify を CDN から読み込むため、初回はネット接続が必要です。
