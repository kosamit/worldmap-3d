"""街路セマンティックセグメンテーション（Cityscapes 19クラス, SegFormer）。

意味コライダー層の検出器。各画素を road/sidewalk/building/.../car/person 等に分類し、
本プロジェクト用のカテゴリ（floor/wall/sky/pole/vegetation/person/vehicle）へ束ねる。

CUDA 前提（DA3 と同じ）。transformers の SegFormer を遅延ロード。
docs/semantic_scene_colliders.md の戦略1の検出器に対応。
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

# 既定は軽量な b1。精度を上げるなら b4/b5 を環境変数で指定。
SEMSEG_MODEL_ID = os.environ.get(
    "SEMSEG_MODEL", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
)

# Cityscapes trainId(0..18) → 本プロジェクトのカテゴリ。
CITYSCAPES_TO_CATEGORY = {
    0: "floor",   1: "floor",   9: "floor",          # road, sidewalk, terrain
    2: "wall",    3: "wall",    4: "wall",           # building, wall, fence
    10: "sky",                                       # sky
    5: "pole",    6: "pole",    7: "pole",           # pole, traffic light, traffic sign
    8: "vegetation",                                 # vegetation
    11: "person", 12: "person",                      # person, rider
    13: "vehicle", 14: "vehicle", 15: "vehicle",     # car, truck, bus
    16: "vehicle", 17: "vehicle", 18: "vehicle",     # train, motorcycle, bicycle
}
CATEGORIES = ["floor", "wall", "sky", "pole", "vegetation", "person", "vehicle", "other"]

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        _processor = SegformerImageProcessor.from_pretrained(SEMSEG_MODEL_ID)
        _model = SegformerForSemanticSegmentation.from_pretrained(SEMSEG_MODEL_ID)
        _model = _model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return _model, _processor


def unload() -> None:
    """ロード済みセグメンテーションモデルを解放してGPUメモリを空ける。"""
    global _model, _processor
    _model = None
    _processor = None


def segment_trainids(images: list[Image.Image], batch_size: int = 4) -> np.ndarray:
    """画像群 → Cityscapes trainId ラベルマップ (N,H,W) int16（入力解像度に合わせて補間）。"""
    import torch
    import torch.nn.functional as F

    model, processor = _load()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        sizes = [(im.height, im.width) for im in batch]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits  # (B,C,h/4,w/4)
        for j, (h, w) in enumerate(sizes):
            up = F.interpolate(logits[j : j + 1], size=(h, w), mode="bilinear", align_corners=False)
            out.append(up.argmax(1)[0].to("cpu", torch.int16).numpy())
    return np.stack(out)


def trainids_to_category(trainids: np.ndarray) -> np.ndarray:
    """trainId マップ → カテゴリ index マップ（CATEGORIES の添字, int8）。未知は 'other'。"""
    other = CATEGORIES.index("other")
    cat = np.full(trainids.shape, other, dtype=np.int8)
    for tid, name in CITYSCAPES_TO_CATEGORY.items():
        cat[trainids == tid] = CATEGORIES.index(name)
    return cat


def segment_categories(images: list[Image.Image]) -> np.ndarray:
    """画像群 → カテゴリ index マップ (N,H,W) int8（floor/wall/sky/...）。"""
    return trainids_to_category(segment_trainids(images))
