"""YOLO セグメンテーションで、指定クラス（人・車など）を再構成前に除去する。

各画像にインスタンスセグメンテーションをかけ、除去対象クラスの画素マスクを返す。
呼び出し側はそのマスク位置の深度を無効化して再構成から外す（穴になる）。

CUDA 前提（DA3 と同じ）。ultralytics(YOLO) を遅延 import。
"""
from __future__ import annotations

import os

import numpy as np

# 既定モデル（軽量セグメンテーション）。初回に重みを自動DL（~6MB）。
SEG_MODEL_ID = os.environ.get("YOLO_SEG_MODEL", "yolo11n-seg.pt")
_models: dict[str, object] = {}


def _load(model_id: str | None = None):
    model_id = model_id or SEG_MODEL_ID
    if model_id not in _models:
        from ultralytics import YOLO

        _models[model_id] = YOLO(model_id)
    return _models[model_id]


def class_names(model_id: str | None = None) -> dict[int, str]:
    """このモデルが扱えるクラス名（id->name）。"""
    return dict(_load(model_id).names)


def removal_masks(
    images_uint8: np.ndarray,
    classes_to_remove: list[str],
    conf: float = 0.25,
    dilate_px: int = 6,
    model_id: str | None = None,
) -> np.ndarray:
    """(N,H,W,3) uint8 から、除去対象クラスの画素マスク (N,H,W) bool を返す（True=除去）。

    classes_to_remove: 除去するクラス名のリスト（例 ["person", "car"]）。空なら全False。
    dilate_px: マスクを少し膨張させ、輪郭のハロー（縁の溶け）も一緒に消す。
    """
    n, h, w = images_uint8.shape[:3]
    out = np.zeros((n, h, w), dtype=bool)
    want_names = {c.strip().lower() for c in classes_to_remove if c.strip()}
    if not want_names:
        return out

    import cv2

    model = _load(model_id)
    names = model.names  # id -> name
    want_ids = {i for i, nm in names.items() if str(nm).lower() in want_names}
    if not want_ids:
        return out

    imgs = [np.ascontiguousarray(images_uint8[i][:, :, :3]) for i in range(n)]
    results = model.predict(imgs, conf=conf, verbose=False, retina_masks=True)
    for i, r in enumerate(results):
        if r.masks is None or r.boxes is None:
            continue
        cls = r.boxes.cls.cpu().numpy().astype(int)
        data = r.masks.data.cpu().numpy()  # (k, mh, mw) 0/1
        acc = np.zeros((h, w), dtype=bool)
        for j, c in enumerate(cls):
            if c not in want_ids:
                continue
            mj = data[j]
            if mj.shape != (h, w):
                mj = cv2.resize(mj.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            acc |= mj > 0.5
        if dilate_px > 0 and acc.any():
            k = np.ones((dilate_px, dilate_px), np.uint8)
            acc = cv2.dilate(acc.astype(np.uint8), k, iterations=1) > 0
        out[i] = acc
    return out
