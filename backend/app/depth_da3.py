"""Depth Anything 3 (CUDA 専用) を現行パイプラインのドロップインとして使う。

DA3 は深度(遠いほど大)を返すため、disparity = 1/depth に変換して
estimate_disparity(image) -> 視差(大きいほど手前) のコントラクトに合わせる。

注: DA3 / xformers は CUDA 前提。Mac(MPS) では import 不可なので、この
モジュールは active_backend()=="da3" のときだけ遅延 import される。
PC 側でのみ実行・検証すること。
"""

import os
import tempfile

import numpy as np
from PIL import Image

# DA3METRIC-* はメートル絶対値。相対で良ければ DA3-LARGE 等でも可。
DA3_MODEL_ID = os.environ.get("DA3_MODEL", "depth-anything/DA3METRIC-LARGE")

_model = None
_EPS = 1e-6


def _load():
    global _model
    if _model is None:
        import torch
        from depth_anything_3.api import DepthAnything3

        model = DepthAnything3.from_pretrained(DA3_MODEL_ID)
        _model = model.to(device="cuda")
    return _model


def estimate_disparity(image: Image.Image) -> np.ndarray:
    """視差マップ (H, W) float32 を返す。値が大きいほどカメラに近い。"""
    if image.mode != "RGB":
        image = image.convert("RGB")

    model = _load()

    # DA3 の inference は画像パスのリストを受ける
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
        image.save(path)
    try:
        prediction = model.inference([path])
    finally:
        os.unlink(path)

    depth = np.asarray(prediction.depth[0], dtype=np.float32)  # (H, W) 遠い=大
    disparity = 1.0 / np.clip(depth, _EPS, None)  # 手前=大

    # 入力画像サイズに合わせる（DA3 は内部解像度で返すことがある）
    width, height = image.size
    if disparity.shape != (height, width):
        disp_img = Image.fromarray(disparity)
        disparity = np.asarray(
            disp_img.resize((width, height), Image.BILINEAR), dtype=np.float32
        )
    return disparity
