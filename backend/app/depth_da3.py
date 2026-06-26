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

# モデルIDごとにロード済みモデルをキャッシュ（切替時に再ロードできるよう dict 化）。
_models: dict[str, object] = {}
_EPS = 1e-6


def _load(model_id: str | None = None):
    model_id = model_id or DA3_MODEL_ID
    if model_id not in _models:
        from depth_anything_3.api import DepthAnything3

        model = DepthAnything3.from_pretrained(model_id)
        _models[model_id] = model.to(device="cuda")
    return _models[model_id]


def estimate_disparity(image: Image.Image, model_id: str | None = None) -> np.ndarray:
    """視差マップ (H, W) float32 を返す。値が大きいほどカメラに近い。"""
    if image.mode != "RGB":
        image = image.convert("RGB")

    model = _load(model_id)

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
