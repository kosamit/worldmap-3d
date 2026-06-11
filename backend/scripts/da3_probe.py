"""Depth Anything 3 の実APIを検証するプローブ（PC/WSL2 で手動実行）。

目的: prediction の属性・depth等の shape/dtype/値域を実機で確認し、
depth_da3.py を正しく合わせるための情報を出す。フルサーバ起動前の疎通確認。

使い方 (backend ディレクトリで venv 有効化後):
    python scripts/da3_probe.py                 # 合成画像で検証
    python scripts/da3_probe.py path/to/img.jpg # 任意画像で検証
    DA3_MODEL=depth-anything/DA3-BASE python scripts/da3_probe.py
"""

import os
import sys

import numpy as np
from PIL import Image

TMP_INPUT = "/tmp/da3_probe_input.png"


def make_test_image(width: int = 640, height: int = 640) -> Image.Image:
    xs = np.linspace(0, 255, width)
    ys = np.linspace(0, 255, height)
    r = np.tile(xs, (height, 1))
    g = np.tile(ys.reshape(-1, 1), (1, width))
    b = np.full((height, width), 128)
    arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def describe(name, value):
    if value is None:
        print(f"  {name}: None/absent")
        return
    try:
        arr = np.asarray(value)
    except Exception as exc:  # noqa: BLE001
        print(f"  {name}: <{type(value).__name__}> (asarray失敗: {exc})")
        return
    extra = ""
    if arr.dtype.kind == "f" and arr.size:
        extra = f" min={float(arr.min()):.4f} max={float(arr.max()):.4f}"
    print(f"  {name}: shape={arr.shape} dtype={arr.dtype}{extra}")


def main():
    image = (
        Image.open(sys.argv[1]).convert("RGB")
        if len(sys.argv) > 1
        else make_test_image()
    )
    model_id = os.environ.get("DA3_MODEL", "depth-anything/DA3METRIC-LARGE")

    import torch

    print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
    print("model:", model_id)

    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_id).to(device="cuda")

    image.save(TMP_INPUT)
    prediction = model.inference([TMP_INPUT])

    print("prediction type:", type(prediction).__name__)
    print("public attrs:", [a for a in dir(prediction) if not a.startswith("_")])
    print("fields:")
    for name in ("depth", "conf", "intrinsics", "extrinsics", "processed_images"):
        describe(name, getattr(prediction, name, None))

    print("PROBE_OK")


if __name__ == "__main__":
    main()
