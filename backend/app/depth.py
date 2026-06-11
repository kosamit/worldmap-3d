"""単一画像から相対深度(視差)を推定する。

バックエンドを選択できる:
- dav2: Depth Anything V2 (HuggingFace transformers)。Mac の MPS / CPU 向け。
- da3 : Depth Anything 3 (CUDA 専用)。外部 GPU マシン向け。

いずれも公開コントラクトは同じ: estimate_depth(image) -> 視差マップ (大きいほど手前)。
環境変数 DEPTH_BACKEND = auto(既定) | dav2 | da3 で切り替える。
"""

import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Small=速い/軽い, Base/Large=高品質。MPS では Small が無難。
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

_pipe = None


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _da3_available() -> bool:
    try:
        import depth_anything_3  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - 未インストール/環境差を全て握る
        return False


def active_backend() -> str:
    """実際に使う深度バックエンド名を返す ("da3" | "dav2")。"""
    pref = os.environ.get("DEPTH_BACKEND", "auto").lower()
    if pref in ("da3", "dav2"):
        return pref
    # auto: CUDA があり DA3 が入っていれば DA3、なければ DAv2
    if torch.cuda.is_available() and _da3_available():
        return "da3"
    return "dav2"


def get_pipeline(model_id: str = DEFAULT_MODEL_ID):
    """深度推定パイプラインを遅延初期化して返す（プロセス内で1回だけロード）。"""
    global _pipe
    if _pipe is None:
        from transformers import pipeline

        _pipe = pipeline("depth-estimation", model=model_id, device=_select_device())
    return _pipe


def estimate_depth(image: Image.Image) -> np.ndarray:
    """視差マップ (H, W) float32 を返す。値が大きいほどカメラに近い。

    active_backend() に応じて DA3 / DAv2 へ振り分ける。
    """
    if active_backend() == "da3":
        from . import depth_da3

        return depth_da3.estimate_disparity(image)
    return _estimate_dav2(image)


def _estimate_dav2(image: Image.Image) -> np.ndarray:
    """Depth Anything V2 (transformers) で視差を推定。入力画像サイズへリサイズ。"""
    if image.mode != "RGB":
        image = image.convert("RGB")

    pipe = get_pipeline()
    out = pipe(image)

    depth = out.get("predicted_depth")
    if depth is None:
        # 一部バージョンは正規化済み PIL の "depth" のみ返す
        return np.asarray(out["depth"], dtype=np.float32)

    depth_t = depth.float()
    if depth_t.ndim == 2:
        depth_t = depth_t[None, None]
    elif depth_t.ndim == 3:
        depth_t = depth_t[None]

    width, height = image.size
    depth_t = F.interpolate(
        depth_t, size=(height, width), mode="bicubic", align_corners=False
    )
    return depth_t[0, 0].detach().cpu().numpy().astype(np.float32)
