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

# モデルIDごとにパイプラインをキャッシュ（モデル切替時に再ロードできるよう dict 化）。
_pipes: dict[str, object] = {}

# UI のドロップダウン用プリセット。active_backend() に応じて出し分ける。
DAV2_PRESETS = [
    {"id": "depth-anything/Depth-Anything-V2-Small-hf", "label": "DAv2 Small（速い・軽い）"},
    {"id": "depth-anything/Depth-Anything-V2-Base-hf", "label": "DAv2 Base"},
    {"id": "depth-anything/Depth-Anything-V2-Large-hf", "label": "DAv2 Large（高品質・重い）"},
]
DA3_PRESETS = [
    {"id": "depth-anything/DA3METRIC-LARGE", "label": "DA3 Metric Large（絶対距離・高品質）"},
    {"id": "depth-anything/DA3METRIC-BASE", "label": "DA3 Metric Base"},
    {"id": "depth-anything/DA3-LARGE", "label": "DA3 Large（相対）"},
]


def model_presets() -> list[dict]:
    """現在のバックエンドで選べる深度モデルのプリセット一覧。"""
    return DA3_PRESETS if active_backend() == "da3" else DAV2_PRESETS


def default_model() -> str:
    """現在のバックエンドの既定モデルID。"""
    if active_backend() == "da3":
        from . import depth_da3

        return depth_da3.DA3_MODEL_ID
    return DEFAULT_MODEL_ID


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
    """深度推定パイプラインを遅延初期化して返す（モデルIDごとにキャッシュ）。"""
    model_id = model_id or DEFAULT_MODEL_ID
    if model_id not in _pipes:
        from transformers import pipeline

        _pipes[model_id] = pipeline(
            "depth-estimation", model=model_id, device=_select_device()
        )
    return _pipes[model_id]


def estimate_depth(image: Image.Image, model: str | None = None) -> np.ndarray:
    """視差マップ (H, W) float32 を返す。値が大きいほどカメラに近い。

    active_backend() に応じて DA3 / DAv2 へ振り分ける。
    model でモデルIDを指定すると、そのモデルを使う（None なら既定）。
    """
    if active_backend() == "da3":
        from . import depth_da3

        return depth_da3.estimate_disparity(image, model_id=model)
    return _estimate_dav2(image, model_id=model or DEFAULT_MODEL_ID)


def _estimate_dav2(image: Image.Image, model_id: str = DEFAULT_MODEL_ID) -> np.ndarray:
    """Depth Anything V2 (transformers) で視差を推定。入力画像サイズへリサイズ。"""
    if image.mode != "RGB":
        image = image.convert("RGB")

    pipe = get_pipeline(model_id)
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
