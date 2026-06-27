"""入力画像の高精細化（DA3 推論前の前処理）。

Street View の JPEG はブロックノイズ・ボケがあり、これがそのまま DA3 の
processed_images（＝メッシュの頂点色）に乗る。推論前にクリーンにする。

mode:
  "light"  : cv2 でノイズ除去＋アンシャープ（依存なし・軽量）。
  "esrgan" : Real-ESRGAN(x4) で超解像（spandrel 経由）。GPUを使うので処理後に解放。

注意: DA3 は内部で process_res に縮小するため、超解像の真価は process_res を上げて
(例 720〜1008) 初めて出る。process_res が低いと結局縮小されて差が出にくい。
"""
from __future__ import annotations

from pathlib import Path

import threading

import numpy as np

from .reconstruct_da3 import _noop

_ESRGAN_WEIGHTS = Path(__file__).resolve().parent.parent / "data/models/RealESRGAN_x4plus.pth"
_esrgan_model = None
_gpu_lock = threading.Lock()  # GPUモデルはスレッド安全でないので逐次化（equirectは並列取得）


def unload() -> None:
    """ESRGAN を VRAM から解放（公開）。タイル単位処理の後にまとめて呼ぶ。"""
    _unload_esrgan()


def enhance_one(im, mode="esrgan"):
    """単一 PIL.Image を高精細化して返す（モデルは保持＝逐次再利用、解放は unload()）。"""
    from PIL import Image

    rgb = np.asarray(im.convert("RGB"))
    if mode == "esrgan":
        with _gpu_lock:
            res = _esrgan_one(rgb)
    else:
        res = _light_one(rgb, True, True, 1)
    return Image.fromarray(res)


def _light_one(rgb, denoise, sharpen, upscale):
    import cv2

    out = rgb
    if denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)
    if upscale and upscale > 1:
        h, w = out.shape[:2]
        out = cv2.resize(out, (int(w * upscale), int(h * upscale)),
                         interpolation=cv2.INTER_LANCZOS4)
    if sharpen:
        blur = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = cv2.addWeighted(out, 1.5, blur, -0.5, 0)
    return out


def _load_esrgan():
    global _esrgan_model
    if _esrgan_model is None:
        import torch
        from spandrel import ModelLoader
        if not _ESRGAN_WEIGHTS.exists():
            raise FileNotFoundError(
                f"Real-ESRGAN 重みが見つかりません: {_ESRGAN_WEIGHTS}")
        m = ModelLoader().load_from_file(str(_ESRGAN_WEIGHTS))
        _esrgan_model = m.cuda().eval() if torch.cuda.is_available() else m.cpu().eval()
    return _esrgan_model


def _unload_esrgan():
    """ESRGAN を VRAM から解放（後続の DA3 推論と競合させない）。"""
    global _esrgan_model
    if _esrgan_model is not None:
        _esrgan_model = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _esrgan_one(rgb, max_side=1536):
    import cv2
    import torch

    m = _load_esrgan()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.from_numpy(rgb.copy()).float().div(255).permute(2, 0, 1).unsqueeze(0).to(dev)
    with torch.no_grad():
        o = m(t)
    o = o.clamp(0, 1).squeeze(0).permute(1, 2, 0).mul(255).byte().cpu().numpy()
    # x4 は巨大になるのでメモリ上限へ縮小（それでも SR の鮮明さは残る）。
    h, w = o.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        o = cv2.resize(o, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return o


def enhance_images(images, mode="light", denoise=True, sharpen=True, upscale=1,
                   max_side=1536, progress=None):
    """list[PIL.Image] → 高精細化した list[PIL.Image]。"""
    from PIL import Image

    progress = progress or _noop
    out = []
    n = len(images)
    try:
        for i, im in enumerate(images):
            progress("street_view", i, n, f"画像を高精細化({mode}) {i + 1}/{n} ...")
            rgb = np.asarray(im.convert("RGB"))
            if mode == "esrgan":
                res = _esrgan_one(rgb, max_side=max_side)
            else:
                res = _light_one(rgb, denoise, sharpen, upscale)
            out.append(Image.fromarray(res))
    finally:
        if mode == "esrgan":
            _unload_esrgan()  # DA3 推論前に必ず解放
    return out
