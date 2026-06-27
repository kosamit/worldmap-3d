"""入力画像の高精細化（DA3 推論前の前処理）。

Street View の JPEG はブロックノイズ・ボケがあり、これがそのまま DA3 の
processed_images（＝メッシュの頂点色）に乗る。推論前に「ノイズ除去＋アンシャープ」で
クリーンにし、必要なら超解像で拡大する。

注意: DA3 は内部で process_res（既定504）に縮小するため、process_res 以下への拡大は
最終出力に効かない。よって既定は拡大なし（upscale=1）でノイズ除去＋シャープのみ。
process_res を上げる場合のみ upscale>1 が効く。

依存追加なし（cv2 + numpy）。将来 Real-ESRGAN を入れるならここに分岐を足す。
"""
from __future__ import annotations

import numpy as np

from .reconstruct_da3 import _noop


def _enhance_one(rgb, denoise, sharpen, upscale):
    import cv2

    out = rgb
    if denoise:
        # 色ノイズ/JPEGブロックを軽く除去（強すぎると質感が溶けるので弱め）。
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)
    if upscale and upscale > 1:
        h, w = out.shape[:2]
        out = cv2.resize(out, (int(w * upscale), int(h * upscale)),
                         interpolation=cv2.INTER_LANCZOS4)
    if sharpen:
        # アンシャープマスク（エッジを立てて「きれいに」見せる）。
        blur = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = cv2.addWeighted(out, 1.5, blur, -0.5, 0)
    return out


def enhance_images(images, denoise=True, sharpen=True, upscale=1, progress=None):
    """list[PIL.Image] → 高精細化した list[PIL.Image]。

    upscale: 拡大倍率（DA3 の process_res を上げない限り 1 推奨）。
    """
    from PIL import Image

    progress = progress or _noop
    out = []
    n = len(images)
    for i, im in enumerate(images):
        progress("street_view", i, n, f"画像を高精細化 {i + 1}/{n} ...")
        rgb = np.asarray(im.convert("RGB"))
        out.append(Image.fromarray(_enhance_one(rgb, denoise, sharpen, upscale)))
    return out
