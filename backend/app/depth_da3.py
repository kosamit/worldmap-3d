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

# マルチビュー（カメラポーズ推定つき）用モデル。DA3METRIC/MONO は深度のみで
# ポーズを返さないため、cam_enc/cam_dec を持つ標準 DA3 を使う必要がある。
DA3_MULTIVIEW_MODEL_ID = os.environ.get(
    "DA3_MULTIVIEW_MODEL", "depth-anything/DA3-LARGE"
)

# 選択可能なマルチビューモデル（小さいほど省メモリ・高速・精度↓）。
# いずれもポーズ推定対応（DA3METRIC/MONO は深度のみで不可）。
DA3_MULTIVIEW_MODELS = [
    {"id": "depth-anything/DA3-SMALL", "label": "SMALL（最省メモリ・高速）"},
    {"id": "depth-anything/DA3-BASE", "label": "BASE（中・medium相当）"},
    {"id": "depth-anything/DA3-LARGE", "label": "LARGE（高精度・既定）"},
]

# モデルIDごとにロード済みモデルをキャッシュ（切替時に再ロードできるよう dict 化）。
_models: dict[str, object] = {}
_EPS = 1e-6


def _load(model_id: str | None = None):
    model_id = model_id or DA3_MODEL_ID
    if model_id not in _models:
        from depth_anything_3.api import DepthAnything3

        # 省メモリ: 別モデルに切り替えるときは旧モデルをVRAMから解放（Large→Small等を
        # 試すと両方VRAMに残るのを防ぐ）。同一モデル再利用時は解放されない。
        if _models:
            import gc

            import torch
            _models.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model = DepthAnything3.from_pretrained(model_id)
        _models[model_id] = model.to(device="cuda")
    return _models[model_id]


def unload() -> None:
    """ロード済み DA3 モデルを解放してGPUメモリを空ける（次回使用時に再ロード）。"""
    _models.clear()


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


# DA3 が選べる「参照ビュー選択」戦略。マルチビューでどの視点を基準座標にするか。
REF_VIEW_STRATEGIES = ("saddle_balanced", "saddle_sim_range", "first", "middle")
# 入力画像のリサイズ方式。high_res=lower_bound（短辺基準で大きめ）/ low_res=upper_bound（軽い）。
PROCESS_RES_METHODS = ("upper_bound_resize", "lower_bound_resize", "upper_bound_crop")


def infer_multiview(
    images: list[Image.Image],
    model_id: str | None = None,
    *,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
) -> dict:
    """複数画像を1回の推論にまとめ、視点間で整合した深度＋カメラポーズを返す。

    DA3 のマルチビュー機能（カメラデコーダ）を使う。返り値は dict:
      depth:            (N, H, W) float32  遠い=大（相対 or メートル）
      conf:             (N, H, W) float32  信頼度
      sky:              (N, H, W) bool または None  空（オブジェクト）判定マスク
      is_metric:        int  1ならメートル絶対値、0なら相対スケール
      extrinsics:       (N, 3, 4) または (N, 4, 4) float32  world->camera
      intrinsics:       (N, 3, 3) float32  処理解像度 (H, W) 基準
      processed_images: (N, H, W, 3) uint8 推論に使われた画像（深度と同解像度）

    パラメータ（DA3 のフルオプションを露出）:
      process_res:        処理解像度（大きいほど精細・重い）。
      process_res_method: リサイズ方式（PROCESS_RES_METHODS）。
      use_ray_pose:       True で「レイ（光線）ベースのポーズ推定」を使う。カメラ
                          デコーダの代わりに各画素レイから RANSAC でポーズ/内部
                          パラメータを解く。視点配置によってはこちらが安定する。
      ref_view_strategy:  基準ビューの選び方（REF_VIEW_STRATEGIES）。

    使うモデルは cam_enc/cam_dec を持つ必要がある（DA3METRIC/MONO は不可）。
    """
    model_id = model_id or DA3_MULTIVIEW_MODEL_ID
    model = _load(model_id)

    if ref_view_strategy not in REF_VIEW_STRATEGIES:
        ref_view_strategy = "saddle_balanced"
    if process_res_method not in PROCESS_RES_METHODS:
        process_res_method = "upper_bound_resize"

    arrays = [np.asarray(im.convert("RGB")) for im in images]
    prediction = model.inference(
        arrays,
        process_res=int(process_res),
        process_res_method=process_res_method,
        use_ray_pose=bool(use_ray_pose),
        ref_view_strategy=ref_view_strategy,
    )

    if prediction.extrinsics is None or prediction.intrinsics is None:
        raise ValueError(
            f"モデル {model_id} はカメラポーズを返しません"
            "（マルチビューには DA3-LARGE 等のカメラ対応モデルが必要）"
        )

    sky = getattr(prediction, "sky", None)
    return {
        "depth": np.asarray(prediction.depth, dtype=np.float32),
        "conf": (
            None if prediction.conf is None else np.asarray(prediction.conf, dtype=np.float32)
        ),
        "sky": (None if sky is None else np.asarray(sky, dtype=bool)),
        "is_metric": int(getattr(prediction, "is_metric", 0) or 0),
        "extrinsics": np.asarray(prediction.extrinsics, dtype=np.float32),
        "intrinsics": np.asarray(prediction.intrinsics, dtype=np.float32),
        "processed_images": np.asarray(prediction.processed_images, dtype=np.uint8),
    }
