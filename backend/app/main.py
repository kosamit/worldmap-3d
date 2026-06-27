"""WorldMap 3D バックエンド (FastAPI)。

エンドポイント:
- GET  /api/health                     稼働確認と推論デバイス
- GET  /api/config                     フロント用の公開設定 (Maps JS キー等)
- POST /api/reconstruct                画像アップロード → 3D 化
- POST /api/reconstruct/streetview     緯度経度 → Street View 取得 → 3D 化
- POST /api/reconstruct/multiview      周辺の複数地点 → DA3マルチビュー → 3D 化
- POST /api/reconstruct/route          地図で選んだ道沿いの点列 → 連結 3D 化
- GET  /api/scenes                     蓄積済みシーン一覧
- GET  /scenes/<id>/scene.glb          生成済み glb (静的配信)
"""

import datetime
import io
import json
import os
import threading
import time
import uuid

from dotenv import load_dotenv

load_dotenv()  # backend/.env から GOOGLE_MAPS_API_KEY 等を読み込む

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import depth, depth_da3, storage
from .depth import _select_device, active_backend, estimate_depth
from .reconstruct_da3 import build_multiview_pointcloud
from .reconstruct import (
    DEFAULT_FOV_DEG,
    reconstruct_mesh,
)
from .route import (
    MAX_ROUTE_POINTS,
    build_route_scene,
    fetch_route_panoramas,
    snap_route_points,
)
from .equirect import build_equirectangular
from .streetview import (
    fetch_streetview,
    fetch_streetview_panorama,
    gather_nearby_viewpoints,
    sample_viewpoint_images,
)
from .tour import MAX_TOUR_NODES, build_tour

MAX_PANORAMA_VIEWS = 16
# パノラマは視点数ぶん連結するため、1 視点あたりの解像度を下げて総頂点数を抑える。
PANORAMA_MAX_WIDTH = 256
# マルチビュー1回でDA3に入れる総画像枚数の上限（GPUメモリ予算）。
# 実測: 72枚(process_res=504)でピーク約12GB。16GB級GPUで安全な枠として 80。
MAX_MULTIVIEW_IMAGES = 80

# 3D化ジョブの進捗管理（プロセス内メモリ）。フロントがポーリングして進捗を表示する。
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
# フェーズ → 全体進捗(%)のレンジ割り当て。
_PHASE_RANGES = {
    "street_view": (0, 35),
    "depth": (35, 75),
    "mesh": (75, 95),
    "save": (95, 100),
}

app = FastAPI(title="WorldMap 3D Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

storage.DATA_DIR.mkdir(parents=True, exist_ok=True)
storage.TOURS_DIR.mkdir(parents=True, exist_ok=True)
storage.PANOS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/scenes", StaticFiles(directory=str(storage.DATA_DIR)), name="scenes")
app.mount("/tours", StaticFiles(directory=str(storage.TOURS_DIR)), name="tours")
app.mount("/panos", StaticFiles(directory=str(storage.PANOS_DIR)), name="panos")

# 3D品質 比較ギャラリー（verify/run_experiments.sh の出力）を配信（存在すれば）。
from pathlib import Path as _Path  # noqa: E402

_GALLERY_DIR = _Path(__file__).resolve().parents[2] / "verify"
if _GALLERY_DIR.exists():
    app.mount("/gallery", StaticFiles(directory=str(_GALLERY_DIR), html=True), name="gallery")


# 推論パイプライン(transformers)は非リエントラント。同時実行クラッシュを防ぐため直列化。
_infer_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_job() -> str:
    """進捗ジョブを作成して job_id を返す。古い完了済みジョブは間引く。"""
    jid = uuid.uuid4().hex[:12]
    with _jobs_lock:
        # 完了/失敗済みが溜まりすぎたら古い順に削除（メモリ肥大防止）。
        done = [k for k, v in _jobs.items() if v["status"] != "running"]
        if len(done) > 50:
            for k in done[: len(done) - 50]:
                _jobs.pop(k, None)
        _jobs[jid] = {
            "status": "running",  # running | done | error
            "phase": "queued",
            "step": 0,
            "total": 0,
            "percent": 0,
            "message": "開始しています ...",
            "result": None,
            "error": None,
        }
    return jid


def _update_job(jid: str, **fields) -> None:
    with _jobs_lock:
        job = _jobs.get(jid)
        if job is not None:
            job.update(fields)


def _job_progress(jid: str):
    """ジョブ worker から呼ぶ progress(phase, step, total, message)。"""

    def progress(phase: str, step: int, total: int, message: str) -> None:
        lo, hi = _PHASE_RANGES.get(phase, (0, 100))
        frac = (step / total) if total else 1.0
        pct = lo + (hi - lo) * max(0.0, min(1.0, frac))
        _update_job(
            jid,
            phase=phase,
            step=int(step),
            total=int(total),
            percent=int(round(pct)),
            message=message,
        )

    return progress


def _process(image: Image.Image, source: str, fov_deg: float, location=None) -> dict:
    with _infer_lock:
        disparity = estimate_depth(image)
        mesh, info = reconstruct_mesh(image, disparity, fov_deg=fov_deg)

    sid = storage.new_scene_id()
    meta = {
        "id": sid,
        "source": source,
        "created_at": _now_iso(),
        "location": location,
        **info,
    }
    storage.save_scene(mesh, meta)
    meta["glb_url"] = f"/scenes/{sid}/scene.glb"
    return meta


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": _select_device(),
        "backend": active_backend(),
    }


@app.get("/api/config")
def config():
    """フロントが地図を描画するための公開設定を返す。

    Maps JavaScript API キーはブラウザに露出する前提（HTTP リファラ制限で保護する）。
    """
    maps_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    return {
        "maps_api_key": maps_key,
        "has_maps_key": bool(maps_key),
        "max_route_points": MAX_ROUTE_POINTS,
        # 3D化（深度メッシュ）の可変パラメータ用情報
        "depth_backend": active_backend(),
        "depth_default_model": depth.default_model(),
        "model_presets": depth.model_presets(),
        # 高精度3D化（DA3 マルチビュー）。カメラポーズ対応モデルが必要。
        "multiview_available": active_backend() == "da3",
        "multiview_default_model": depth_da3.DA3_MULTIVIEW_MODEL_ID,
        "multiview_model_presets": [
            {"id": "depth-anything/DA3-SMALL", "label": "DA3 Small（最省メモリ・高速）"},
            {"id": "depth-anything/DA3-BASE", "label": "DA3 Base（medium相当・軽い）"},
            {"id": "depth-anything/DA3-LARGE", "label": "DA3 Large（推奨・カメラ対応）"},
            {"id": "depth-anything/DA3-GIANT", "label": "DA3 Giant（最高品質・重い）"},
        ],
        "multiview_defaults": {
            "max_views": 3,
            "heading_count": 8,
            "pitch_count": 3,
            "radius_m": 12.0,
            "fov": 90.0,
            "max_width": 256,
            "conf_percentile": 10.0,
            "ensure_percentile": 90.0,
            "far_clip_m": 0.0,
            "height_clip_m": 40.0,
            "edge_factor": 0.7,
            "discontinuity_ratio": 0.15,
            "cut_stretch": True,
            "front_discontinuity": 0.45,
            "ground_cap": False,
            "camera_height_m": 2.5,
            "tsdf_voxel": 0.12,
            "process_res": 504,
            "process_res_method": "upper_bound_resize",
            "use_ray_pose": True,
            "ref_view_strategy": "saddle_balanced",
            "drop_sky": True,
            "filter_black_bg": False,
            "filter_white_bg": False,
            "anchor_gps": True,
            "raw": False,
            "remove_objects": False,
            "remove_classes": "person",
            "inpaint": False,
        },
        "multiview_ref_view_strategies": [
            {"id": "saddle_balanced", "label": "saddle_balanced（推奨・バランス）"},
            {"id": "saddle_sim_range", "label": "saddle_sim_range（類似範囲）"},
            {"id": "first", "label": "first（先頭ビュー基準）"},
            {"id": "middle", "label": "middle（中央ビュー基準）"},
        ],
        "multiview_process_res_methods": [
            {"id": "upper_bound_resize", "label": "低解像（軽い・既定）"},
            {"id": "lower_bound_resize", "label": "高解像（精細・重い）"},
            {"id": "upper_bound_crop", "label": "クロップ"},
        ],
    }


@app.post("/api/reconstruct")
def reconstruct(file: UploadFile = File(...), fov: float = Form(DEFAULT_FOV_DEG)):
    # 同期 def: ブロッキングな推論をスレッドプールで実行し、イベントループを塞がない。
    try:
        data = file.file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"画像が不正です: {exc}")
    try:
        return _process(image, "upload", fov)
    except Exception as exc:  # noqa: BLE001 - 500 を HTTPException 化して CORS ヘッダを維持
        raise HTTPException(status_code=500, detail=f"3D生成に失敗しました: {exc}")


@app.post("/api/reconstruct/streetview")
def reconstruct_streetview(
    lat: float = Form(...),
    lng: float = Form(...),
    heading: float = Form(0.0),
    pitch: float = Form(0.0),
    fov: float = Form(90.0),
    api_key: str | None = Form(None),
):
    try:
        image, used_fov = fetch_streetview(
            lat, lng, heading, pitch, fov, api_key=api_key
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    location = {"lat": lat, "lng": lng, "heading": heading, "pitch": pitch}
    try:
        return _process(image, "streetview", used_fov, location=location)
    except Exception as exc:  # noqa: BLE001 - 500 を HTTPException 化して CORS ヘッダを維持
        raise HTTPException(status_code=500, detail=f"3D生成に失敗しました: {exc}")


@app.get("/api/reconstruct/progress/{job_id}")
def reconstruct_progress(job_id: str):
    """3D化ジョブの進捗を返す。status は running | done | error。"""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ジョブが見つかりません")
        return dict(job)


def _gpu_util_pct() -> int | None:
    """nvidia-smi から GPU 使用率(%)をベストエフォートで取得。取れなければ None。"""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1.5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


@app.get("/api/gpu")
def gpu_status():
    """GPU のメモリ使用量・使用率を返す。フロントのパネルが定期ポーリングして表示する。

    used/total はGPU全体の物理メモリ（他プロセス込み）。torch_* はこのプロセスの
    確保量。available=False のときは CUDA 無し or 取得失敗。
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        free, total = torch.cuda.mem_get_info()
        used = total - free
        mib = 1024 * 1024
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "used_mb": round(used / mib),
            "total_mb": round(total / mib),
            "free_mb": round(free / mib),
            "used_pct": round(used / total * 100, 1),
            "torch_allocated_mb": round(torch.cuda.memory_allocated() / mib),
            "torch_reserved_mb": round(torch.cuda.memory_reserved() / mib),
            "util_pct": _gpu_util_pct(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


@app.post("/api/gpu/free")
def gpu_free():
    """ロード済みモデル(DA3/深度/LaMa/YOLO/ESRGAN)を解放し CUDA メモリを空ける。

    パネルのボタンから手動で呼ぶ。次回の3D化では各モデルを再ロードする（初回は遅い）。
    """
    freed = []
    for name, fn in (
        ("DA3", lambda: depth_da3.unload()),
        ("depth", lambda: depth.unload()),
    ):
        try:
            fn()
            freed.append(name)
        except Exception:  # noqa: BLE001
            pass
    for name, mod in (("LaMa/YOLO", "segment"), ("ESRGAN", "enhance"), ("SemSeg", "semseg")):
        try:
            __import__(f"app.{mod}", fromlist=["unload"]).unload()
            freed.append(name)
        except Exception:  # noqa: BLE001
            pass
    _free_cuda()
    return {"freed": freed, "gpu": gpu_status()}


def _pitch_rows(pitch_count: int) -> list[float]:
    """上下(ピッチ)方向のサンプル角リストを返す。fov≈90 を前提に天地を覆う。

    1 → 水平のみ / 3 → 下・水平・上 / 5 → さらに細かく。これで全周（水平だけでなく
    足元〜頭上）を覆い、上下の抜けを埋める。
    """
    # 先頭は必ず水平(0)。先頭画像が基準カメラ(ext[0])になり、その向きで地面の
    # 水平＝重力方向が決まるため、ここを水平にしないとシーン全体が傾く。
    presets = {
        1: [0.0],
        2: [0.0, -40.0],
        3: [0.0, -45.0, 45.0],
        4: [0.0, -45.0, 45.0, -75.0],
        5: [0.0, -30.0, 30.0, -60.0, 60.0],
    }
    return presets.get(int(pitch_count), [0.0, -45.0, 45.0])


def _free_cuda() -> None:
    """CUDA メモリを解放（蓄積でGPUが飽和し、2回目以降が遅延/停止するのを防ぐ）。

    empty_cache だけだと Python 参照が残るテンソルが解放されないため、先に
    gc.collect() で回収してから空ける。
    """
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _is_oom(exc: Exception) -> bool:
    return (
        exc.__class__.__name__ == "OutOfMemoryError"
        or "out of memory" in str(exc).lower()
    )


def _fetch_streetview_retry(*args, attempts: int = 3, **kwargs):
    """fetch_streetview を数回リトライ（ネットワーク瞬断で全体が落ちるのを防ぐ）。"""
    last = None
    for k in range(attempts):
        try:
            return fetch_streetview(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.6 * (k + 1))
    raise last


class _Heartbeat:
    """長い推論(1回のブロッキング呼び出し)中も、経過時間に応じて進捗バーをじわじわ
    進めて「固まった/止まった」誤認を防ぐ。実際の細かな進捗は取れないため、想定所要
    時間 est_seconds に対する経過割合で 95% までクリープさせる（完了時に本処理が
    100%相当へ進める）。"""

    def __init__(self, progress, phase, base_msg, est_seconds: float):
        self._progress = progress
        self._phase = phase
        self._base = base_msg
        self._est = max(5.0, float(est_seconds))
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        start = time.monotonic()
        while not self._stop.wait(1.0):
            sec = time.monotonic() - start
            frac = min(0.95, sec / self._est)
            # step/total=1000 ぶんでフェーズ内の割合を渡す → バーがフェーズ内を進む。
            self._progress(self._phase, int(frac * 1000), 1000,
                           f"{self._base}（経過 {int(sec)}秒）")

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


def _run_multiview_job(jid, lat, lng, params, api_key):
    """DA3 マルチビューで整合メッシュを作るジョブ本体（別スレッド）。"""
    progress = _job_progress(jid)
    try:
        # 1. 近接視点を収集し各視点から透視画像をサンプル（フェーズ: street_view）
        progress("street_view", 0, 1, "周辺の Street View 地点を収集中 ...")
        # 取得グリッドと高精細化方式を手法ごとに決める。
        if params["method"] == "panorama":
            # フロントの高精細パノと「完全に同じ」高精細グリッド(8方位×3仰角＋上下, fov55/75)
            # で取得 → 同一キャッシュキーで生タイルをそのまま共有・流用。3D化は常にこの
            # 高精細(狭角=実解像度)で取得（ESRGAN不要・未生成ならここで取得）。
            from .equirect import _source_views
            view_list = _source_views(hi=True)
            enhance_mode = None
            grid_desc = f"{len(view_list)}方向(高精細パノ共有グリッド)"
        else:
            hc = params["heading_count"]
            headings = [i * 360.0 / hc for i in range(hc)]
            pitches = _pitch_rows(params["pitch_count"])  # 上下方向の段（天地を埋める）
            view_list = [(h, p, params["fov"]) for h in headings for p in pitches]
            enhance_mode = params["enhance_mode"] if params["enhance_input"] else None
            grid_desc = f"{hc}方向×{len(pitches)}段"
        per_vp = len(view_list)

        # 自動制限：総画像枚数(地点数×枚数/地点)が GPU 予算を超えないよう地点数を抑える。
        budget_views = max(1, MAX_MULTIVIEW_IMAGES // per_vp)
        eff_max_views = min(params["max_views"], budget_views)
        capped = eff_max_views < params["max_views"]
        if capped:
            progress("street_view", 0, 1,
                     f"自動制限: 1地点あたり{per_vp}枚なので地点数を "
                     f"{params['max_views']}→{eff_max_views} に抑制（上限{MAX_MULTIVIEW_IMAGES}枚）")

        viewpoints = gather_nearby_viewpoints(
            lat, lng,
            radius_m=params["radius_m"],
            max_views=eff_max_views,
            api_key=api_key,
        )
        if not viewpoints:
            raise ValueError("この付近に Street View が見つかりませんでした")
        total_imgs = len(viewpoints) * per_vp
        images = []
        view_index = []
        view_angles = []  # 各画像の (heading, pitch, fov)。コライダーのDA3不要姿勢に使う。
        for vi, vp in enumerate(viewpoints):
            for (h, p, vfov) in view_list:
                done = len(images)
                progress("street_view", done, total_imgs,
                         f"Street View 取得 {done + 1}/{total_imgs}（{len(viewpoints)}地点）"
                         + ("・高精細化" if enhance_mode else ""))
                image, _ = _fetch_streetview_retry(
                    vp["lat"], vp["lng"], heading=h, pitch=p,
                    fov=vfov, api_key=api_key, pano=vp.get("pano_id"),
                    # 高精細化はタイル取得時に実施＝フロントの高精細パノと共有キャッシュ。
                    enhance=enhance_mode,
                )
                images.append(image)
                view_index.append(vi)
                view_angles.append((h, p, vfov))
        progress("street_view", total_imgs, total_imgs,
                 f"{len(viewpoints)}地点×{grid_desc} を取得")

        # 1.8 高精細化は各タイル取得時に済んでいる。ESRGANのGPUを解放してからDA3推論。
        if enhance_mode:
            from . import enhance
            enhance.unload()
            _free_cuda()

        # 1.9 物体除去＋生成補完（LaMa）。推論前に画像から物体を消して穴を描き直す。
        #     こうすると「穴」ではなく自然な背景になり、その深度も推定される。
        if params["remove_objects"] and params["remove_classes"] and params["inpaint"]:
            import numpy as _np

            from . import segment
            progress("street_view", total_imgs, total_imgs, "物体除去＋生成補完（LaMa）中 ...")
            arr = _np.stack([_np.asarray(im.convert("RGB")) for im in images])
            masks0 = segment.removal_masks(arr, params["remove_classes"])
            if masks0.any():
                filled = segment.inpaint_masked(arr, masks0)
                images = [Image.fromarray(filled[i]) for i in range(len(filled))]
                progress("street_view", total_imgs, total_imgs,
                         f"生成補完 完了（{int(masks0.reshape(len(masks0),-1).any(1).sum())}枚を補修）")
            _free_cuda()  # YOLO/LaMa のGPUを解放してからDA3推論（メモリ競合で遅くなるのを防ぐ）

        # 2. カメラ姿勢（必要なら深度）。GPU推論は直列化。
        with _infer_lock:
            if params["method"] in ("colliders", "proxy"):
                # 意味コライダー/プロキシは深度不要。既知の取得角(heading/pitch/fov)から
                # カメラ姿勢を解析的に構成し DA3 をスキップする（論文の核心: LiDARも深度も
                # 使わずに歩ける意味3D。docs/new_paper_concept.md §4-5）。proxy は pred を
                # 使わず equirect 写真から直接プロキシを作る。
                progress("depth", 1, 1, "既知取得角からカメラ姿勢を構成（DA3不要）")
                from .colliders import build_analytic_prediction
                pred = build_analytic_prediction(images, view_angles)
            else:
                base = f"DA3 マルチビュー推論中（{total_imgs}枚・モデルロード含む）"
                progress("depth", 0, 1, base + " ...")
                # 想定所要: 1枚あたり約0.5秒＋初回モデルロード余裕。バーのクリープ用。
                est = 12.0 + 0.5 * total_imgs
                with _Heartbeat(progress, "depth", base, est):
                    pred = depth_da3.infer_multiview(
                        images,
                        model_id=params["depth_model"],
                        process_res=params["process_res"],
                        process_res_method=params["process_res_method"],
                        use_ray_pose=params["use_ray_pose"],
                        ref_view_strategy=params["ref_view_strategy"],
                    )
                progress("depth", 1, 1,
                         "深度＋カメラポーズ推定 完了"
                         + ("（レイベース）" if params["use_ray_pose"] else ""))

                # 2.5 物体除去（穴あけ）。inpaint=Trueのときは1.9で補完済みなのでスキップ。
                if params["remove_objects"] and params["remove_classes"] and not params["inpaint"]:
                    progress("depth", 1, 1, "物体除去（YOLO）中 ...")
                    from . import segment
                    masks = segment.removal_masks(
                        pred["processed_images"], params["remove_classes"])
                    if masks.any():
                        pred["depth"] = pred["depth"].copy()
                        pred["depth"][masks] = -1.0  # 非正=無効 → 全手法で除外（穴になる）
                        progress("depth", 1, 1,
                                 f"物体除去 完了（{int(masks.reshape(len(masks),-1).any(1).sum())}枚で検出）")

            # 3. 面の構築（フェーズ: mesh）。method で手法を切替。
            if params["method"] == "tsdf":
                # TSDF 融合（open3d）。重なり層を1枚の連続面へ。ソリッドな歩ける空間。
                from .reconstruct_experiments import build_tsdf_mesh
                with _Heartbeat(progress, "mesh", "TSDF融合中", 8.0 + 0.4 * total_imgs):
                    scene, info = build_tsdf_mesh(
                        pred, view_index, viewpoints, voxel=params["tsdf_voxel"],
                    )
            elif params["method"] == "poisson":
                # Poisson 面再構成（open3d）。穴を水密面で塞ぐ＝柱の裏なども補間で埋める。
                # 1回の長いブロッキング呼び出しなのでハートビートで「停止」誤認を防ぐ。
                from .reconstruct_experiments import build_poisson_mesh
                with _Heartbeat(progress, "mesh", "Poisson面再構成中", 40.0):
                    scene, info = build_poisson_mesh(pred, view_index, viewpoints)
            elif params["method"] == "panorama":
                # equirect RGBD ＋ 生成穴埋め（LaMa）→ 隙間のない球面メッシュ。
                # 視点が複数なら GPS アンカーで実位置に並べ、つなぎ目なく連続的に歩ける
                # シーンにする（build_multipano_scene）。1視点なら従来の単一球。
                from .panorama3d import build_multipano_scene
                # 引き伸ばしカット: ON=front_discontinuity でしきい値カット / OFF=巨大値で全面保持。
                fd = params["front_discontinuity"] if params["cut_stretch"] else 1e9
                with _Heartbeat(progress, "mesh", "パノラマ生成補完中", 30.0 + 8.0 * len(viewpoints)):
                    scene, info = build_multipano_scene(
                        pred, view_index, viewpoints, inpaint=True,
                        front_discontinuity=fd, back_discontinuity=fd,
                        ground_cap=params["ground_cap"],
                        camera_height_m=params["camera_height_m"],
                    )
            elif params["method"] == "primitive":
                # 構造プリミティブ化（平面＋箱/円柱）。ゲームのブロックアウト風オブジェクト。
                from .primitives import build_primitive_mesh
                with _Heartbeat(progress, "mesh", "プリミティブ化中", 20.0):
                    scene, info = build_primitive_mesh(pred, view_index, viewpoints)
            elif params["method"] == "colliders":
                # 意味コライダー層（戦略1・深度不要）: セグメンテーション＋既知カメラ高さで
                # 床/壁/物体のコライダー(メートル)を生成。docs/semantic_scene_colliders.md。
                from .colliders import build_collider_scene
                with _Heartbeat(progress, "mesh", "意味コライダー生成中", 10.0 + 4.0 * total_imgs):
                    scene, info = build_collider_scene(
                        pred, view_index, viewpoints,
                        camera_height_m=params["camera_height_m"], progress=progress)
            elif params["method"] == "gaussian":
                # DA3点群→3D Gaussian Splatting(.ply) 初期化（学習なし）。見た目の写実
                # 表現。docs/new_paper_concept.md の「見た目=3DGS」。GLBは点群フォールバック。
                from .gaussian import build_gaussian_scene
                with _Heartbeat(progress, "mesh", "3D Gaussian生成中", 10.0 + 0.3 * total_imgs):
                    scene, info = build_gaussian_scene(
                        pred, view_index, viewpoints, params, progress=progress)
            elif params["method"] == "proxy":
                # DA3-free 本線: equirect写真＋セグメント＋既知カメラ高さで、歩ける床平面＋壁を
                # 実写テクスチャで張り、見えない床は LaMa 補完。pred は使わない（深度ゼロ依存）。
                # docs/new_paper_concept.md の DA3-free 主張を見た目つきで実体化。
                from .proxy3d import build_proxy_scene
                with _Heartbeat(progress, "mesh", "歩けるプロキシ生成中（DA3不要）", 12.0 + 8.0 * len(viewpoints)):
                    scene, info = build_proxy_scene(
                        viewpoints, camera_height_m=params["camera_height_m"],
                        api_key=api_key, inpaint=params["inpaint"], progress=progress)
            else:
                # GPSアンカー配置で面を張る（既定）。
                scene, info = build_multiview_pointcloud(
                    pred, view_index, viewpoints,
                    max_width=params["max_width"],
                    conf_percentile=params["conf_percentile"],
                    ensure_percentile=params["ensure_percentile"],
                    drop_sky=params["drop_sky"],
                    filter_black_bg=params["filter_black_bg"],
                    filter_white_bg=params["filter_white_bg"],
                    anchor_gps=params["anchor_gps"],
                    far_clip_m=params["far_clip_m"],
                    height_clip_m=params["height_clip_m"],
                    edge_factor=params["edge_factor"],
                    discontinuity_ratio=params["discontinuity_ratio"],
                    raw=params["raw"],
                    mesh=True,
                    progress=progress,
                )

        # 4. 保存（フェーズ: save）
        progress("save", 0, 1, "glb を保存中 ...")
        sid = storage.new_scene_id()
        meta = {
            "id": sid,
            "source": "streetview_multiview",
            "created_at": _now_iso(),
            "location": {
                "lat": lat, "lng": lng,
                "viewpoints": [
                    {"lat": v["lat"], "lng": v["lng"]} for v in viewpoints
                ],
            },
            "depth_backend": active_backend(),
            "depth_model": params["depth_model"] or depth_da3.DA3_MULTIVIEW_MODEL_ID,
            "requested_views": int(params["max_views"]),
            "views_capped": bool(capped),
            "images_used": int(total_imgs),
            "cut_stretch": bool(params["cut_stretch"]),
            "front_discontinuity": float(params["front_discontinuity"]),
            "ground_cap": bool(params["ground_cap"]),
            "camera_height_m": float(params["camera_height_m"]),
            **info,
        }
        # 3DGS は .ply(INRIA保管) と .splat(drei配信) を scene ディレクトリへ別途保存
        # （GLBは点群フォールバック）。一時パスは meta.json に残さず、フロントが読める
        # .splat を splat_url として公開する。
        splat_tmp = meta.pop("_splat_tmp", None)
        drei_tmp = meta.pop("_splat_tmp_drei", None)
        if drei_tmp:
            meta["splat_url"] = f"/scenes/{sid}/scene.splat"
        storage.save_scene(scene, meta)
        if splat_tmp or drei_tmp:
            import shutil
            if splat_tmp:
                shutil.move(splat_tmp, str(storage.scene_dir(sid) / "scene.ply"))
            if drei_tmp:
                shutil.move(drei_tmp, str(storage.scene_dir(sid) / "scene.splat"))
        meta["glb_url"] = f"/scenes/{sid}/scene.glb"
        progress("save", 1, 1, "完了")
        cap_note = (
            f"（要求{params['max_views']}→自動制限{len(viewpoints)}）" if capped
            else (f"（要求{params['max_views']}, 周辺で{len(viewpoints)}見つかった）"
                  if len(viewpoints) < params["max_views"] else "")
        )
        _update_job(
            jid, status="done", percent=100, phase="done",
            message=f"高精度3D化 完了: {info.get('viewpoints')}地点{cap_note} / "
                    f"{total_imgs}枚 / {meta.get('vertex_count', '?')}頂点",
            result=meta,
        )
    except ValueError as exc:
        _update_job(jid, status="error", error=str(exc), message=f"失敗: {exc}")
    except Exception as exc:  # noqa: BLE001
        if _is_oom(exc):
            _update_job(
                jid, status="error", error=str(exc),
                message="GPUメモリ不足で停止しました。方向数・上下段数・地点数・"
                        "処理解像度のいずれかを下げて再実行してください。",
            )
        else:
            _update_job(jid, status="error", error=str(exc),
                        message=f"高精度3D化に失敗しました: {exc}")
    finally:
        # 成功・失敗いずれでも CUDA を解放（次回実行が巻き込まれないように）。
        _free_cuda()


@app.post("/api/enhance/view")
def enhance_view(
    lat: float = Form(...),
    lng: float = Form(...),
    heading: float = Form(0.0),
    pitch: float = Form(0.0),
    fov: float = Form(90.0),
    mode: str = Form("esrgan"),
    size: int = Form(640),
    api_key: str | None = Form(None),
):
    """今表示中のビュー(緯度経度＋向き)を再取得し超解像して PNG を返す。"""
    from . import enhance
    mode = mode if mode in ("light", "esrgan") else "esrgan"
    sz = max(256, min(640, int(size)))
    try:
        image, _ = fetch_streetview(
            lat, lng, heading=heading, pitch=pitch, fov=fov,
            size=f"{sz}x{sz}", api_key=api_key)
        out = enhance.enhance_images([image], mode=mode)[0]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"高解像度化に失敗: {exc}")
    finally:
        _free_cuda()
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/api/reconstruct/multiview")
def reconstruct_multiview(
    lat: float = Form(...),
    lng: float = Form(...),
    max_views: int = Form(3),
    heading_count: int = Form(8),
    radius_m: float = Form(12.0),
    pitch: float = Form(0.0),
    pitch_count: int = Form(3),
    fov: float = Form(90.0),
    max_width: int = Form(256),
    conf_percentile: float = Form(10.0),
    ensure_percentile: float = Form(90.0),
    far_clip_m: float = Form(0.0),
    height_clip_m: float = Form(40.0),
    edge_factor: float = Form(0.7),
    discontinuity_ratio: float = Form(0.15),
    cut_stretch: bool = Form(True),
    front_discontinuity: float = Form(0.45),
    ground_cap: bool = Form(False),
    camera_height_m: float = Form(2.5),
    process_res: int = Form(504),
    process_res_method: str = Form("upper_bound_resize"),
    use_ray_pose: bool = Form(True),
    ref_view_strategy: str = Form("saddle_balanced"),
    enhance_input: bool = Form(False),
    enhance_mode: str = Form("light"),
    enhance_upscale: float = Form(1.0),
    drop_sky: bool = Form(True),
    filter_black_bg: bool = Form(False),
    filter_white_bg: bool = Form(False),
    anchor_gps: bool = Form(True),
    raw: bool = Form(False),
    remove_objects: bool = Form(False),
    remove_classes: str = Form("person"),
    inpaint: bool = Form(False),
    method: str = Form("mesh"),
    tsdf_voxel: float = Form(0.12),
    depth_model: str | None = Form(None),
    api_key: str | None = Form(None),
):
    """周辺の複数 Street View 地点を集め、DA3 マルチビューで整合した歩ける点群を作る。

    ジョブを開始して即 {job_id} を返す。進捗は /api/reconstruct/progress/{job_id}。
    """
    rvs = ref_view_strategy if ref_view_strategy in depth_da3.REF_VIEW_STRATEGIES else "saddle_balanced"
    prm = process_res_method if process_res_method in depth_da3.PROCESS_RES_METHODS else "upper_bound_resize"
    params = {
        "max_views": max(1, min(12 if method == "proxy" else 8, max_views)),  # proxyは安価＝高密度可
        "heading_count": max(2, min(8, heading_count)),
        "radius_m": max(3.0, min(40.0, float(radius_m))),
        "pitch": float(pitch),
        "pitch_count": max(1, min(5, int(pitch_count))),
        "fov": max(60.0, min(120.0, float(fov))),
        "max_width": int(max(64, min(504, max_width))),
        "conf_percentile": max(0.0, min(95.0, float(conf_percentile))),
        "ensure_percentile": max(50.0, min(100.0, float(ensure_percentile))),
        "far_clip_m": max(0.0, min(500.0, float(far_clip_m))),    # 0=無効（奥まで表示）
        "height_clip_m": max(0.0, min(200.0, float(height_clip_m))),  # 0=無効
        "edge_factor": max(0.0, min(3.0, float(edge_factor))),    # スパイク除去(辺÷深度), 0=無効
        "discontinuity_ratio": max(0.01, min(1.0, float(discontinuity_ratio))),
        "cut_stretch": bool(cut_stretch),                          # パノラマ: 引き伸ばし三角をカット
        "front_discontinuity": max(0.05, min(2.0, float(front_discontinuity))),  # カット強度(小=強)
        "ground_cap": bool(ground_cap),                            # パノラマ: 床平面RANSACキャップ
        "camera_height_m": max(0.5, min(5.0, float(camera_height_m))),  # 標尺=カメラ高さ(実寸基準)
        "process_res": int(max(168, min(1008, process_res))),
        "process_res_method": prm,
        "use_ray_pose": bool(use_ray_pose),
        "ref_view_strategy": rvs,
        "enhance_input": bool(enhance_input),
        "enhance_mode": enhance_mode if enhance_mode in ("light", "esrgan") else "light",
        "enhance_upscale": max(1.0, min(4.0, float(enhance_upscale))),
        "drop_sky": bool(drop_sky),
        "filter_black_bg": bool(filter_black_bg),
        "filter_white_bg": bool(filter_white_bg),
        "anchor_gps": bool(anchor_gps),
        "raw": bool(raw),
        "remove_objects": bool(remove_objects),
        "remove_classes": [c.strip() for c in (remove_classes or "").split(",") if c.strip()],
        "inpaint": bool(inpaint),
        "method": method if method in ("tsdf", "poisson", "panorama", "primitive", "colliders", "gaussian", "proxy") else "mesh",
        "tsdf_voxel": max(0.04, min(0.4, float(tsdf_voxel))),
        "depth_model": (depth_model or "").strip() or None,
    }
    # パノラマは地点数=1で従来の単一球、2以上でつなぎ目なし連続シーン(build_multipano_scene)。
    # ユーザーの地点数設定をそのまま尊重する（以前は1に固定していた）。
    jid = _new_job()
    thread = threading.Thread(
        target=_run_multiview_job, args=(jid, lat, lng, params, api_key), daemon=True
    )
    thread.start()
    return {"job_id": jid}


@app.post("/api/reconstruct/route")
def reconstruct_route(
    points: str = Form(...),
    num_views: int = Form(6),
    pitch: float = Form(0.0),
    fov: float = Form(90.0),
    api_key: str | None = Form(None),
):
    """地図で選んだ道沿いの点列を、歩いてつながる 1 つの 3D 空間へ合成する。

    points は [{"lat":.., "lng":..}, ...] の JSON 文字列。各点を実在パノラマへ
    スナップ・重複除去し、最初の点を原点とした実距離オフセットで連結する。
    """
    try:
        raw = json.loads(points)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"points が不正な JSON です: {exc}")
    if not isinstance(raw, list) or not raw:
        raise HTTPException(status_code=400, detail="points は 1 点以上の配列が必要です")

    try:
        coords = [(float(p["lat"]), float(p["lng"])) for p in raw]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail=f"各点は lat/lng を持つ必要があります: {exc}"
        )
    coords = coords[:MAX_ROUTE_POINTS]
    num_views = max(2, min(MAX_PANORAMA_VIEWS, num_views))

    # ネットワーク処理（スナップ・画像取得）はロックの外で行う。
    try:
        snapped = snap_route_points(coords, api_key=api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not snapped:
        raise HTTPException(
            status_code=404,
            detail="選択した道沿いに Street View が見つかりませんでした",
        )
    try:
        panoramas = fetch_route_panoramas(
            snapped, num_views=num_views, pitch=pitch, fov=fov, api_key=api_key
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        with _infer_lock:
            scene, info = build_route_scene(
                panoramas, snapped, fov=fov, max_width=PANORAMA_MAX_WIDTH
            )
        sid = storage.new_scene_id()
        meta = {
            "id": sid,
            "source": "streetview_route",
            "created_at": _now_iso(),
            "location": {
                **info["origin"],
                "points": info["points"],
                "num_views": num_views,
            },
            **info,
        }
        storage.save_scene(scene, meta)
        meta["glb_url"] = f"/scenes/{sid}/scene.glb"
        return meta
    except Exception as exc:  # noqa: BLE001 - 500 を HTTPException 化して CORS ヘッダを維持
        raise HTTPException(status_code=500, detail=f"ルート3D生成に失敗しました: {exc}")


@app.post("/api/streetview/cube")
def streetview_pano(
    pano_id: str = Form(...),
    lat: float = Form(0.0),
    lng: float = Form(0.0),
    out_width: int = Form(2560),
    hi: bool = Form(False),
    api_key: str | None = Form(None),
):
    """1 パノラマ分の equirectangular(全天球)画像を返す（pano_id でキャッシュ）。

    タイルを再投影して全天球にする。hi=True で各タイルを ESRGAN 超解像し out_width も
    上げて高精細化する（3D化と同じ高精細タイルを共有・同じビューワーで見回せる）。
    """
    # hi は狭角(fov55)タイルの実解像度に見合う out_width で再投影（ESRGAN不要）。
    if hi:
        out_width = 4096
    else:
        out_width = max(1024, min(4096, out_width))
    out_dir = storage.pano_dir(pano_id)
    dir_name = out_dir.name
    fname = "equirect_hi.jpg" if hi else "equirect.jpg"
    equirect_path = out_dir / fname

    if not equirect_path.exists():
        try:
            image = build_equirectangular(
                lat, lng, api_key=api_key, pano=pano_id, out_width=out_width, hi=hi,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"パノラマ取得に失敗: {exc}")
        out_dir.mkdir(parents=True, exist_ok=True)
        image.save(str(equirect_path), format="JPEG", quality=90)

    return {
        "pano_id": pano_id,
        "equirect": f"/panos/{dir_name}/{fname}",
    }


@app.post("/api/route/tour")
def route_tour(
    points: str = Form(...),
    face_size: int = Form(640),
    api_key: str | None = Form(None),
):
    """地図で選んだ道沿いの点列を、写真パノラマで繋いだ歩けるツアーにする。

    深度メッシュ化はせず、各ノードのキューブ6面写真をそのまま使う。
    points は [{"lat":.., "lng":..}, ...] の JSON 文字列。
    """
    try:
        raw = json.loads(points)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"points が不正な JSON です: {exc}")
    if not isinstance(raw, list) or not raw:
        raise HTTPException(status_code=400, detail="points は 1 点以上の配列が必要です")
    try:
        coords = [(float(p["lat"]), float(p["lng"])) for p in raw]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail=f"各点は lat/lng を持つ必要があります: {exc}"
        )
    coords = coords[:MAX_TOUR_NODES]
    face_size = max(256, min(640, face_size))

    try:
        return build_tour(coords, api_key=api_key, face_size=face_size)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - 500 を HTTPException 化して CORS ヘッダを維持
        raise HTTPException(status_code=500, detail=f"ツアー生成に失敗しました: {exc}")


@app.get("/api/scenes")
def scenes():
    items = storage.list_scenes()
    items.sort(key=lambda m: m.get("created_at", ""))
    for m in items:
        m["glb_url"] = f"/scenes/{m['id']}/scene.glb"
    return {"scenes": items}
