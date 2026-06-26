"""WorldMap 3D バックエンド (FastAPI)。

エンドポイント:
- GET  /api/health                     稼働確認と推論デバイス
- GET  /api/config                     フロント用の公開設定 (Maps JS キー等)
- POST /api/reconstruct                画像アップロード → 3D 化
- POST /api/reconstruct/streetview     緯度経度 → Street View 取得 → 3D 化
- POST /api/reconstruct/panorama       1 地点を 360° 撮影 → 3D 化
- POST /api/reconstruct/route          地図で選んだ道沿いの点列 → 連結 3D 化
- GET  /api/scenes                     蓄積済みシーン一覧
- GET  /scenes/<id>/scene.glb          生成済み glb (静的配信)
"""

import datetime
import io
import json
import os
import threading
import uuid

from dotenv import load_dotenv

load_dotenv()  # backend/.env から GOOGLE_MAPS_API_KEY 等を読み込む

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import depth, depth_da3, storage
from .depth import _select_device, active_backend, estimate_depth
from .panorama import build_panorama
from .reconstruct_da3 import build_multiview_pointcloud
from .reconstruct import (
    DEFAULT_FOV_DEG,
    DISCONTINUITY_RATIO,
    FAR_M,
    NEAR_M,
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
    """build_panorama / worker から呼ぶ progress(phase, step, total, message)。"""

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
        "max_panorama_views": MAX_PANORAMA_VIEWS,
        "reconstruct_defaults": {
            "near": NEAR_M,
            "far": FAR_M,
            "discontinuity": DISCONTINUITY_RATIO,
            "max_width": PANORAMA_MAX_WIDTH,
            "num_views": 8,
        },
        # 高精度3D化（DA3 マルチビュー）。カメラポーズ対応モデルが必要。
        "multiview_available": active_backend() == "da3",
        "multiview_default_model": depth_da3.DA3_MULTIVIEW_MODEL_ID,
        "multiview_model_presets": [
            {"id": "depth-anything/DA3-LARGE", "label": "DA3 Large（推奨・カメラ対応）"},
            {"id": "depth-anything/DA3-BASE", "label": "DA3 Base（軽い）"},
            {"id": "depth-anything/DA3-GIANT", "label": "DA3 Giant（最高品質・重い）"},
        ],
        "multiview_defaults": {
            "max_views": 3,
            "heading_count": 8,
            "pitch_count": 3,
            "radius_m": 12.0,
            "fov": 90.0,
            "max_width": 256,
            "conf_percentile": 40.0,
            "ensure_percentile": 90.0,
            "far_clip_m": 0.0,
            "height_clip_m": 40.0,
            "process_res": 504,
            "process_res_method": "upper_bound_resize",
            "use_ray_pose": True,
            "ref_view_strategy": "saddle_balanced",
            "drop_sky": True,
            "filter_black_bg": False,
            "filter_white_bg": False,
            "anchor_gps": True,
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


def _run_panorama_job(jid: str, lat, lng, headings, pitch, fov, api_key, params):
    """別スレッドで実行する 3D化本体。進捗を _jobs[jid] に書き込む。"""
    progress = _job_progress(jid)
    num_views = len(headings)
    try:
        # 1. Street View 取得（フェーズ: street_view）
        images = []
        for i, heading in enumerate(headings):
            progress("street_view", i, num_views, f"Street View 取得 {i + 1}/{num_views}")
            image, _ = fetch_streetview(
                lat, lng, heading=heading, pitch=pitch, fov=fov, api_key=api_key
            )
            images.append((image, heading))
        progress("street_view", num_views, num_views, "Street View 取得 完了")

        # 2. 深度推定 + メッシュ合成（フェーズ: depth, mesh）。推論は直列化。
        with _infer_lock:
            scene, info = build_panorama(
                images,
                fov_deg=fov,
                max_width=params["max_width"],
                near_m=params["near"],
                far_m=params["far"],
                discontinuity_ratio=params["discontinuity"],
                depth_model=params["depth_model"],
                progress=progress,
            )

        # 3. 保存（フェーズ: save）
        progress("save", 0, 1, "glb を保存中 ...")
        sid = storage.new_scene_id()
        meta = {
            "id": sid,
            "source": "streetview_panorama",
            "created_at": _now_iso(),
            "location": {
                "lat": lat,
                "lng": lng,
                "pitch": pitch,
                "fov": fov,
                "num_views": num_views,
            },
            "depth_backend": active_backend(),
            **info,
        }
        storage.save_scene(scene, meta)
        meta["glb_url"] = f"/scenes/{sid}/scene.glb"
        progress("save", 1, 1, "完了")
        _update_job(
            jid,
            status="done",
            percent=100,
            phase="done",
            message=f"3D化完了: {meta.get('vertex_count', '?')} 頂点",
            result=meta,
        )
    except ValueError as exc:
        _update_job(jid, status="error", error=str(exc), message=f"失敗: {exc}")
    except Exception as exc:  # noqa: BLE001 - 例外はジョブ状態に記録
        _update_job(
            jid,
            status="error",
            error=str(exc),
            message=f"パノラマ3D生成に失敗しました: {exc}",
        )


def _clamp_reconstruct_params(near, far, discontinuity, max_width, depth_model) -> dict:
    """フロントから来た再構成パラメータを安全な範囲にクランプする。"""
    near = max(0.1, min(100.0, float(near)))
    far = max(near + 0.5, min(1000.0, float(far)))
    discontinuity = max(0.005, min(1.0, float(discontinuity)))
    max_width = int(max(64, min(1024, int(max_width))))
    depth_model = (depth_model or "").strip() or None
    return {
        "near": near,
        "far": far,
        "discontinuity": discontinuity,
        "max_width": max_width,
        "depth_model": depth_model,
    }


@app.post("/api/reconstruct/panorama")
def reconstruct_panorama(
    lat: float = Form(...),
    lng: float = Form(...),
    num_views: int = Form(8),
    pitch: float = Form(0.0),
    fov: float = Form(90.0),
    near: float = Form(NEAR_M),
    far: float = Form(FAR_M),
    discontinuity: float = Form(DISCONTINUITY_RATIO),
    max_width: int = Form(PANORAMA_MAX_WIDTH),
    depth_model: str | None = Form(None),
    api_key: str | None = Form(None),
):
    """同一地点を 360° 分割で撮影し、取り囲む 1 つの 3D 空間へ合成する。

    実処理はバックグラウンドのジョブで行い、即座に {job_id} を返す。フロントは
    GET /api/reconstruct/progress/{job_id} をポーリングして進捗・結果を受け取る。
    """
    num_views = max(2, min(MAX_PANORAMA_VIEWS, num_views))
    headings = [i * 360.0 / num_views for i in range(num_views)]
    params = _clamp_reconstruct_params(near, far, discontinuity, max_width, depth_model)

    jid = _new_job()
    thread = threading.Thread(
        target=_run_panorama_job,
        args=(jid, lat, lng, headings, pitch, fov, api_key, params),
        daemon=True,
    )
    thread.start()
    return {"job_id": jid}


@app.get("/api/reconstruct/progress/{job_id}")
def reconstruct_progress(job_id: str):
    """3D化ジョブの進捗を返す。status は running | done | error。"""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ジョブが見つかりません")
        return dict(job)


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


def _run_multiview_job(jid, lat, lng, params, api_key):
    """DA3 マルチビューで整合メッシュを作るジョブ本体（別スレッド）。"""
    progress = _job_progress(jid)
    try:
        # 1. 近接視点を収集し各視点から透視画像をサンプル（フェーズ: street_view）
        progress("street_view", 0, 1, "周辺の Street View 地点を収集中 ...")
        hc = params["heading_count"]
        headings = [i * 360.0 / hc for i in range(hc)]
        pitches = _pitch_rows(params["pitch_count"])  # 上下方向の段（天地を埋める）

        # 自動制限：総画像枚数(地点数×方向数×段数)が GPU 予算を超えないよう地点数を抑える。
        per_vp = hc * len(pitches)
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
        for vi, vp in enumerate(viewpoints):
            for h in headings:
                for p in pitches:
                    done = len(images)
                    progress("street_view", done, total_imgs,
                             f"Street View 取得 {done + 1}/{total_imgs}（{len(viewpoints)}地点）")
                    image, _ = fetch_streetview(
                        vp["lat"], vp["lng"], heading=h, pitch=p,
                        fov=params["fov"], api_key=api_key, pano=vp.get("pano_id"),
                    )
                    images.append(image)
                    view_index.append(vi)
        progress("street_view", total_imgs, total_imgs,
                 f"{len(viewpoints)}地点×{hc}方向×{len(pitches)}段 を取得")

        # 2. DA3 マルチビュー推論（フェーズ: depth）。推論は直列化。
        with _infer_lock:
            progress("depth", 0, 1,
                     f"DA3 マルチビュー推論中（{total_imgs}枚・モデルロード含む）...")
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

            # 3. 整合メッシュの構築（フェーズ: mesh）。GPSアンカー配置で面を張る。
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
            **info,
        }
        storage.save_scene(scene, meta)
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
        _update_job(jid, status="error", error=str(exc),
                    message=f"高精度3D化に失敗しました: {exc}")


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
    conf_percentile: float = Form(40.0),
    ensure_percentile: float = Form(90.0),
    far_clip_m: float = Form(0.0),
    height_clip_m: float = Form(40.0),
    process_res: int = Form(504),
    process_res_method: str = Form("upper_bound_resize"),
    use_ray_pose: bool = Form(True),
    ref_view_strategy: str = Form("saddle_balanced"),
    drop_sky: bool = Form(True),
    filter_black_bg: bool = Form(False),
    filter_white_bg: bool = Form(False),
    anchor_gps: bool = Form(True),
    depth_model: str | None = Form(None),
    api_key: str | None = Form(None),
):
    """周辺の複数 Street View 地点を集め、DA3 マルチビューで整合した歩ける点群を作る。

    ジョブを開始して即 {job_id} を返す。進捗は /api/reconstruct/progress/{job_id}。
    """
    rvs = ref_view_strategy if ref_view_strategy in depth_da3.REF_VIEW_STRATEGIES else "saddle_balanced"
    prm = process_res_method if process_res_method in depth_da3.PROCESS_RES_METHODS else "upper_bound_resize"
    params = {
        "max_views": max(1, min(8, max_views)),
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
        "process_res": int(max(168, min(1008, process_res))),
        "process_res_method": prm,
        "use_ray_pose": bool(use_ray_pose),
        "ref_view_strategy": rvs,
        "drop_sky": bool(drop_sky),
        "filter_black_bg": bool(filter_black_bg),
        "filter_white_bg": bool(filter_white_bg),
        "anchor_gps": bool(anchor_gps),
        "depth_model": (depth_model or "").strip() or None,
    }
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
    api_key: str | None = Form(None),
):
    """1 パノラマ分の equirectangular(全天球)画像を返す（pano_id でキャッシュ）。

    多数の狭角タイルを再投影して高精細にする。フロントは隣接リンクを辿りながら
    進む先のパノラマだけを都度取得する。
    """
    out_width = max(1024, min(4096, out_width))
    out_dir = storage.pano_dir(pano_id)
    dir_name = out_dir.name
    equirect_path = out_dir / "equirect.jpg"

    if not equirect_path.exists():
        try:
            image = build_equirectangular(
                lat, lng, api_key=api_key, pano=pano_id, out_width=out_width
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"パノラマ取得に失敗: {exc}")
        out_dir.mkdir(parents=True, exist_ok=True)
        image.save(str(equirect_path), format="JPEG", quality=85)

    return {
        "pano_id": pano_id,
        "equirect": f"/panos/{dir_name}/equirect.jpg",
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
