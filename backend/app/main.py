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

from . import depth, storage
from .depth import _select_device, active_backend, estimate_depth
from .panorama import build_panorama
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
from .streetview import fetch_streetview, fetch_streetview_panorama
from .tour import MAX_TOUR_NODES, build_tour

MAX_PANORAMA_VIEWS = 16
# パノラマは視点数ぶん連結するため、1 視点あたりの解像度を下げて総頂点数を抑える。
PANORAMA_MAX_WIDTH = 256

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
