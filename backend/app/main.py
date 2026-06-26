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

from dotenv import load_dotenv

load_dotenv()  # backend/.env から GOOGLE_MAPS_API_KEY 等を読み込む

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import storage
from .depth import _select_device, active_backend, estimate_depth
from .panorama import build_panorama
from .reconstruct import DEFAULT_FOV_DEG, reconstruct_mesh
from .route import (
    MAX_ROUTE_POINTS,
    build_route_scene,
    fetch_route_panoramas,
    snap_route_points,
)
from .streetview import fetch_streetview, fetch_streetview_panorama

MAX_PANORAMA_VIEWS = 16
# パノラマは視点数ぶん連結するため、1 視点あたりの解像度を下げて総頂点数を抑える。
PANORAMA_MAX_WIDTH = 256

app = FastAPI(title="WorldMap 3D Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

storage.DATA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/scenes", StaticFiles(directory=str(storage.DATA_DIR)), name="scenes")


# 推論パイプライン(transformers)は非リエントラント。同時実行クラッシュを防ぐため直列化。
_infer_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


@app.post("/api/reconstruct/panorama")
def reconstruct_panorama(
    lat: float = Form(...),
    lng: float = Form(...),
    num_views: int = Form(8),
    pitch: float = Form(0.0),
    fov: float = Form(90.0),
    api_key: str | None = Form(None),
):
    """同一地点を 360° 分割で撮影し、取り囲む 1 つの 3D 空間へ合成する。"""
    num_views = max(2, min(MAX_PANORAMA_VIEWS, num_views))
    headings = [i * 360.0 / num_views for i in range(num_views)]
    try:
        images = fetch_streetview_panorama(
            lat, lng, headings, pitch=pitch, fov=fov, api_key=api_key
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        with _infer_lock:
            scene, info = build_panorama(
                images, fov_deg=fov, max_width=PANORAMA_MAX_WIDTH
            )
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
            **info,
        }
        storage.save_scene(scene, meta)
        meta["glb_url"] = f"/scenes/{sid}/scene.glb"
        return meta
    except Exception as exc:  # noqa: BLE001 - 500 を HTTPException 化して CORS ヘッダを維持
        raise HTTPException(status_code=500, detail=f"パノラマ3D生成に失敗しました: {exc}")


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


@app.get("/api/scenes")
def scenes():
    items = storage.list_scenes()
    items.sort(key=lambda m: m.get("created_at", ""))
    for m in items:
        m["glb_url"] = f"/scenes/{m['id']}/scene.glb"
    return {"scenes": items}
