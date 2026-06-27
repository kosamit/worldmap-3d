"""DA3-free 歩けるプロキシ3D（戦略の本筋）。

深度モデルを一切使わず、Street View の equirect 写真＋セマンティック・セグメンテーション
＋**既知カメラ高さ（測量基準）**だけで、歩ける・写実・補完つきの3Dを作る。

  床: 下向き光線 d は既知の床平面 y=-H に当たる → 半径 r=H/(-d_y)（メートル, DA3不要）。
  壁: 各経度カラムの床到達距離に立てた鉛直面（写真テクスチャがディテールを担う）。
  空: 落とす（面でない）。  穴(天底ブラー等): LaMa で補完。

得られる mesh は (1) 床/壁が実寸で当たり判定可＝ゲーム的コライダー、(2) 実写テクスチャ＝
写実、(3) 見えない床テクスチャを生成補完、の3点を単眼セグのみで満たす。複数視点は
GPS(ENU)で実位置に並べ、PanoFader で最近傍パノを見せてつなぎ目なく歩ける。

docs/new_paper_concept.md の DA3-free 本線。3DGS/DA3 点群路線の「雲・空洞」を回避する。
"""
from __future__ import annotations

import logging

import numpy as np
import trimesh

from .equirect import build_equirectangular
from . import semseg
from .panorama3d import _equirect_dirs, _eq_layer, _inpaint_equirect
from .reconstruct_da3 import _noop, _viewpoint_enu

logger = logging.getLogger(__name__)


def _build_proxy_mesh(eq_img, camera_height_m, inpaint, far_clip_m):
    """1枚の equirect 写真 → (verts, faces, rgba, stats)。既知高さで床平面＋壁＋テクスチャ。"""
    from PIL import Image

    color = np.asarray(eq_img.convert("RGB"))
    H, W = color.shape[:2]
    C = semseg.CATEGORIES
    FLOOR, SKY = C.index("floor"), C.index("sky")

    cat = semseg.segment_categories([eq_img])[0]
    cat = np.asarray(Image.fromarray(cat.astype(np.uint8)).resize((W, H), Image.NEAREST))

    dirs = _equirect_dirs(H, W)            # (H,W,3) y上, lon0=+Z
    dy = dirs[..., 1]

    # --- 床: 既知の地面平面 y=-H。下向き光線は全て床に当たる（幾何は既知）ので
    #     セグメントに依らず床面を張り、テクスチャ欠落(天底/空誤検出)は後で補完する。
    floor_mask = dy < -0.05
    r_floor = np.zeros((H, W), np.float32)
    r_floor[floor_mask] = np.clip(camera_height_m / (-dy[floor_mask]), 0.3, far_clip_m)
    floor_texture_hole = floor_mask & (cat != FLOOR)   # 床面だが写真が無効＝補完対象

    # --- 壁: 各カラムの床到達距離に鉛直面を立てる ---
    r_base = np.zeros(W, np.float32)
    for w in range(W):
        col = r_floor[:, w][floor_mask[:, w]]
        r_base[w] = np.percentile(col, 90) if col.size else 0.0
    good = r_base > 0
    if good.any():
        idx = np.where(good, np.arange(W), 0)
        np.maximum.accumulate(idx, out=idx)
        r_base = r_base[idx]
        r_base[r_base == 0] = np.median(r_base[r_base > 0])
    else:
        r_base[:] = min(8.0, far_clip_m)
    r_wall = np.broadcast_to(r_base[None, :], (H, W))
    wall_mask = (cat != SKY) & (~floor_mask) & (dy > -0.25)

    radius = np.where(floor_mask, r_floor, r_wall).astype(np.float32)
    valid = floor_mask | wall_mask

    # --- 補完: 床テクスチャ欠落を LaMa で埋める（幾何の既知半径は保持し色だけ採用）。
    color_filled = color
    lama = False
    if inpaint:
        color_valid = valid & (~floor_texture_hole)
        color_filled, _r, lama = _inpaint_equirect(color, radius, color_valid)

    verts, faces, rgba = _eq_layer(color_filled, radius, valid, scale=1.0, discontinuity=0.25)
    stats = {
        "floor_px": int(floor_mask.sum()),
        "wall_px": int(wall_mask.sum()),
        "sky_dropped_px": int((cat == SKY).sum()),
        "texture_holes_px": int(floor_texture_hole.sum()),
        "inpaint_lama": bool(lama),
    }
    return verts, faces, rgba, stats


def build_proxy_scene(viewpoints, camera_height_m=2.5, api_key=None, eq_w=1024,
                      inpaint=True, far_clip_m=40.0, progress=None):
    """DA3不要・歩けるプロキシシーン。各視点の equirect 写真からプロキシを作り
    GPS(ENU)で実位置に並べる。(trimesh.Scene, info) を返す。

    複数視点は PanoFader 用に geometry を pano_i と命名し info["panos"] に中心を載せる。
    """
    progress = progress or _noop
    scene = trimesh.Scene()
    panos = []
    n = len(viewpoints)
    total_floor = total_wall = total_holes = 0
    lama_any = False
    enu = _viewpoint_enu(viewpoints)                  # (V,3) east,north,0 メートル, 先頭=原点

    for i, vp in enumerate(viewpoints):
        progress("mesh", i, n, f"プロキシ生成（DA3不要・{i + 1}/{n}地点）: 写真取得→セグメント→床/壁→補完 ...")
        eq = build_equirectangular(
            vp["lat"], vp["lng"], api_key=api_key, pano=vp.get("pano_id"), out_width=eq_w
        )
        verts, faces, rgba, st = _build_proxy_mesh(eq, camera_height_m, inpaint, far_clip_m)

        east, north = float(enu[i, 0]), float(enu[i, 1])
        offset = np.array([east, 0.0, -north], np.float32)   # three.js x=東, z=-北, y=上
        verts = verts + offset

        m = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgba, process=False)
        scene.add_geometry(m, geom_name=f"pano_{i}")
        panos.append({"name": f"pano_{i}", "center": [east, 0.0, -north]})
        total_floor += st["floor_px"]; total_wall += st["wall_px"]
        total_holes += st["texture_holes_px"]; lama_any |= st["inpaint_lama"]

    info = {
        "representation": "proxy_textured",
        "method": "proxy",
        "depth_free": True,
        "panos": panos,
        "viewpoints": n,
        "vertex_count": int(sum(len(g.vertices) for g in scene.geometry.values())),
        "face_count": int(sum(len(g.faces) for g in scene.geometry.values())),
        "proxy_floor_px": total_floor,
        "proxy_wall_px": total_wall,
        "proxy_texture_holes_px": total_holes,
        "proxy_inpaint_lama": lama_any,
        "camera_height_m": float(camera_height_m),
    }
    return scene, info
