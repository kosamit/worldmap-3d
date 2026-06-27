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
from .panorama3d import _equirect_dirs, _eq_layer, _inpaint_equirect
from .reconstruct_da3 import _noop, _viewpoint_enu

logger = logging.getLogger(__name__)

# ADE20K(屋内対応)セグメンテーション。Cityscapes(屋外運転)は屋内の遠い床を「壁」と誤判定し
# 床/壁接地が近傍に潰れる→壁が球になる。ADE20Kは floor/wall/ceiling を持ち、床が壁まで届く
# ＝接地距離が方位ごとに変わる＝実形状の奥行きが出る。
_ADE_MODEL_ID = "nvidia/segformer-b1-finetuned-ade-512-512"
_ADE_FLOOR = {3, 6, 11, 13, 28, 29, 53}   # floor, road, sidewalk, earth, rug, field, path
_ADE_SKY = {2}
_ade: dict = {}


def _ade_labels(eq_img):
    """equirect写真 → ADE20K ラベルマップ(H,W)。モデルはモジュールにキャッシュ。"""
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

    if not _ade:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _ade["dev"] = dev
        _ade["proc"] = SegformerImageProcessor.from_pretrained(_ADE_MODEL_ID)
        _ade["model"] = SegformerForSemanticSegmentation.from_pretrained(_ADE_MODEL_ID).to(dev).eval()
    W, H = eq_img.size
    with torch.no_grad():
        inp = _ade["proc"](images=eq_img, return_tensors="pt").to(_ade["dev"])
        logits = _ade["model"](**inp).logits
        up = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return up.argmax(1)[0].cpu().numpy()


def unload() -> None:
    """ADE20K モデルを解放（VRAM 返却）。"""
    _ade.clear()


def _build_proxy_mesh(eq_img, camera_height_m, inpaint, far_clip_m):
    """1枚の equirect 写真 → (verts, faces, rgba, stats)。既知高さで床平面＋壁＋テクスチャ。"""
    color = np.asarray(eq_img.convert("RGB"))
    H, W = color.shape[:2]

    lab = _ade_labels(eq_img)              # (H,W) ADE20K ラベル
    is_floor = np.isin(lab, list(_ADE_FLOOR))
    is_sky = np.isin(lab, list(_ADE_SKY))

    dirs = _equirect_dirs(H, W)            # (H,W,3) y上, lon0=+Z
    dy = dirs[..., 1]

    eps = 0.05
    down = dy < -eps
    r_geo = np.where(down, camera_height_m / (-np.minimum(dy, -eps)), 0.0)  # 床平面までの距離

    # --- 壁基準距離 D[col]: 各カラムで「見えている床(セグメント)」が届く最遠距離
    #     ＝床と壁の接地境界。ここに壁が立つ。既知高さ×接地点 が単眼の奥行き手がかり。
    #     これがカラムごとに変わる＝部屋の実形状＝歩くと視差が出る（球ではない）。
    seg_floor = down & is_floor
    D = np.zeros(W, np.float32)
    for w in range(W):
        col = r_geo[:, w][seg_floor[:, w]]
        D[w] = np.percentile(col, 90) if col.size >= 3 else 0.0
    good = D > 0
    if good.any():                       # 床が見えないカラムは最近傍の壁距離で補間
        idx = np.where(good, np.arange(W), 0)
        np.maximum.accumulate(idx, out=idx)
        D = D[idx]
        D[D == 0] = np.median(D[good])
    else:
        D[:] = min(8.0, far_clip_m)
    D = np.clip(D, 1.0, far_clip_m)
    Dcol = D[None, :]                     # (1,W)

    # --- 床: 既知平面を接地境界 D まで張る（境界より遠い下向き画素は壁が遮るので床にしない）。
    #     セグメント外でも幾何は既知なので床面にし、テクスチャ欠落は後で LaMa 補完。
    floor_mask = down & (r_geo <= Dcol)
    r_floor = np.where(floor_mask, r_geo, 0.0).astype(np.float32)
    floor_texture_hole = floor_mask & (~is_floor)

    # --- 壁/天井/物体: 接地境界より上の非空画素を D[col] に立てる（カラムごとに距離が違う）。
    #     天頂付近(急な上)は半径Dだと過剰に高くなるので落とす。
    wall_mask = (~floor_mask) & (~is_sky) & (dy < 0.5)
    r_wall = np.broadcast_to(Dcol, (H, W))

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
        "sky_dropped_px": int(is_sky.sum()),
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
