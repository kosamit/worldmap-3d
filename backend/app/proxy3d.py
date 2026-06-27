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


def _detect_object_spans(D, near_max=8.0, ratio=0.62, min_w=4, max_w_frac=0.16):
    """D[col] の谷＝自立した近接物体(柱など)のカラム範囲 [(c0,c1,D_obj),...] を返す。

    広域背景距離(大窓中央値)より十分近く（ratio倍未満）かつ近距離(near_max未満)で、
    幅が現実的な連続カラム区間を物体とみなす。経度シームは無視（稀）。
    """
    W = len(D)
    k = max(8, W // 8)
    Dpad = np.concatenate([D[-k:], D, D[:k]])
    bg = np.array([np.median(Dpad[i:i + 2 * k + 1]) for i in range(W)])  # 広域背景距離
    is_obj = (D < ratio * bg) & (D < near_max)
    spans, c = [], 0
    while c < W:
        if is_obj[c]:
            j = c
            while j < W and is_obj[j]:
                j += 1
            if min_w <= (j - c) <= max_w_frac * W:
                spans.append((c, j, float(np.median(D[c:j]))))
            c = j
        else:
            c += 1
    return spans


def _object_quad(color, c0, c1, D_obj, obj_rows_mask):
    """物体カラム範囲を、原点に正対する平面の実写テクスチャ板にする。(v,f,rgba) or None。

    放射状の弧をやめ「平面」にすることで、横移動時の引き伸ばし(弧の歪み)を抑える。
    平面は中心方位 hc に水平距離 D_obj、横は right、縦は仰角→ D_obj*tan(el)。
    """
    H, W = color.shape[:2]
    rows = np.where(obj_rows_mask[:, c0:c1].any(axis=1))[0]
    cols = np.arange(c0, c1)
    if rows.size < 2 or cols.size < 2:
        return None
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    sub = color[r0:r1, c0:c1]
    h, w = sub.shape[:2]
    lon = (np.arange(W) / W - 0.5) * 2 * np.pi
    lat = (0.5 - np.arange(H) / H) * np.pi
    lonc = lon[(c0 + c1) // 2]
    hc = np.array([np.sin(lonc), 0.0, np.cos(lonc)], np.float64)
    right = np.array([np.cos(lonc), 0.0, -np.sin(lonc)], np.float64)
    u = (lon[c0:c1] - lonc) * D_obj            # (w,) 横位置(弦)
    yy = D_obj * np.tan(lat[r0:r1])            # (h,) 平面上の高さ
    P = (D_obj * hc)[None, None, :] + u[None, :, None] * right[None, None, :]
    P = np.broadcast_to(P, (h, w, 3)).copy()
    P[:, :, 1] += yy[:, None]
    verts = P.reshape(-1, 3).astype(np.float32)
    rgba = np.concatenate([sub.reshape(-1, 3), np.full((h * w, 1), 255, np.uint8)], 1)
    idx = np.arange(h * w).reshape(h, w)
    tl, tr = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    bl, br = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.stack([tl, bl, tr], 1), np.stack([tr, bl, br], 1)], 0)
    return verts, faces, rgba


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

    # --- 自立した近接物体(柱など)を平面テクスチャ板として切り出し、シェルからは除去する。
    #     放射状シェルは1方向1距離＝厚みを持てず横移動で弧が伸びる。物体だけ別の平面板に
    #     すれば伸びが減る（厚み付けの第一段。将来は箱化）。
    objects = []
    for c0, c1, D_obj in _detect_object_spans(D):
        quad = _object_quad(color, c0, c1, D_obj, wall_mask)
        if quad is not None:
            objects.append(quad)
            wall_mask[:, c0:c1] = False        # 物体カラムの壁はシェルから除去（板が担当）

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
        "objects": len(objects),
    }
    return verts, faces, rgba, objects, stats


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
    total_floor = total_wall = total_holes = total_obj = 0
    lama_any = False
    enu = _viewpoint_enu(viewpoints)                  # (V,3) east,north,0 メートル, 先頭=原点

    for i, vp in enumerate(viewpoints):
        progress("mesh", i, n, f"プロキシ生成（DA3不要・{i + 1}/{n}地点）: 写真取得→セグメント→床/壁→補完 ...")
        eq = build_equirectangular(
            vp["lat"], vp["lng"], api_key=api_key, pano=vp.get("pano_id"), out_width=eq_w
        )
        verts, faces, rgba, objects, st = _build_proxy_mesh(eq, camera_height_m, inpaint, far_clip_m)

        east, north = float(enu[i, 0]), float(enu[i, 1])
        offset = np.array([east, 0.0, -north], np.float32)   # three.js x=東, z=-北, y=上
        verts = verts + offset

        m = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgba, process=False)
        scene.add_geometry(m, geom_name=f"pano_{i}")
        # 切り出した物体板（柱など）も同じ pano に属させ PanoFader でフェード共有。
        for k, (ov, of, oc) in enumerate(objects):
            om = trimesh.Trimesh(vertices=ov + offset, faces=of, vertex_colors=oc, process=False)
            scene.add_geometry(om, geom_name=f"pano_{i}_obj_{k}")
        panos.append({"name": f"pano_{i}", "center": [east, 0.0, -north]})
        total_floor += st["floor_px"]; total_wall += st["wall_px"]
        total_holes += st["texture_holes_px"]; lama_any |= st["inpaint_lama"]
        total_obj += st["objects"]

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
        "proxy_objects": total_obj,
        "proxy_inpaint_lama": lama_any,
        "camera_height_m": float(camera_height_m),
    }
    return scene, info
