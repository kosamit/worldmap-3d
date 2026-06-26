"""DA3 マルチビュー予測（深度＋カメラポーズ）から、視点間で整合した歩ける
メッシュ(glb)を構築する。

単眼1枚ずつを独立に正規化する従来手法と違い、DA3 が推定した共通スケールの深度と
カメラ外部/内部パラメータを使って各ビューを同一ワールドへ逆投影する。これにより
継ぎ目のない一貫した3D空間が得られる。

座標規約は DA3 の glb エクスポートに合わせる:
- ピクセル逆投影: Xc = depth * K^-1 @ [u,v,1]（OpenCV: x右/y下/z前）
- ワールド化:     Xw = (w2c)^-1 @ [Xc;1]
- glTF 整列:      最初のカメラ基準で Y,Z を反転し、点群中央を原点へ（Three.js は Y-up）

相対スケールの深度なので、実在パノラマ視点間の実距離（haversine）と DA3 が推定した
カメラ間距離の比から、メートルスケールを復元する。
"""

import math

import numpy as np
import trimesh

from .streetview import haversine_m


def _noop(*_args, **_kwargs):
    pass


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """相似変換 (s, R, t) を最小二乗で解く。dst ≈ s * R @ src + t。

    src, dst: (M,3)。点数が少ない/退化しているときは恒等に近い安全値を返す。
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    m = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = (dc.T @ sc) / m
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1.0
    R = u @ s_mat @ vt
    var_s = (sc ** 2).sum() / m
    scale = float((d * np.diag(s_mat)).sum() / var_s) if var_s > 1e-12 else 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def _umeyama_2d(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """水平面(2D)の相似変換 (s, R(2x2), t) を解く。dst ≈ s * R @ src + t。

    視点中心は地面上でほぼ同一平面に乗るため、鉛直まわりの「向き(ヨー)」と
    スケールだけを 2D で安定に解く。鉛直方向は別途 DA3 の重力で固定する。
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    m = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = (dc.T @ sc) / m
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[1, 1] = -1.0
    R = u @ s_mat @ vt
    var_s = (sc ** 2).sum() / m
    scale = float((d * np.diag(s_mat)).sum() / var_s) if var_s > 1e-9 else 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def _viewpoint_enu(viewpoints: list[dict]) -> np.ndarray:
    """各視点を、先頭視点を原点とする局所 ENU メートル座標 (V,3) で返す。

    x=東(east), y=北(north), z=上(0)。緯度経度の微小差を平面近似でメートル化する。
    """
    lat0 = float(viewpoints[0]["lat"])
    lng0 = float(viewpoints[0]["lng"])
    coslat = math.cos(math.radians(lat0))
    out = np.zeros((len(viewpoints), 3), dtype=np.float64)
    for i, vp in enumerate(viewpoints):
        out[i, 0] = (float(vp["lng"]) - lng0) * 111320.0 * coslat  # east
        out[i, 1] = (float(vp["lat"]) - lat0) * 111320.0          # north
    return out


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    if ext.shape == (4, 4):
        return ext.astype(np.float64)
    if ext.shape == (3, 4):
        h = np.eye(4, dtype=np.float64)
        h[:3, :4] = ext
        return h
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")


def _alignment_transform(ext_w2c0: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    """DA3 と同じ glTF 整列行列 A を返す（X' = A @ [X;1]）。

    最初のカメラ向きへ整列 → CV(x右/y下/z前) から glTF(x右/y上/z後) へ Y,Z 反転 →
    点群の中央値を原点へ平行移動。
    """
    w2c0 = _as_homogeneous44(ext_w2c0)
    m = np.eye(4, dtype=np.float64)
    m[1, 1] = -1.0
    m[2, 2] = -1.0
    a_no_center = m @ w2c0
    if points_world.shape[0] > 0:
        pts = trimesh.transform_points(points_world, a_no_center)
        center = np.median(pts, axis=0)
    else:
        center = np.zeros(3, dtype=np.float64)
    t = np.eye(4, dtype=np.float64)
    t[:3, 3] = -center
    return t @ a_no_center


def _metric_scale(cam_centers: np.ndarray, view_index, viewpoints) -> float | None:
    """視点間の実距離 / DA3推定距離 の中央値からメートル/単位 のスケールを推定。

    視点(パノラマ)が2地点以上ないとベースラインが無く推定不能 → None。
    """
    groups: dict[int, list] = {}
    for c, vi in zip(cam_centers, view_index):
        groups.setdefault(int(vi), []).append(c)
    centers = {vi: np.mean(np.array(v), axis=0) for vi, v in groups.items()}
    vis = sorted(centers)
    ratios = []
    for ia in range(len(vis)):
        for ib in range(ia + 1, len(vis)):
            va, vb = vis[ia], vis[ib]
            da3 = float(np.linalg.norm(centers[va] - centers[vb]))
            real = haversine_m(
                viewpoints[va]["lat"], viewpoints[va]["lng"],
                viewpoints[vb]["lat"], viewpoints[vb]["lng"],
            )
            if da3 > 1e-6 and real > 0.5:
                ratios.append(real / da3)
    if ratios:
        return float(np.median(ratios))
    return None


def build_multiview_mesh(
    prediction: dict,
    view_index: list[int],
    viewpoints: list[dict],
    max_width: int = 224,
    discontinuity_ratio: float = 0.06,
    conf_percentile: float = 40.0,
    far_clip_m: float = 60.0,
    progress=None,
):
    """DA3 マルチビュー予測から整合メッシュ(trimesh)と info を返す。

    prediction: depth_da3.infer_multiview の返り値
    view_index: 各画像がどの視点(viewpoints のインデックス)由来か
    viewpoints: [{"lat","lng",...}]（メートルスケール復元に使用）
    """
    progress = progress or _noop
    depth = prediction["depth"]  # (N,H,W) 遠い=大
    conf = prediction.get("conf")  # (N,H,W) or None
    sky = prediction.get("sky")  # (N,H,W) bool or None
    K = prediction["intrinsics"].astype(np.float64)  # (N,3,3)
    ext = prediction["extrinsics"]  # (N,3,4)/(N,4,4) w2c
    images = prediction["processed_images"]  # (N,H,W,3) uint8
    n, h, w = depth.shape

    step = max(1, int(round(w / max(64, max_width))))
    us = np.arange(0, w, step)
    vs = np.arange(0, h, step)
    uu, vv = np.meshgrid(us, vs)  # (h2,w2)
    h2, w2 = uu.shape
    pix_u = uu.astype(np.float64)
    pix_v = vv.astype(np.float64)

    # 信頼度しきい値（全ビュー共通のパーセンタイル）
    conf_thr = None
    if conf is not None:
        conf_thr = float(np.percentile(conf, conf_percentile))

    cam_centers = np.zeros((n, 3), dtype=np.float64)
    all_verts = []
    all_faces = []
    all_colors = []
    voff = 0

    for i in range(n):
        progress("mesh", i, n, f"整合メッシュ生成 {i + 1}/{n}")
        c2w = np.linalg.inv(_as_homogeneous44(ext[i]))  # (4,4)
        cam_centers[i] = c2w[:3, 3]

        z = depth[i][np.ix_(vs, us)].astype(np.float64)  # (h2,w2)
        col = images[i][np.ix_(vs, us)].reshape(-1, 3)  # (h2*w2,3)
        valid = np.isfinite(z) & (z > 0)
        if conf_thr is not None:
            valid &= conf[i][np.ix_(vs, us)] >= conf_thr
        if sky is not None:
            valid &= ~sky[i][np.ix_(vs, us)]

        fx, fy = K[i][0, 0], K[i][1, 1]
        cx, cy = K[i][0, 2], K[i][1, 2]
        xc = (pix_u - cx) / fx * z
        yc = (pix_v - cy) / fy * z
        cam = np.stack([xc, yc, z], axis=-1).reshape(-1, 3)  # (h2*w2,3)
        cam_h = np.concatenate([cam, np.ones((cam.shape[0], 1))], axis=1)
        world = (c2w @ cam_h.T)[:3].T.astype(np.float32)  # (h2*w2,3)

        # グリッド三角形化（無効頂点 or 深度不連続の面は分断）
        idx = np.arange(h2 * w2).reshape(h2, w2)
        tl = idx[:-1, :-1].ravel()
        tr = idx[:-1, 1:].ravel()
        bl = idx[1:, :-1].ravel()
        br = idx[1:, 1:].ravel()
        zf = z.ravel()
        vflat = valid.ravel()
        quad_valid = vflat[tl] & vflat[tr] & vflat[bl] & vflat[br]
        zq = np.stack([zf[tl], zf[tr], zf[bl], zf[br]], axis=1)
        zmax = zq.max(axis=1)
        zmin = zq.min(axis=1)
        zmean = zq.mean(axis=1) + 1e-9
        # 相対深度なので「平均深度に対する比」で不連続を判定（スケール非依存）
        cont = (zmax - zmin) / zmean < discontinuity_ratio * 4.0
        keep = quad_valid & cont

        tri1 = np.stack([tl, bl, tr], axis=1)[keep]
        tri2 = np.stack([tr, bl, br], axis=1)[keep]
        faces = np.concatenate([tri1, tri2], axis=0) + voff

        all_verts.append(world)
        all_colors.append(col.astype(np.uint8))
        all_faces.append(faces)
        voff += world.shape[0]

    progress("mesh", n, n, "整合メッシュ生成 完了")
    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0) if all_faces else np.zeros((0, 3), int)
    colors = np.concatenate(all_colors, axis=0)

    # glTF 整列（Three.js は Y-up）。点群中央を原点へ。
    a = _alignment_transform(ext[0], verts.astype(np.float64))
    verts = trimesh.transform_points(verts.astype(np.float64), a).astype(np.float32)

    # メートルスケール復元（視点が2地点以上あるとき）。
    scale = _metric_scale(cam_centers, view_index, viewpoints)
    if scale is None:
        # フォールバック: シーン半径を約15mに正規化。
        radius = float(np.percentile(np.linalg.norm(verts, axis=1), 95)) or 1.0
        scale = 15.0 / radius
    verts *= scale

    # 遠すぎる点（空など）に触れる面を分断: 中心からの距離で間引く。
    if far_clip_m and far_clip_m > 0:
        dist = np.linalg.norm(verts, axis=1)
        far_v = dist > far_clip_m
        if far_v.any() and faces.shape[0] > 0:
            face_far = far_v[faces].any(axis=1)
            faces = faces[~face_far]

    alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
    vcolors = np.concatenate([colors, alpha], axis=1)
    mesh = trimesh.Trimesh(
        vertices=verts, faces=faces, vertex_colors=vcolors, process=False
    )
    mesh.remove_unreferenced_vertices()

    info = {
        "views": int(n),
        "viewpoints": int(len(viewpoints)),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "metric_scale": float(scale),
        "max_width": int(max_width),
        "discontinuity_ratio": float(discontinuity_ratio),
        "conf_percentile": float(conf_percentile),
        "far_clip_m": float(far_clip_m),
    }
    return mesh, info


def _adaptive_conf_thresh(
    conf: np.ndarray,
    sky: np.ndarray | None,
    conf_percentile: float,
    ensure_percentile: float,
    base: float,
) -> float:
    """DA3 公式 get_conf_thresh と同じ適応しきい値を返す。

    空（sky）を除いた信頼度分布の下位パーセンタイルを下限、上位パーセンタイルを
    上限としてベース値をクランプする。これにより「信頼度の低い溶けた点」を確実に
    切りつつ、切りすぎて何も残らない事態を防ぐ。
    """
    if sky is not None and (~sky).sum() > 10:
        pixels = conf[~sky]
    else:
        pixels = conf
    lower = float(np.percentile(pixels, conf_percentile))
    upper = float(np.percentile(pixels, ensure_percentile))
    return float(min(max(base, lower), upper))


def build_multiview_pointcloud(
    prediction: dict,
    view_index: list[int],
    viewpoints: list[dict],
    max_width: int = 400,
    conf_percentile: float = 40.0,
    ensure_percentile: float = 90.0,
    conf_thresh_base: float = 1.05,
    drop_sky: bool = True,
    sky_depth_percentile: float = 95.0,
    filter_black_bg: bool = False,
    filter_white_bg: bool = False,
    anchor_gps: bool = True,
    far_clip_m: float = 35.0,
    height_clip_m: float = 9.0,
    max_points: int = 600000,
    progress=None,
):
    """DA3 マルチビュー予測から、視点間で整合した「点群」(trimesh.Scene)と info を返す。

    DA3 公式（export_to_glb）と同じく深度を面に張らず点群として扱う。これにより
    オクルージョン境界の引き伸ばし（溶けたメッシュ）が発生しない。さらに DA3 が
    出力する「空（オブジェクト）判定マスク」と「適応的な信頼度しきい値」を使って
    ぐちゃぐちゃの主因（空・遠景・低信頼の溶け）を根元から除去する。

    GPS アンカー（anchor_gps=True・視点2地点以上）:
      DA3 は視点間の並進（ベースライン）を実測の数分の一に圧縮し、しかも一定でない
      ため、複数視点を重ねると位置がズレて「ぐちゃぐちゃ」になる。これを防ぐため
      DA3 の各視点の「深度」と「向き(回転)」だけ採用し、視点の「位置」は実 GPS 座標で
      固定する。DA3 視点配置→実ENU配置のヨー＋スケールを 2D Umeyama で合わせ、鉛直は
      DA3 の重力で固定、各視点中心を実 GPS に스ナップする。

    パラメータ（DA3 のフルオプションを露出）:
      max_width:           各ビューの水平サンプル解像度（点密度）。
      conf_percentile:     適応信頼度しきい値の下位パーセンタイル（小さいほど残す）。
      ensure_percentile:   同・上位クランプ。切りすぎ防止。
      conf_thresh_base:    信頼度しきい値のベース値（DA3 既定 1.05）。
      drop_sky:            True で空と判定された画素を完全に除去（街並み向き）。
      sky_depth_percentile: drop_sky=False のとき、空の深度を非空深度のこの
                           パーセンタイルで埋める（無限遠化を防ぐ）。
      filter_black_bg:     ほぼ黒の背景画素を除去対象にする。
      filter_white_bg:     ほぼ白の背景画素を除去対象にする。
      anchor_gps:          True で視点位置を実 GPS で固定（上記）。視点1地点や
                           False のときは従来の DA3 ワールド配置にフォールバック。
      far_clip_m:          原点(=シーン中心)から水平にこの距離より遠い点を除去。
      height_clip_m:       推定地面からこの高さより上（頭上ドーム）を除去。
      max_points:          最終点数の上限（超えたら間引き）。
    """
    progress = progress or _noop
    depth = prediction["depth"]
    conf = prediction.get("conf")
    sky = prediction.get("sky")  # (N,H,W) bool or None
    is_metric = int(prediction.get("is_metric", 0) or 0)
    K = prediction["intrinsics"].astype(np.float64)
    ext = prediction["extrinsics"]
    images = prediction["processed_images"]
    n, h, w = depth.shape

    # 黒/白背景を「高信頼の有効画素」扱いにして残す DA3 公式とは逆に、ここでは
    # それらを除去したいので無効化マスクを作る。
    bg_drop = np.zeros((n, h, w), dtype=bool)
    if filter_black_bg:
        bg_drop |= (images < 16).all(axis=-1)
    if filter_white_bg:
        bg_drop |= (images >= 240).all(axis=-1)

    # 空の深度を埋める（drop しない場合のみ）。無限遠の空が点群を歪めるのを防ぐ。
    depth = depth.copy()
    if sky is not None and not drop_sky and sky.any():
        non_sky = depth[~sky]
        if non_sky.size > 0:
            cap = float(np.percentile(non_sky, sky_depth_percentile))
            depth[sky] = cap

    step = max(1, int(round(w / max(64, max_width))))
    us = np.arange(0, w, step)
    vs = np.arange(0, h, step)
    uu, vv = np.meshgrid(us, vs)
    pix_u = uu.astype(np.float64).ravel()
    pix_v = vv.astype(np.float64).ravel()

    conf_thr = (
        _adaptive_conf_thresh(conf, sky, conf_percentile, ensure_percentile, conf_thresh_base)
        if conf is not None
        else None
    )

    # 各ビューの c2w（DA3 ワールド）と視点ごとの平均カメラ中心を先に求める。
    c2ws = [np.linalg.inv(_as_homogeneous44(ext[i])) for i in range(n)]
    cam_centers = np.array([c[:3, 3] for c in c2ws], dtype=np.float64)
    view_index = list(view_index)
    vp_ids = sorted(set(view_index))

    # GPS アンカーの前計算：DA3 視点配置 → 実 ENU 配置のヨー＋スケール（2D）。
    anchored = False
    s2 = 1.0
    if anchor_gps and len(vp_ids) >= 2:
        q_vp = np.array(
            [cam_centers[[i for i in range(n) if view_index[i] == v]].mean(axis=0) for v in vp_ids]
        )  # (V,3) DA3 ワールドの視点中心
        p_vp = _viewpoint_enu([viewpoints[v] for v in vp_ids])  # (V,3) ENU メートル
        # DA3 の水平面は (x,z)（y は重力≒下向き）。ENU 水平 (east,north) への「向き(ヨー)」
        # だけ Umeyama で採用する。スケールは Umeyama だと視点が近似共線・DA3 ベース
        # ラインが不整合なとき退化するため使わない。
        _, R2, _ = _umeyama_2d(q_vp[:, [0, 2]], p_vp[:, [0, 2]])
        # 幾何スケールは「実距離 / DA3距離」のペア中央値（外れ値に強い）から取る。
        s2 = _metric_scale(cam_centers, view_index, viewpoints)
        if s2 is None or not np.isfinite(s2) or s2 <= 0:
            s2 = 1.0
        p_by_vp = {v: p_vp[k] for k, v in enumerate(vp_ids)}
        q_by_vp = {v: q_vp[k] for k, v in enumerate(vp_ids)}
        anchored = True

    all_pts = []
    all_col = []
    for i in range(n):
        progress("mesh", i, n, f"整合点群 生成 {i + 1}/{n}")
        c2w = c2ws[i]
        z = depth[i][np.ix_(vs, us)].astype(np.float64).ravel()
        valid = np.isfinite(z) & (z > 0)
        if conf_thr is not None:
            valid &= conf[i][np.ix_(vs, us)].ravel() >= conf_thr
        if sky is not None and drop_sky:
            valid &= ~sky[i][np.ix_(vs, us)].ravel()
        valid &= ~bg_drop[i][np.ix_(vs, us)].ravel()
        if not valid.any():
            continue
        fx, fy = K[i][0, 0], K[i][1, 1]
        cx, cy = K[i][0, 2], K[i][1, 2]
        zc = z[valid]
        xc = (pix_u[valid] - cx) / fx * zc
        yc = (pix_v[valid] - cy) / fy * zc
        cam = np.stack([xc, yc, zc], axis=-1)  # (M,3) カメラ座標(CV)

        if anchored:
            v = view_index[i]
            # DA3 ワールドで「視点中心からの相対位置」に置く（同一視点の各方向は中心共有）。
            wd = (c2w[:3, :3] @ cam.T).T + (c2w[:3, 3] - q_by_vp[v])  # (M,3)
            en = (s2 * (R2 @ wd[:, [0, 2]].T)).T  # (M,2) 水平 east,north（相対）
            east = en[:, 0] + p_by_vp[v][0]
            north = en[:, 1] + p_by_vp[v][1]
            up = -s2 * wd[:, 1]  # DA3 は下向き正なので反転して上向きへ
            # glTF(Y-up): x=east, y=up, z=-north
            world = np.stack([east, up, -north], axis=-1).astype(np.float32)
        else:
            cam_h = np.concatenate([cam, np.ones((cam.shape[0], 1))], axis=1)
            world = (c2w @ cam_h.T)[:3].T.astype(np.float32)

        col = images[i][np.ix_(vs, us)].reshape(-1, 3)[valid].astype(np.uint8)
        all_pts.append(world)
        all_col.append(col)

    progress("mesh", n, n, "整合点群 生成 完了")
    if not all_pts:
        raise ValueError("有効な点が残りませんでした（信頼度しきい値や空マスクが厳しすぎます）")
    pts = np.concatenate(all_pts, axis=0).astype(np.float64)
    col = np.concatenate(all_col, axis=0)

    if anchored:
        # 既に実 ENU メートル・Y-up。中央原点へ平行移動するだけ。
        pts -= np.median(pts, axis=0)
        scale, scale_source = float(s2), "gps_anchored"
    else:
        # glTF 整列（Y-up・中央原点）
        a = _alignment_transform(ext[0], pts)
        pts = trimesh.transform_points(pts, a)
        # メートルスケール復元。なければ DA3 がメートル絶対値なら等倍、相対なら半径15mへ。
        scale = _metric_scale(cam_centers, view_index, viewpoints)
        scale_source = "viewpoint_baseline"
        if scale is None:
            if is_metric:
                scale, scale_source = 1.0, "da3_metric"
            else:
                radius = float(np.percentile(np.linalg.norm(pts, axis=1), 95)) or 1.0
                scale, scale_source = 15.0 / radius, "radius_normalized"
        pts *= scale
    pts = pts.astype(np.float32)

    # 地面の高さを推定：中心付近(水平半径6m)の点の下位パーセンタイル。
    horiz = np.linalg.norm(pts[:, [0, 2]], axis=1)
    near = horiz < 6.0
    ground_y = float(np.percentile(pts[near, 1], 8)) if near.sum() > 50 else float(
        np.percentile(pts[:, 1], 5)
    )

    # クリップ：遠方(水平) と 頭上(地面からの高さ)
    keep = horiz <= far_clip_m
    keep &= pts[:, 1] <= ground_y + height_clip_m
    pts = pts[keep]
    col = col[keep]
    if pts.shape[0] == 0:
        raise ValueError("クリップ後に点が残りませんでした（far_clip_m / height_clip_m が厳しすぎます）")

    # 上限を超えたらランダム間引き（決定的になるよう等間隔で間引く）
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(np.int64)
        pts = pts[idx]
        col = col[idx]

    alpha = np.full((col.shape[0], 1), 255, dtype=np.uint8)
    rgba = np.concatenate([col, alpha], axis=1)
    cloud = trimesh.PointCloud(vertices=pts, colors=rgba)
    scene = trimesh.Scene()
    scene.add_geometry(cloud)

    # 点サイズの目安：シーン水平サイズ / sqrt(点数)
    diag = float(np.linalg.norm(pts[:, [0, 2]].max(axis=0) - pts[:, [0, 2]].min(axis=0))) if len(pts) else 1.0
    point_size = float(np.clip(diag / max(1.0, np.sqrt(len(pts))) * 1.5, 0.03, 0.25))

    info = {
        "representation": "pointcloud",
        "views": int(n),
        "viewpoints": int(len(viewpoints)),
        "point_count": int(len(pts)),
        "vertex_count": int(len(pts)),
        "metric_scale": float(scale),
        "scale_source": scale_source,
        "gps_anchored": bool(anchored),
        "is_metric": is_metric,
        "ground_y": float(ground_y),
        "point_size": point_size,
        "conf_percentile": float(conf_percentile),
        "ensure_percentile": float(ensure_percentile),
        "conf_thresh": float(conf_thr) if conf_thr is not None else None,
        "sky_masked": bool(sky is not None and drop_sky),
        "far_clip_m": float(far_clip_m),
        "height_clip_m": float(height_clip_m),
    }
    return scene, info
