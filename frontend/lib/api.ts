// FastAPI バックエンドとの通信ヘルパーと型定義。

import type { LatLng } from "./geo";

export const DEFAULT_BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export interface ModelPreset {
  id: string;
  label: string;
}

export interface MultiviewDefaults {
  max_views: number;
  heading_count: number;
  pitch_count: number;
  radius_m: number;
  fov: number;
  max_width: number;
  conf_percentile: number;
  ensure_percentile: number;
  far_clip_m: number;
  height_clip_m: number;
  edge_factor: number;
  discontinuity_ratio: number;
  cut_stretch: boolean;
  front_discontinuity: number;
  ground_cap: boolean;
  camera_height_m: number;
  tsdf_voxel: number;
  process_res: number;
  process_res_method: string;
  use_ray_pose: boolean;
  ref_view_strategy: string;
  drop_sky: boolean;
  filter_black_bg: boolean;
  filter_white_bg: boolean;
  anchor_gps: boolean;
  raw: boolean;
  remove_objects: boolean;
  remove_classes: string;
  inpaint: boolean;
}

export interface AppConfig {
  maps_api_key: string | null;
  has_maps_key: boolean;
  max_route_points: number;
  depth_backend?: string;
  depth_default_model?: string;
  model_presets?: ModelPreset[];
  multiview_available?: boolean;
  multiview_default_model?: string;
  multiview_model_presets?: ModelPreset[];
  multiview_defaults?: MultiviewDefaults;
  multiview_ref_view_strategies?: ModelPreset[];
  multiview_process_res_methods?: ModelPreset[];
}

// 3D化ジョブの進捗。
export interface Progress {
  status: "running" | "done" | "error";
  phase: string;
  step: number;
  total: number;
  percent: number;
  message: string;
  result?: SceneMeta | null;
  error?: string | null;
}

export interface SceneLocation {
  lat?: number;
  lng?: number;
  points?: number;
  num_views?: number;
}

export interface SceneMeta {
  id: string;
  source: string;
  created_at: string;
  glb_url: string;
  location?: SceneLocation | null;
  vertex_count?: number;
  points?: number;
  viewpoints?: number;
  requested_views?: number;
  views_capped?: boolean;
  images_used?: number;
  representation?: string;       // "gaussian" など
  splat_url?: string;            // 3DGS の .splat 配信パス（drei <Splat> 用）
  gaussian_count?: number;
}

export interface RouteParams {
  points: LatLng[];
  num_views: number;
  fov: number;
}

export interface TourNode {
  index: number;
  lat: number;
  lng: number;
  pano_id?: string | null;
  x: number;
  z: number;
  faces: Record<string, string>;
}

export interface TourMeta {
  id: string;
  origin: { lat: number; lng: number };
  node_count: number;
  nodes: TourNode[];
}

export interface PanoResult {
  pano_id: string;
  equirect: string;
}

function normalizeBase(url: string): string {
  return url.replace(/\/+$/, "");
}

async function safeDetail(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return data.detail ?? JSON.stringify(data);
  } catch {
    return res.statusText;
  }
}

export async function fetchConfig(base: string): Promise<AppConfig> {
  const res = await fetch(`${normalizeBase(base)}/api/config`);
  if (!res.ok) throw new Error(`設定の取得に失敗 (${res.status})`);
  return res.json();
}

// GPU メモリ使用量・使用率（パネルの定期表示用）。
export interface GpuStatus {
  available: boolean;
  name?: string;
  used_mb?: number;
  total_mb?: number;
  free_mb?: number;
  used_pct?: number;
  torch_allocated_mb?: number;
  torch_reserved_mb?: number;
  util_pct?: number | null;
  error?: string;
}

export async function fetchGpu(base: string): Promise<GpuStatus> {
  const res = await fetch(`${normalizeBase(base)}/api/gpu`);
  if (!res.ok) throw new Error(`GPU情報の取得に失敗 (${res.status})`);
  return res.json();
}

export async function freeGpu(
  base: string,
): Promise<{ freed: string[]; gpu: GpuStatus }> {
  const res = await fetch(`${normalizeBase(base)}/api/gpu/free`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`GPU解放に失敗 (${res.status})`);
  return res.json();
}

export async function fetchScenes(base: string): Promise<SceneMeta[]> {
  const res = await fetch(`${normalizeBase(base)}/api/scenes`);
  if (!res.ok) throw new Error(`シーン一覧の取得に失敗 (${res.status})`);
  const data = await res.json();
  return data.scenes ?? [];
}

export async function reconstructRoute(
  base: string,
  params: RouteParams,
): Promise<SceneMeta> {
  const form = new FormData();
  form.append("points", JSON.stringify(params.points));
  form.append("num_views", String(params.num_views));
  form.append("fov", String(params.fov));

  const res = await fetch(`${normalizeBase(base)}/api/reconstruct/route`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await safeDetail(res);
    throw new Error(`道沿いルートの3D化に失敗 (${res.status}): ${detail}`);
  }
  return res.json();
}

export function glbUrl(base: string, meta: SceneMeta): string {
  return `${normalizeBase(base)}${meta.glb_url}`;
}

// 3DGS シーンの .splat フル URL（別オリジンの :8000 を直接 fetch する）。
// splat_url が無い（非 gaussian）シーンでは null。
export function splatUrl(base: string, meta: SceneMeta): string | null {
  return meta.splat_url ? `${normalizeBase(base)}${meta.splat_url}` : null;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// 3D化ジョブの進捗を完了までポーリングし、結果 meta を返す共通ヘルパー。
async function pollJob(
  root: string,
  jobId: string,
  onProgress?: (p: Progress) => void,
): Promise<SceneMeta> {
  for (let i = 0; i < 3600; i++) {
    await sleep(700);
    let prog: Progress;
    try {
      const pres = await fetch(`${root}/api/reconstruct/progress/${jobId}`);
      if (!pres.ok) {
        if (pres.status === 404) throw new Error("ジョブが見つかりません");
        continue; // 一時的なエラーはリトライ
      }
      prog = (await pres.json()) as Progress;
    } catch {
      continue; // ネットワーク瞬断はリトライ
    }
    onProgress?.(prog);
    if (prog.status === "done") {
      if (!prog.result) throw new Error("結果が空です");
      return prog.result;
    }
    if (prog.status === "error") {
      throw new Error(prog.error || prog.message || "3D化に失敗しました");
    }
  }
  throw new Error("3D化がタイムアウトしました");
}

// 高精度3D化（DA3 マルチビュー整合メッシュ）のパラメータ。
export interface MultiviewParams {
  lat: number;
  lng: number;
  maxViews?: number;
  headingCount?: number;
  pitchCount?: number;
  radiusM?: number;
  fov?: number;
  maxWidth?: number;
  confPercentile?: number;
  ensurePercentile?: number;
  farClipM?: number;
  heightClipM?: number;
  edgeFactor?: number;
  discontinuityRatio?: number;
  cutStretch?: boolean;
  frontDiscontinuity?: number;
  groundCap?: boolean;
  cameraHeightM?: number;
  processRes?: number;
  processResMethod?: string;
  useRayPose?: boolean;
  refViewStrategy?: string;
  dropSky?: boolean;
  filterBlackBg?: boolean;
  filterWhiteBg?: boolean;
  anchorGps?: boolean;
  rawMode?: boolean;
  removeObjects?: boolean;
  removeClasses?: string;
  inpaint?: boolean;
  method?:
    | "mesh"
    | "tsdf"
    | "poisson"
    | "panorama"
    | "primitive"
    | "colliders"
    | "gaussian"
    | "proxy";
  tsdfVoxel?: number;
  depthModel?: string | null;
  enhanceInput?: boolean;
  enhanceMode?: "light" | "esrgan";
  enhanceUpscale?: number;
}

/**
 * 周辺の複数 Street View 地点を集め、DA3 マルチビューで整合した歩けるメッシュを作る。
 * ジョブを開始し、完了までポーリングして結果 meta を返す。
 */
export async function reconstructMultiview(
  base: string,
  params: MultiviewParams,
  onProgress?: (p: Progress) => void,
): Promise<SceneMeta> {
  const root = normalizeBase(base);
  const form = new FormData();
  form.append("lat", String(params.lat));
  form.append("lng", String(params.lng));
  if (params.maxViews != null) form.append("max_views", String(params.maxViews));
  if (params.headingCount != null)
    form.append("heading_count", String(params.headingCount));
  if (params.pitchCount != null)
    form.append("pitch_count", String(params.pitchCount));
  if (params.radiusM != null) form.append("radius_m", String(params.radiusM));
  if (params.fov != null) form.append("fov", String(params.fov));
  if (params.maxWidth != null) form.append("max_width", String(params.maxWidth));
  if (params.confPercentile != null)
    form.append("conf_percentile", String(params.confPercentile));
  if (params.ensurePercentile != null)
    form.append("ensure_percentile", String(params.ensurePercentile));
  if (params.farClipM != null) form.append("far_clip_m", String(params.farClipM));
  if (params.heightClipM != null)
    form.append("height_clip_m", String(params.heightClipM));
  if (params.edgeFactor != null)
    form.append("edge_factor", String(params.edgeFactor));
  if (params.discontinuityRatio != null)
    form.append("discontinuity_ratio", String(params.discontinuityRatio));
  if (params.cutStretch != null)
    form.append("cut_stretch", String(params.cutStretch));
  if (params.frontDiscontinuity != null)
    form.append("front_discontinuity", String(params.frontDiscontinuity));
  if (params.groundCap != null)
    form.append("ground_cap", String(params.groundCap));
  if (params.cameraHeightM != null)
    form.append("camera_height_m", String(params.cameraHeightM));
  if (params.processRes != null)
    form.append("process_res", String(params.processRes));
  if (params.processResMethod)
    form.append("process_res_method", params.processResMethod);
  if (params.useRayPose != null)
    form.append("use_ray_pose", String(params.useRayPose));
  if (params.refViewStrategy)
    form.append("ref_view_strategy", params.refViewStrategy);
  if (params.dropSky != null) form.append("drop_sky", String(params.dropSky));
  if (params.filterBlackBg != null)
    form.append("filter_black_bg", String(params.filterBlackBg));
  if (params.filterWhiteBg != null)
    form.append("filter_white_bg", String(params.filterWhiteBg));
  if (params.anchorGps != null)
    form.append("anchor_gps", String(params.anchorGps));
  if (params.rawMode != null) form.append("raw", String(params.rawMode));
  if (params.removeObjects != null) form.append("remove_objects", String(params.removeObjects));
  if (params.removeClasses) form.append("remove_classes", params.removeClasses);
  if (params.inpaint != null) form.append("inpaint", String(params.inpaint));
  if (params.method) form.append("method", params.method);
  if (params.tsdfVoxel != null) form.append("tsdf_voxel", String(params.tsdfVoxel));
  if (params.depthModel) form.append("depth_model", params.depthModel);
  if (params.enhanceInput != null)
    form.append("enhance_input", String(params.enhanceInput));
  if (params.enhanceMode) form.append("enhance_mode", params.enhanceMode);
  if (params.enhanceUpscale != null)
    form.append("enhance_upscale", String(params.enhanceUpscale));

  const res = await fetch(`${root}/api/reconstruct/multiview`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await safeDetail(res);
    throw new Error(`高精度3D化の開始に失敗 (${res.status}): ${detail}`);
  }
  const { job_id: jobId } = (await res.json()) as { job_id: string };
  return pollJob(root, jobId, onProgress);
}

export async function fetchPano(
  base: string,
  params: { panoId: string; outWidth?: number; hi?: boolean },
): Promise<PanoResult> {
  const form = new FormData();
  form.append("pano_id", params.panoId);
  if (params.outWidth) form.append("out_width", String(params.outWidth));
  if (params.hi) form.append("hi", "true");
  const res = await fetch(`${normalizeBase(base)}/api/streetview/cube`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await safeDetail(res);
    throw new Error(`パノラマ取得に失敗 (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function fetchTour(
  base: string,
  params: { points: LatLng[]; faceSize?: number },
): Promise<TourMeta> {
  const form = new FormData();
  form.append("points", JSON.stringify(params.points));
  if (params.faceSize) form.append("face_size", String(params.faceSize));

  const res = await fetch(`${normalizeBase(base)}/api/route/tour`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await safeDetail(res);
    throw new Error(`Street Viewツアーの生成に失敗 (${res.status}): ${detail}`);
  }
  return res.json();
}
