// FastAPI バックエンドとの通信ヘルパーと型定義。

import type { LatLng } from "./geo";

export const DEFAULT_BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export interface ModelPreset {
  id: string;
  label: string;
}

export interface ReconstructDefaults {
  near: number;
  far: number;
  discontinuity: number;
  max_width: number;
  num_views: number;
}

export interface AppConfig {
  maps_api_key: string | null;
  has_maps_key: boolean;
  max_route_points: number;
  depth_backend?: string;
  depth_default_model?: string;
  model_presets?: ModelPreset[];
  max_panorama_views?: number;
  reconstruct_defaults?: ReconstructDefaults;
}

// 3D化パラメータ（フロントの「詳細設定」と対応）。
export interface ReconstructParams {
  lat: number;
  lng: number;
  numViews?: number;
  fov?: number;
  near?: number;
  far?: number;
  discontinuity?: number;
  maxWidth?: number;
  depthModel?: string | null;
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

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * 3D化ジョブを開始し、完了までポーリングする。進捗は onProgress に通知する。
 * 完了時に生成シーンの meta を返す。失敗時は例外を投げる。
 */
export async function reconstructPanorama(
  base: string,
  params: ReconstructParams,
  onProgress?: (p: Progress) => void,
): Promise<SceneMeta> {
  const root = normalizeBase(base);
  const form = new FormData();
  form.append("lat", String(params.lat));
  form.append("lng", String(params.lng));
  form.append("num_views", String(params.numViews ?? 8));
  form.append("fov", String(params.fov ?? 90));
  if (params.near != null) form.append("near", String(params.near));
  if (params.far != null) form.append("far", String(params.far));
  if (params.discontinuity != null)
    form.append("discontinuity", String(params.discontinuity));
  if (params.maxWidth != null) form.append("max_width", String(params.maxWidth));
  if (params.depthModel) form.append("depth_model", params.depthModel);

  const res = await fetch(`${root}/api/reconstruct/panorama`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await safeDetail(res);
    throw new Error(`3D化の開始に失敗 (${res.status}): ${detail}`);
  }
  const { job_id: jobId } = (await res.json()) as { job_id: string };

  // 進捗ポーリング（モデルロードを含むため十分に長く待つ）。
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

export async function fetchPano(
  base: string,
  params: { panoId: string; outWidth?: number },
): Promise<PanoResult> {
  const form = new FormData();
  form.append("pano_id", params.panoId);
  if (params.outWidth) form.append("out_width", String(params.outWidth));
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
