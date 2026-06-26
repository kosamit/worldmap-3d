// FastAPI バックエンドとの通信ヘルパーと型定義。

import type { LatLng } from "./geo";

export const DEFAULT_BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export interface AppConfig {
  maps_api_key: string | null;
  has_maps_key: boolean;
  max_route_points: number;
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
