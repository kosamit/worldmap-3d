// バックエンド (FastAPI) との通信ヘルパー。

export function normalizeBase(url) {
  return url.replace(/\/+$/, "");
}

export async function fetchScenes(base) {
  const res = await fetch(`${normalizeBase(base)}/api/scenes`);
  if (!res.ok) throw new Error(`シーン一覧の取得に失敗 (${res.status})`);
  const data = await res.json();
  return data.scenes ?? [];
}

export async function reconstructFromUpload(base, file, fov) {
  const form = new FormData();
  form.append("file", file);
  form.append("fov", String(fov));
  const res = await fetch(`${normalizeBase(base)}/api/reconstruct`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await safeDetail(res);
    throw new Error(`3D化に失敗 (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function reconstructPanorama(base, params) {
  const form = new FormData();
  for (const [key, value] of Object.entries(params)) {
    if (value !== "" && value !== null && value !== undefined) {
      form.append(key, String(value));
    }
  }
  const res = await fetch(`${normalizeBase(base)}/api/reconstruct/panorama`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await safeDetail(res);
    throw new Error(`360°パノラマの3D化に失敗 (${res.status}): ${detail}`);
  }
  return res.json();
}

async function safeDetail(res) {
  try {
    const data = await res.json();
    return data.detail ?? JSON.stringify(data);
  } catch {
    return res.statusText;
  }
}
