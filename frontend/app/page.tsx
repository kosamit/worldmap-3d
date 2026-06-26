"use client";

import { useCallback, useEffect, useState } from "react";
import {
  DEFAULT_BACKEND,
  fetchConfig,
  fetchScenes,
  glbUrl,
  reconstructRoute,
  type AppConfig,
  type SceneMeta,
} from "@/lib/api";
import type { LatLng } from "@/lib/geo";
import SceneViewer from "@/components/SceneViewer";
import MapPicker from "@/components/MapPicker";

export default function Home() {
  const [backend, setBackend] = useState(DEFAULT_BACKEND);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [scenes, setScenes] = useState<SceneMeta[]>([]);
  const [points, setPoints] = useState<LatLng[]>([]);
  const [views, setViews] = useState(6);
  const [step, setStep] = useState(20);
  const [resetSignal, setResetSignal] = useState(0);
  const [currentGlb, setCurrentGlb] = useState<string | null>(null);
  const [status, setStatus] = useState("準備完了");
  const [statusError, setStatusError] = useState(false);
  const [busy, setBusy] = useState(false);
  // three / google-maps はブラウザ専用なので、マウント後にだけ描画する。
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const say = useCallback((message: string, isError = false) => {
    setStatus(message);
    setStatusError(isError);
  }, []);

  const refreshScenes = useCallback(async () => {
    try {
      setScenes(await fetchScenes(backend));
    } catch (err) {
      say((err as Error).message, true);
    }
  }, [backend, say]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cfg = await fetchConfig(backend);
        if (cancelled) return;
        setConfig(cfg);
        if (!cfg.has_maps_key) {
          say("Maps APIキー未設定（backend/.env の GOOGLE_MAPS_API_KEY）", true);
        }
      } catch (err) {
        if (!cancelled) say((err as Error).message, true);
      }
      refreshScenes();
    })();
    return () => {
      cancelled = true;
    };
  }, [backend, refreshScenes, say]);

  const onPointsChange = useCallback((p: LatLng[]) => setPoints(p), []);

  const loadScene = useCallback(
    (meta: SceneMeta) => {
      setCurrentGlb(glbUrl(backend, meta));
      const verts = meta.vertex_count ? `, ${meta.vertex_count}頂点` : "";
      say(`表示中: ${meta.id} (${meta.source}${verts})`);
    },
    [backend, say],
  );

  const handleRoute = useCallback(async () => {
    if (points.length < 2) {
      say("地図で始点と終点をクリックしてください", true);
      return;
    }
    setBusy(true);
    say(`道沿い${points.length}点を360°撮影して3D化中（数十秒〜）...`);
    try {
      const meta = await reconstructRoute(backend, {
        points,
        num_views: views,
        fov: 90,
      });
      await refreshScenes();
      loadScene(meta);
    } catch (err) {
      say((err as Error).message, true);
    } finally {
      setBusy(false);
    }
  }, [points, views, backend, refreshScenes, loadScene, say]);

  const handleClear = useCallback(() => {
    setResetSignal((s) => s + 1);
    setPoints([]);
  }, []);

  return (
    <div className="layout">
      <aside className="panel">
        <h1>WorldMap 3D</h1>

        <label className="field">
          バックエンドURL
          <input
            type="text"
            value={backend}
            onChange={(e) => setBackend(e.target.value)}
          />
        </label>

        <section>
          <div className="sectionHead">
            <h2>蓄積シーン</h2>
            <button type="button" onClick={refreshScenes}>
              更新
            </button>
          </div>
          <ul className="sceneList">
            {scenes.length === 0 ? (
              <li className="empty">（まだありません）</li>
            ) : (
              scenes.map((meta) => (
                <li key={meta.id} onClick={() => loadScene(meta)}>
                  {meta.id} —{" "}
                  {meta.location?.lat != null && meta.location?.lng != null
                    ? `${meta.location.lat.toFixed(4)}, ${meta.location.lng.toFixed(4)}`
                    : meta.source}
                </li>
              ))
            )}
          </ul>
        </section>

        <section>
          <h2>地図から道沿い3D化</h2>
          <div className="map">
            {mounted && config?.maps_api_key ? (
              <MapPicker
                apiKey={config.maps_api_key}
                maxPoints={config.max_route_points}
                stepMeters={step}
                resetSignal={resetSignal}
                onPointsChange={onPointsChange}
              />
            ) : (
              <div className="mapPlaceholder">
                {config
                  ? "Maps APIキーが利用できません"
                  : "地図を読み込み中 ..."}
              </div>
            )}
          </div>

          <div className="grid2">
            <label className="inline">
              視点数
              <input
                type="number"
                min={2}
                max={16}
                value={views}
                onChange={(e) => setViews(Number(e.target.value) || 6)}
              />
            </label>
            <label className="inline">
              間隔(m)
              <input
                type="number"
                min={5}
                max={100}
                value={step}
                onChange={(e) => setStep(Number(e.target.value) || 20)}
              />
            </label>
          </div>
          <div className="grid2 row2">
            <button type="button" onClick={handleRoute} disabled={busy}>
              {busy ? "生成中..." : "道沿いを3D化"}
            </button>
            <button type="button" className="ghost" onClick={handleClear}>
              クリア
            </button>
          </div>
          <p className="hint">
            地図を1回クリックで<strong>始点</strong>、もう1回で
            <strong>終点</strong>。間に <strong>{points.length}</strong>{" "}
            点を取り、各点を360°撮影して歩いてつながる3D空間に合成します。
          </p>
        </section>

        <p className={statusError ? "status error" : "status"}>{status}</p>
      </aside>

      <main className="main">
        {mounted && <SceneViewer glbUrl={currentGlb} />}
      </main>
    </div>
  );
}
