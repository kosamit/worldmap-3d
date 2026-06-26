"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_BACKEND,
  fetchConfig,
  fetchPano,
  type AppConfig,
} from "@/lib/api";
import type { LatLng } from "@/lib/geo";
import TourViewer, { type PanoLink } from "@/components/TourViewer";
import MapPicker from "@/components/MapPicker";

interface PanoState {
  panoId: string;
  lat: number;
  lng: number;
  links: PanoLink[];
}

function headingDiff(a: number, b: number): number {
  const d = Math.abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
}

export default function Home() {
  const [backend, setBackend] = useState(DEFAULT_BACKEND);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [current, setCurrent] = useState<PanoState | null>(null);
  const [equirect, setEquirect] = useState<string | null>(null);
  const [status, setStatus] = useState("準備完了");
  const [statusError, setStatusError] = useState(false);
  const [busy, setBusy] = useState(false);
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const [history, setHistory] = useState<string[]>([]);
  const svcRef = useRef<google.maps.StreetViewService | null>(null);
  const facingRef = useRef(0);

  const say = useCallback((message: string, isError = false) => {
    setStatus(message);
    setStatusError(isError);
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cfg = await fetchConfig(backend);
        if (!cancelled) setConfig(cfg);
        if (!cancelled && !cfg.has_maps_key) {
          say("Maps APIキー未設定（backend/.env の GOOGLE_MAPS_API_KEY）", true);
        }
      } catch (err) {
        if (!cancelled) say((err as Error).message, true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [backend, say]);

  const onMapsReady = useCallback(() => {
    if (!svcRef.current && window.google?.maps) {
      svcRef.current = new window.google.maps.StreetViewService();
    }
  }, []);

  // 指定パノラマ(または座標)を表示する。faces を取得して current を更新。
  const showPano = useCallback(
    async (req: google.maps.StreetViewLocationRequest | google.maps.StreetViewPanoRequest) => {
      const svc = svcRef.current;
      if (!svc) {
        say("Street View サービスが未準備です", true);
        return;
      }
      setBusy(true);
      try {
        const { data } = await svc.getPanorama(req);
        const loc = data.location;
        if (!loc?.latLng || !loc.pano) throw new Error("パノラマ情報が不正です");
        const links: PanoLink[] = (data.links ?? [])
          .filter((l): l is google.maps.StreetViewLink & { pano: string } => !!l?.pano)
          .map((l) => ({ heading: l.heading ?? 0, pano: l.pano }));
        setCurrent({
          panoId: loc.pano,
          lat: loc.latLng.lat(),
          lng: loc.latLng.lng(),
          links,
        });
        const pano = await fetchPano(backend, { panoId: loc.pano, outWidth: 2560 });
        setEquirect(pano.equirect);
        say("移動しました。ドラッグで見回し、W/矢印で隣へ");
      } catch {
        say("この付近に Street View が見つかりません", true);
      } finally {
        setBusy(false);
      }
    },
    [backend, say],
  );

  // 地図クリック：そこに降り立つ（履歴リセット）。
  const handlePick = useCallback(
    (point: LatLng) => {
      setHistory([]);
      say("降り立っています ...");
      showPano({
        location: point,
        radius: 100,
        source: window.google.maps.StreetViewSource.OUTDOOR,
      });
    },
    [showPano, say],
  );

  // 隣接ノードへ移動（履歴に積む）。
  const stepLink = useCallback(
    (pano: string) => {
      if (current) setHistory((h) => [...h, current.panoId]);
      showPano({ pano });
    },
    [current, showPano],
  );

  // 見ている方向に最も近い隣をたどって進む。
  const handleForward = useCallback(() => {
    if (!current || current.links.length === 0) {
      say("この先に道がありません", true);
      return;
    }
    const facing = facingRef.current;
    const best = current.links.reduce((a, b) =>
      headingDiff(b.heading, facing) < headingDiff(a.heading, facing) ? b : a,
    );
    stepLink(best.pano);
  }, [current, stepLink, say]);

  const handleBack = useCallback(() => {
    const prev = history[history.length - 1];
    if (!prev) {
      say("戻れる履歴がありません", true);
      return;
    }
    setHistory((h) => h.slice(0, -1));
    showPano({ pano: prev });
  }, [history, showPano, say]);

  const handleStepLink = useCallback(
    (pano: string) => stepLink(pano),
    [stepLink],
  );

  const handleFacingChange = useCallback((bearing: number) => {
    facingRef.current = bearing;
  }, []);

  const mapPoint: LatLng | null = current
    ? { lat: current.lat, lng: current.lng }
    : null;

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
          <h2>現在地</h2>
          <div className="map">
            {mounted && config?.maps_api_key ? (
              <MapPicker
                apiKey={config.maps_api_key}
                current={mapPoint}
                onPick={handlePick}
                onReady={onMapsReady}
              />
            ) : (
              <div className="mapPlaceholder">
                {config ? "Maps APIキーが利用できません" : "地図を読み込み中 ..."}
              </div>
            )}
          </div>
          <p className="hint">
            地図を検索 / <strong>1回クリック</strong>でその地点に降り立ちます。
            そこから<strong>両隣</strong>の Street View を都度たどって歩けます。
            {busy && " （取得中…）"}
          </p>
        </section>

        <p className={statusError ? "status error" : "status"}>{status}</p>
      </aside>

      <main className="main">
        {mounted && (
          <TourViewer
            base={backend}
            equirect={equirect}
            links={current?.links ?? []}
            canBack={history.length > 0}
            onForward={handleForward}
            onBack={handleBack}
            onStepLink={handleStepLink}
            onFacingChange={handleFacingChange}
          />
        )}
      </main>
    </div>
  );
}
