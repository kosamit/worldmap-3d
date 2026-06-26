"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_BACKEND,
  fetchConfig,
  fetchPano,
  glbUrl,
  reconstructPanorama,
  reconstructMultiview,
  type AppConfig,
  type Progress,
} from "@/lib/api";
import type { LatLng } from "@/lib/geo";
import TourViewer, { type PanoLink } from "@/components/TourViewer";
import SceneViewer from "@/components/SceneViewer";
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

// 3D化（深度メッシュ）の可変パラメータ。
interface Params3D {
  depthModel: string;
  numViews: number;
  near: number;
  far: number;
  discontinuity: number;
  maxWidth: number;
}

const FALLBACK_PARAMS: Params3D = {
  depthModel: "",
  numViews: 8,
  near: 1.0,
  far: 25.0,
  discontinuity: 0.08,
  maxWidth: 256,
};

// 高精度3D化（DA3マルチビュー）の可変パラメータ。
interface MultiParams {
  depthModel: string;
  maxViews: number;
  headingCount: number;
  radiusM: number;
  confPercentile: number;
  ensurePercentile: number;
  farClipM: number;
  heightClipM: number;
  processRes: number;
  processResMethod: string;
  useRayPose: boolean;
  refViewStrategy: string;
  dropSky: boolean;
  filterBlackBg: boolean;
  filterWhiteBg: boolean;
  anchorGps: boolean;
}

const FALLBACK_MULTI: MultiParams = {
  depthModel: "",
  maxViews: 4,
  headingCount: 4,
  radiusM: 12,
  confPercentile: 40,
  ensurePercentile: 90,
  farClipM: 45,
  heightClipM: 9,
  processRes: 504,
  processResMethod: "upper_bound_resize",
  useRayPose: false,
  refViewStrategy: "saddle_balanced",
  dropSky: true,
  filterBlackBg: false,
  filterWhiteBg: false,
  anchorGps: true,
};

export default function Home() {
  const [backend, setBackend] = useState(DEFAULT_BACKEND);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [current, setCurrent] = useState<PanoState | null>(null);
  const [equirect, setEquirect] = useState<string | null>(null);
  const [status, setStatus] = useState("準備完了");
  const [statusError, setStatusError] = useState(false);
  const [busy, setBusy] = useState(false);
  // 地図ピンの向き表示用（整数度に丸めて再描画を抑える）。
  const [facingDeg, setFacingDeg] = useState(0);
  // 表示モード: パノラマ写真 or 深度メッシュ(3D化)
  const [mode, setMode] = useState<"pano" | "mesh">("pano");
  const [meshGlb, setMeshGlb] = useState<string | null>(null);
  const [building3d, setBuilding3d] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [params, setParams] = useState<Params3D>(FALLBACK_PARAMS);
  const [multi, setMulti] = useState<MultiParams>(FALLBACK_MULTI);
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const setParam = useCallback(
    <K extends keyof Params3D>(key: K, value: Params3D[K]) =>
      setParams((p) => ({ ...p, [key]: value })),
    [],
  );
  const setMultiParam = useCallback(
    <K extends keyof MultiParams>(key: K, value: MultiParams[K]) =>
      setMulti((p) => ({ ...p, [key]: value })),
    [],
  );

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
        if (!cancelled) {
          setConfig(cfg);
          // バックエンドの既定値で 3D化パラメータを初期化。
          const d = cfg.reconstruct_defaults;
          setParams({
            depthModel: cfg.depth_default_model ?? "",
            numViews: d?.num_views ?? FALLBACK_PARAMS.numViews,
            near: d?.near ?? FALLBACK_PARAMS.near,
            far: d?.far ?? FALLBACK_PARAMS.far,
            discontinuity: d?.discontinuity ?? FALLBACK_PARAMS.discontinuity,
            maxWidth: d?.max_width ?? FALLBACK_PARAMS.maxWidth,
          });
          const m = cfg.multiview_defaults;
          setMulti({
            depthModel: cfg.multiview_default_model ?? "",
            maxViews: m?.max_views ?? FALLBACK_MULTI.maxViews,
            headingCount: m?.heading_count ?? FALLBACK_MULTI.headingCount,
            radiusM: m?.radius_m ?? FALLBACK_MULTI.radiusM,
            confPercentile: m?.conf_percentile ?? FALLBACK_MULTI.confPercentile,
            ensurePercentile: m?.ensure_percentile ?? FALLBACK_MULTI.ensurePercentile,
            farClipM: m?.far_clip_m ?? FALLBACK_MULTI.farClipM,
            heightClipM: m?.height_clip_m ?? FALLBACK_MULTI.heightClipM,
            processRes: m?.process_res ?? FALLBACK_MULTI.processRes,
            processResMethod: m?.process_res_method ?? FALLBACK_MULTI.processResMethod,
            useRayPose: m?.use_ray_pose ?? FALLBACK_MULTI.useRayPose,
            refViewStrategy: m?.ref_view_strategy ?? FALLBACK_MULTI.refViewStrategy,
            dropSky: m?.drop_sky ?? FALLBACK_MULTI.dropSky,
            filterBlackBg: m?.filter_black_bg ?? FALLBACK_MULTI.filterBlackBg,
            filterWhiteBg: m?.filter_white_bg ?? FALLBACK_MULTI.filterWhiteBg,
            anchorGps: m?.anchor_gps ?? FALLBACK_MULTI.anchorGps,
          });
        }
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
    setFacingDeg((prev) => {
      const rounded = Math.round(bearing);
      return rounded === prev ? prev : rounded;
    });
  }, []);

  // 今いる地点を深度推定で立体メッシュ化し、歩ける3Dモデルに切り替える。
  const handle3D = useCallback(async () => {
    if (!current) {
      say("先に地図で地点を選んでください", true);
      return;
    }
    setBuilding3d(true);
    setProgress({
      status: "running",
      phase: "queued",
      step: 0,
      total: 0,
      percent: 0,
      message: "開始しています ...",
    });
    say("この地点を3D化中 ...");
    try {
      const meta = await reconstructPanorama(
        backend,
        {
          lat: current.lat,
          lng: current.lng,
          numViews: params.numViews,
          near: params.near,
          far: params.far,
          discontinuity: params.discontinuity,
          maxWidth: params.maxWidth,
          depthModel: params.depthModel || null,
        },
        (p) => setProgress(p),
      );
      setMeshGlb(glbUrl(backend, meta));
      setMode("mesh");
      say(`3D化完了: ${meta.vertex_count ?? "?"} 頂点。WASDで歩けます`);
    } catch (err) {
      say((err as Error).message, true);
    } finally {
      setBuilding3d(false);
    }
  }, [current, backend, say, params]);

  // 周辺の複数地点を集め、DA3マルチビューで整合した高精度メッシュを作る。
  const handle3DMulti = useCallback(async () => {
    if (!current) {
      say("先に地図で地点を選んでください", true);
      return;
    }
    setBuilding3d(true);
    setProgress({
      status: "running",
      phase: "queued",
      step: 0,
      total: 0,
      percent: 0,
      message: "開始しています ...",
    });
    say("高精度3D化中（周辺地点を収集→マルチビュー推論）...");
    try {
      const meta = await reconstructMultiview(
        backend,
        {
          lat: current.lat,
          lng: current.lng,
          maxViews: multi.maxViews,
          headingCount: multi.headingCount,
          radiusM: multi.radiusM,
          confPercentile: multi.confPercentile,
          ensurePercentile: multi.ensurePercentile,
          farClipM: multi.farClipM,
          heightClipM: multi.heightClipM,
          processRes: multi.processRes,
          processResMethod: multi.processResMethod,
          useRayPose: multi.useRayPose,
          refViewStrategy: multi.refViewStrategy,
          dropSky: multi.dropSky,
          filterBlackBg: multi.filterBlackBg,
          filterWhiteBg: multi.filterWhiteBg,
          anchorGps: multi.anchorGps,
          depthModel: multi.depthModel || null,
        },
        (p) => setProgress(p),
      );
      setMeshGlb(glbUrl(backend, meta));
      setMode("mesh");
      say(`高精度3D化完了: ${meta.vertex_count ?? "?"} 頂点。WASDで歩けます`);
    } catch (err) {
      say((err as Error).message, true);
    } finally {
      setBuilding3d(false);
    }
  }, [current, backend, say, multi]);

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
                facing={facingDeg}
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

          {config?.multiview_available && (
            <button
              type="button"
              className="primaryWide"
              onClick={handle3DMulti}
              disabled={!current || building3d}
              title="周辺の複数Street View地点を集め、DA3マルチビューで整合した高精度メッシュを作ります"
            >
              {building3d ? "生成中..." : "★ 高精度3D化（マルチビュー）"}
            </button>
          )}

          <div className="grid2 row2">
            <button
              type="button"
              onClick={handle3D}
              disabled={!current || building3d}
            >
              {building3d ? "3D化中..." : "この地点を3D化（簡易）"}
            </button>
            <button
              type="button"
              className="ghost"
              onClick={() => setMode("pano")}
              disabled={mode === "pano"}
            >
              パノラマに戻る
            </button>
          </div>

          {/* 3D化の進捗バー */}
          {progress && (building3d || progress.status !== "done") && (
            <div className="progress">
              <div className="progressBar">
                <div
                  className="progressFill"
                  style={{ width: `${progress.percent}%` }}
                  data-error={progress.status === "error"}
                />
              </div>
              <div className="progressText">
                <span>{progress.message}</span>
                <span>{progress.percent}%</span>
              </div>
            </div>
          )}

          {/* 詳細設定（折りたたみ） */}
          <button
            type="button"
            className="ghost disclosure"
            onClick={() => setShowSettings((v) => !v)}
          >
            {showSettings ? "▾ 3D化 詳細設定" : "▸ 3D化 詳細設定"}
            {config?.depth_backend ? `（${config.depth_backend}）` : ""}
          </button>

          {showSettings && (
            <div className="settings">
              <label className="field">
                深度モデル
                <input
                  type="text"
                  list="depthModels"
                  value={params.depthModel}
                  onChange={(e) => setParam("depthModel", e.target.value)}
                  disabled={building3d}
                />
                <datalist id="depthModels">
                  {(config?.model_presets ?? []).map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                </datalist>
              </label>

              <div className="grid2">
                <label className="field">
                  視点数 (2–{config?.max_panorama_views ?? 16})
                  <input
                    type="number"
                    min={2}
                    max={config?.max_panorama_views ?? 16}
                    step={1}
                    value={params.numViews}
                    onChange={(e) => setParam("numViews", Number(e.target.value))}
                    disabled={building3d}
                  />
                </label>
                <label className="field">
                  解像度 max_width
                  <input
                    type="number"
                    min={64}
                    max={1024}
                    step={32}
                    value={params.maxWidth}
                    onChange={(e) => setParam("maxWidth", Number(e.target.value))}
                    disabled={building3d}
                  />
                </label>
                <label className="field">
                  near (m)
                  <input
                    type="number"
                    min={0.1}
                    step={0.5}
                    value={params.near}
                    onChange={(e) => setParam("near", Number(e.target.value))}
                    disabled={building3d}
                  />
                </label>
                <label className="field">
                  far (m)
                  <input
                    type="number"
                    min={1}
                    step={1}
                    value={params.far}
                    onChange={(e) => setParam("far", Number(e.target.value))}
                    disabled={building3d}
                  />
                </label>
                <label className="field">
                  不連続しきい値
                  <input
                    type="number"
                    min={0.005}
                    max={1}
                    step={0.005}
                    value={params.discontinuity}
                    onChange={(e) =>
                      setParam("discontinuity", Number(e.target.value))
                    }
                    disabled={building3d}
                  />
                </label>
              </div>
              <p className="hint">
                モデルを変えると初回のみロードで時間がかかります。near/far は距離レンジ、
                不連続しきい値を上げると面が繋がりやすく（小さいと境界で分断）。
              </p>

              {config?.multiview_available && (
                <>
                  <h2 className="settingsHead">★ 高精度3D化（マルチビュー）</h2>
                  <label className="field">
                    深度モデル（カメラ対応必須）
                    <input
                      type="text"
                      list="mvModels"
                      value={multi.depthModel}
                      onChange={(e) => setMultiParam("depthModel", e.target.value)}
                      disabled={building3d}
                    />
                    <datalist id="mvModels">
                      {(config?.multiview_model_presets ?? []).map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.label}
                        </option>
                      ))}
                    </datalist>
                  </label>
                  <div className="grid2">
                    <label className="field">
                      地点数 (1–8)
                      <input
                        type="number"
                        min={1}
                        max={8}
                        step={1}
                        value={multi.maxViews}
                        onChange={(e) =>
                          setMultiParam("maxViews", Number(e.target.value))
                        }
                        disabled={building3d}
                      />
                    </label>
                    <label className="field">
                      方向数 (2–8)
                      <input
                        type="number"
                        min={2}
                        max={8}
                        step={1}
                        value={multi.headingCount}
                        onChange={(e) =>
                          setMultiParam("headingCount", Number(e.target.value))
                        }
                        disabled={building3d}
                      />
                    </label>
                    <label className="field">
                      収集半径 (m)
                      <input
                        type="number"
                        min={3}
                        max={40}
                        step={1}
                        value={multi.radiusM}
                        onChange={(e) =>
                          setMultiParam("radiusM", Number(e.target.value))
                        }
                        disabled={building3d}
                      />
                    </label>
                    <label className="field">
                      信頼度カット (%)
                      <input
                        type="number"
                        min={0}
                        max={95}
                        step={5}
                        value={multi.confPercentile}
                        onChange={(e) =>
                          setMultiParam("confPercentile", Number(e.target.value))
                        }
                        disabled={building3d}
                      />
                    </label>
                    <label className="field">
                      信頼度上限クランプ (%)
                      <input
                        type="number"
                        min={50}
                        max={100}
                        step={5}
                        value={multi.ensurePercentile}
                        onChange={(e) =>
                          setMultiParam("ensurePercentile", Number(e.target.value))
                        }
                        disabled={building3d}
                      />
                    </label>
                    <label className="field">
                      遠方クリップ (m)
                      <input
                        type="number"
                        min={5}
                        max={200}
                        step={5}
                        value={multi.farClipM}
                        onChange={(e) =>
                          setMultiParam("farClipM", Number(e.target.value))
                        }
                        disabled={building3d}
                      />
                    </label>
                    <label className="field">
                      頭上クリップ (m)
                      <input
                        type="number"
                        min={2}
                        max={50}
                        step={1}
                        value={multi.heightClipM}
                        onChange={(e) =>
                          setMultiParam("heightClipM", Number(e.target.value))
                        }
                        disabled={building3d}
                      />
                    </label>
                    <label className="field">
                      処理解像度 (168–1008)
                      <input
                        type="number"
                        min={168}
                        max={1008}
                        step={28}
                        value={multi.processRes}
                        onChange={(e) =>
                          setMultiParam("processRes", Number(e.target.value))
                        }
                        disabled={building3d}
                      />
                    </label>
                    <label className="field">
                      リサイズ方式
                      <select
                        value={multi.processResMethod}
                        onChange={(e) =>
                          setMultiParam("processResMethod", e.target.value)
                        }
                        disabled={building3d}
                      >
                        {(config?.multiview_process_res_methods ?? []).map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="field">
                      参照ビュー戦略
                      <select
                        value={multi.refViewStrategy}
                        onChange={(e) =>
                          setMultiParam("refViewStrategy", e.target.value)
                        }
                        disabled={building3d}
                      >
                        {(config?.multiview_ref_view_strategies ?? []).map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <div className="grid2">
                    <label className="checkField">
                      <input
                        type="checkbox"
                        checked={multi.anchorGps}
                        onChange={(e) =>
                          setMultiParam("anchorGps", e.target.checked)
                        }
                        disabled={building3d}
                      />
                      視点位置をGPSで固定（推奨）
                    </label>
                    <label className="checkField">
                      <input
                        type="checkbox"
                        checked={multi.useRayPose}
                        onChange={(e) =>
                          setMultiParam("useRayPose", e.target.checked)
                        }
                        disabled={building3d}
                      />
                      レイベースのポーズ推定
                    </label>
                    <label className="checkField">
                      <input
                        type="checkbox"
                        checked={multi.dropSky}
                        onChange={(e) =>
                          setMultiParam("dropSky", e.target.checked)
                        }
                        disabled={building3d}
                      />
                      空（オブジェクト判定）を除去
                    </label>
                    <label className="checkField">
                      <input
                        type="checkbox"
                        checked={multi.filterBlackBg}
                        onChange={(e) =>
                          setMultiParam("filterBlackBg", e.target.checked)
                        }
                        disabled={building3d}
                      />
                      黒背景を除去
                    </label>
                    <label className="checkField">
                      <input
                        type="checkbox"
                        checked={multi.filterWhiteBg}
                        onChange={(e) =>
                          setMultiParam("filterWhiteBg", e.target.checked)
                        }
                        disabled={building3d}
                      />
                      白背景を除去
                    </label>
                  </div>
                  <p className="hint">
                    地点数×方向数 の画像をDA3に一括投入して整合。<b>視点位置をGPSで固定</b>は
                    DA3が圧縮しがちな視点間距離を実GPS座標で上書きし、複数視点を重ねたときの
                    「ぐちゃぐちゃ」を防ぐ最重要オプションです。<b>レイベースのポーズ推定</b>は
                    各画素レイから RANSAC でカメラを解く別法。信頼度カット（下限）と上限クランプ
                    の間で適応しきい値が決まります。空・遠景は遠方/頭上クリップで除去します。
                  </p>
                </>
              )}
            </div>
          )}

          <p className="hint">
            「3D化」は今いる地点を深度推定で立体メッシュ化し、WASDで歩けます。
          </p>
        </section>

        <p className={statusError ? "status error" : "status"}>{status}</p>
      </aside>

      <main className="main">
        {mounted && mode === "mesh" ? (
          <SceneViewer glbUrl={meshGlb} />
        ) : (
          mounted && (
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
          )
        )}
      </main>
    </div>
  );
}
