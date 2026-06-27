"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_BACKEND,
  fetchConfig,
  fetchPano,
  glbUrl,
  splatUrl,
  reconstructMultiview,
  fetchGpu,
  freeGpu,
  type AppConfig,
  type GpuStatus,
  type Progress,
} from "@/lib/api";
import type { LatLng } from "@/lib/geo";
import TourViewer, { type PanoLink } from "@/components/TourViewer";
import SceneViewer, { type PanoInfo } from "@/components/SceneViewer";
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

// 高精度3D化（DA3マルチビュー）の可変パラメータ。
interface MultiParams {
  depthModel: string;
  maxViews: number;
  headingCount: number;
  pitchCount: number;
  radiusM: number;
  confPercentile: number;
  ensurePercentile: number;
  farClipM: number;
  heightClipM: number;
  edgeFactor: number;
  discontinuityRatio: number;
  cutStretch: boolean;
  frontDiscontinuity: number;
  groundCap: boolean;
  cameraHeightM: number;
  tsdfVoxel: number;
  fov: number;
  processRes: number;
  processResMethod: string;
  useRayPose: boolean;
  refViewStrategy: string;
  enhanceInput: boolean;
  enhanceMode: "light" | "esrgan";
  dropSky: boolean;
  filterBlackBg: boolean;
  filterWhiteBg: boolean;
  anchorGps: boolean;
  rawMode: boolean;
  removeObjects: boolean;
  removeClasses: string;
  inpaint: boolean;
}

const FALLBACK_MULTI: MultiParams = {
  depthModel: "",
  maxViews: 3,
  headingCount: 8,
  pitchCount: 3,
  radiusM: 12,
  confPercentile: 10,
  ensurePercentile: 90,
  farClipM: 0,
  heightClipM: 40,
  edgeFactor: 0.7,
  discontinuityRatio: 0.15,
  cutStretch: true,
  frontDiscontinuity: 0.45,
  groundCap: false,
  cameraHeightM: 2.5,
  tsdfVoxel: 0.12,
  fov: 90,
  processRes: 504,
  processResMethod: "upper_bound_resize",
  useRayPose: true,
  refViewStrategy: "saddle_balanced",
  enhanceInput: false,
  enhanceMode: "light",
  dropSky: true,
  filterBlackBg: false,
  filterWhiteBg: false,
  anchorGps: true,
  rawMode: false,
  removeObjects: false,
  removeClasses: "person",
  inpaint: false,
};

// 現在地(lat,lng,向き)を URL に載せ、Google Maps のように共有・ブックマーク可能にする。
function parseLocationFromUrl():
  | { lat: number; lng: number; heading: number | null }
  | null {
  if (typeof window === "undefined") return null;
  const sp = new URLSearchParams(window.location.search);
  const lat = Number(sp.get("lat"));
  const lng = Number(sp.get("lng"));
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
  if (lat < -90 || lat > 90 || lng < -180 || lng > 180) return null;
  const hRaw = sp.get("h") ?? sp.get("heading");
  const h = hRaw == null ? NaN : Number(hRaw);
  return { lat, lng, heading: Number.isFinite(h) ? h : null };
}

function writeLocationToUrl(
  lat: number,
  lng: number,
  heading: number | null,
): void {
  if (typeof window === "undefined") return;
  const sp = new URLSearchParams(window.location.search);
  sp.set("lat", lat.toFixed(6));
  sp.set("lng", lng.toFixed(6));
  if (heading != null && Number.isFinite(heading)) {
    sp.set("h", String(((Math.round(heading) % 360) + 360) % 360));
  }
  const url = `${window.location.pathname}?${sp.toString()}${window.location.hash}`;
  // replaceState: 見回しのたびに履歴を積まずに URL だけ更新する。
  window.history.replaceState(null, "", url);
}

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
  // 3DGS シーンの .splat URL（あれば SceneViewer が写実 splat を描画）。
  const [meshSplat, setMeshSplat] = useState<string | null>(null);
  // 3D化した時点でパノラマで向いていた方位（北=0,時計回り）。3D初期視線に使う。
  const [sceneHeadingDeg, setSceneHeadingDeg] = useState<number | null>(null);
  // 複数パノ（つなぎ目なし）の各パノ中心。ビューワーの距離フェードに渡す。
  const [scenePanos, setScenePanos] = useState<PanoInfo[] | undefined>(undefined);
  const [building3d, setBuilding3d] = useState(false);
  const [enhancing, setEnhancing] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [gpu, setGpu] = useState<GpuStatus | null>(null);
  const [freeing, setFreeing] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [multi, setMulti] = useState<MultiParams>(FALLBACK_MULTI);
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

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
          const m = cfg.multiview_defaults;
          setMulti({
            depthModel: cfg.multiview_default_model ?? "",
            maxViews: m?.max_views ?? FALLBACK_MULTI.maxViews,
            headingCount: m?.heading_count ?? FALLBACK_MULTI.headingCount,
            pitchCount: m?.pitch_count ?? FALLBACK_MULTI.pitchCount,
            radiusM: m?.radius_m ?? FALLBACK_MULTI.radiusM,
            confPercentile: m?.conf_percentile ?? FALLBACK_MULTI.confPercentile,
            ensurePercentile: m?.ensure_percentile ?? FALLBACK_MULTI.ensurePercentile,
            farClipM: m?.far_clip_m ?? FALLBACK_MULTI.farClipM,
            heightClipM: m?.height_clip_m ?? FALLBACK_MULTI.heightClipM,
            edgeFactor: m?.edge_factor ?? FALLBACK_MULTI.edgeFactor,
            discontinuityRatio:
              m?.discontinuity_ratio ?? FALLBACK_MULTI.discontinuityRatio,
            cutStretch: m?.cut_stretch ?? FALLBACK_MULTI.cutStretch,
            frontDiscontinuity:
              m?.front_discontinuity ?? FALLBACK_MULTI.frontDiscontinuity,
            groundCap: m?.ground_cap ?? FALLBACK_MULTI.groundCap,
            cameraHeightM: m?.camera_height_m ?? FALLBACK_MULTI.cameraHeightM,
            tsdfVoxel: m?.tsdf_voxel ?? FALLBACK_MULTI.tsdfVoxel,
            fov: m?.fov ?? FALLBACK_MULTI.fov,
            processRes: m?.process_res ?? FALLBACK_MULTI.processRes,
            processResMethod: m?.process_res_method ?? FALLBACK_MULTI.processResMethod,
            useRayPose: m?.use_ray_pose ?? FALLBACK_MULTI.useRayPose,
            refViewStrategy: m?.ref_view_strategy ?? FALLBACK_MULTI.refViewStrategy,
            enhanceInput: FALLBACK_MULTI.enhanceInput,
            enhanceMode: FALLBACK_MULTI.enhanceMode,
            dropSky: m?.drop_sky ?? FALLBACK_MULTI.dropSky,
            filterBlackBg: m?.filter_black_bg ?? FALLBACK_MULTI.filterBlackBg,
            filterWhiteBg: m?.filter_white_bg ?? FALLBACK_MULTI.filterWhiteBg,
            anchorGps: m?.anchor_gps ?? FALLBACK_MULTI.anchorGps,
            rawMode: m?.raw ?? FALLBACK_MULTI.rawMode,
            removeObjects: m?.remove_objects ?? FALLBACK_MULTI.removeObjects,
            removeClasses: m?.remove_classes ?? FALLBACK_MULTI.removeClasses,
            inpaint: m?.inpaint ?? FALLBACK_MULTI.inpaint,
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

  // GPU メモリ使用量を定期取得（生成中は短間隔で）。失敗時は静かに前回値を保持。
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const g = await fetchGpu(backend);
        if (!cancelled) setGpu(g);
      } catch {
        /* バックエンド未起動など: 無視して次回リトライ */
      }
    };
    tick();
    const ms = building3d ? 1000 : 2500;
    const id = setInterval(tick, ms);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [backend, building3d]);

  // ロード済みモデルを解放してGPUメモリを空ける（実行前のメモリ確保用）。
  const handleFreeGpu = useCallback(async () => {
    setFreeing(true);
    try {
      const { freed, gpu: g } = await freeGpu(backend);
      setGpu(g);
      say(
        freed.length
          ? `GPU解放: ${freed.join(", ")} を解放しました`
          : "GPU解放: 解放対象なし（モデル未ロード）",
      );
    } catch (err) {
      say((err as Error).message, true);
    } finally {
      setFreeing(false);
    }
  }, [backend, say]);

  const [mapsReady, setMapsReady] = useState(false);
  const urlInitDone = useRef(false);
  const onMapsReady = useCallback(() => {
    if (!svcRef.current && window.google?.maps) {
      svcRef.current = new window.google.maps.StreetViewService();
    }
    setMapsReady(true);
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

  // 初回: URL に座標があればそこへ降り立つ（共有リンク・リロード復元）。
  useEffect(() => {
    if (!mapsReady || urlInitDone.current) return;
    urlInitDone.current = true;
    const loc = parseLocationFromUrl();
    if (!loc) return;
    if (loc.heading != null) {
      facingRef.current = loc.heading;
      setFacingDeg(Math.round(loc.heading));
    }
    setHistory([]);
    showPano({
      location: { lat: loc.lat, lng: loc.lng },
      radius: 100,
      source: window.google.maps.StreetViewSource.OUTDOOR,
    });
  }, [mapsReady, showPano]);

  // 現在地・向きを URL に反映（履歴を汚さない replaceState）。
  useEffect(() => {
    if (!current) return;
    writeLocationToUrl(current.lat, current.lng, facingDeg);
  }, [current, facingDeg]);

  // 周辺の複数地点を集め、DA3マルチビューで整合した高精度メッシュを作る。
  // method="tsdf" で TSDF 融合（重なり層を1枚の連続面へ＝ソリッド）。
  const handle3DMulti = useCallback(async (method: "mesh" | "tsdf" | "poisson" | "panorama" | "primitive" | "colliders" | "gaussian" = "mesh") => {
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
    say(
      method === "tsdf"
        ? "TSDF 3D化中（周辺地点を収集→推論→TSDF融合）..."
        : "高精度3D化中（周辺地点を収集→マルチビュー推論）...",
    );
    try {
      const meta = await reconstructMultiview(
        backend,
        {
          lat: current.lat,
          lng: current.lng,
          maxViews: multi.maxViews,
          headingCount: multi.headingCount,
          pitchCount: multi.pitchCount,
          radiusM: multi.radiusM,
          confPercentile: multi.confPercentile,
          ensurePercentile: multi.ensurePercentile,
          farClipM: multi.farClipM,
          heightClipM: multi.heightClipM,
          edgeFactor: multi.edgeFactor,
          discontinuityRatio: multi.discontinuityRatio,
          cutStretch: multi.cutStretch,
          frontDiscontinuity: multi.frontDiscontinuity,
          groundCap: multi.groundCap,
          cameraHeightM: multi.cameraHeightM,
          tsdfVoxel: multi.tsdfVoxel,
          fov: multi.fov,
          processRes: multi.processRes,
          processResMethod: multi.processResMethod,
          useRayPose: multi.useRayPose,
          refViewStrategy: multi.refViewStrategy,
          enhanceInput: multi.enhanceInput,
          enhanceMode: multi.enhanceMode,
          dropSky: multi.dropSky,
          filterBlackBg: multi.filterBlackBg,
          filterWhiteBg: multi.filterWhiteBg,
          anchorGps: multi.anchorGps,
          rawMode: multi.rawMode,
          removeObjects: multi.removeObjects,
          removeClasses: multi.removeClasses,
          inpaint: multi.inpaint,
          method,
          depthModel: multi.depthModel || null,
        },
        (p) => setProgress(p),
      );
      setSceneHeadingDeg(facingRef.current);
      setMeshGlb(glbUrl(backend, meta));
      setMeshSplat(splatUrl(backend, meta));
      setScenePanos((meta as { panos?: PanoInfo[] }).panos);
      setMode("mesh");
      {
        const vp = meta.viewpoints ?? "?";
        const cap =
          meta.views_capped && meta.requested_views
            ? `（要求${meta.requested_views}→自動制限）`
            : "";
        say(
          `高精度3D化完了: ${vp}地点${cap} / ${meta.images_used ?? "?"}枚 / ${
            meta.vertex_count ?? "?"
          }頂点。WASDで歩けます`,
        );
      }
    } catch (err) {
      say((err as Error).message, true);
    } finally {
      setBuilding3d(false);
    }
  }, [current, backend, say, multi]);

  // 今のパノラマを高精細版に作り直してテクスチャを差し替える（その場で見回せる）。
  const handleEnhanceView = useCallback(async () => {
    if (!current) return;
    setEnhancing(true);
    say("このパノラマを高精細化中（26枚を狭角ズームで実解像度取得）...");
    try {
      const pano = await fetchPano(backend, { panoId: current.panoId, hi: true });
      setEquirect(pano.equirect);
      say("高精細パノラマに切り替えました（そのまま見回せます）");
    } catch (e) {
      say(`高精細化に失敗: ${e}`, true);
    } finally {
      setEnhancing(false);
    }
  }, [current, backend, say]);

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
            <>
              <button
                type="button"
                className="primaryWide pano"
                onClick={() => handle3DMulti("panorama")}
                disabled={!current || building3d}
                title="この1地点の360°を生成AI(LaMa)で穴埋めし、隙間のない球面空間を作って見回せます（推奨）"
              >
                {building3d ? "生成中..." : "● パノラマ生成3D（推奨・隙間なし）"}
              </button>
              <button
                type="button"
                className="primaryWide"
                onClick={() => handle3DMulti("mesh")}
                disabled={!current || building3d}
                title="周辺の複数Street View地点を集め、DA3マルチビューで整合したメッシュを作ります"
              >
                {building3d ? "生成中..." : "★ マルチビュー3D化"}
              </button>
              <button
                type="button"
                className="primaryWide"
                onClick={() => handle3DMulti("colliders")}
                disabled={!current || building3d}
                title="セマンティック検出＋既知カメラ高さで、床/壁/物体のコライダー(箱/円柱)を生成（深度不要・実験）"
              >
                {building3d ? "生成中..." : "🧱 意味コライダー化（実験）"}
              </button>
              <button
                type="button"
                className="primaryWide"
                onClick={() => handle3DMulti("gaussian")}
                disabled={!current || building3d}
                title="DA3点群を3D Gaussian Splatting(.splat)へ初期化し、写実 splat をアプリ内で観賞（学習なし・実験）"
              >
                {building3d ? "生成中..." : "✨ 3DGS（写実splat・実験）"}
              </button>
            </>
          )}

          <button
            type="button"
            className="primaryWide"
            onClick={handleEnhanceView}
            disabled={!current || enhancing}
            title="今のパノラマの26枚を狭角ズーム(実解像度)で取り直して高精細化し、その場で見回せます"
          >
            {enhancing ? "高精細化中..." : "🔍 このパノラマを高精細化"}
          </button>

          <div className="grid2 row2">
            <button
              type="button"
              className="ghost"
              onClick={() => setMode("pano")}
              disabled={mode === "pano"}
            >
              パノラマに戻る
            </button>
          </div>

          {config?.multiview_available && (
            <a
              className="galleryLink"
              href={`${backend.replace(/\/$/, "")}/gallery/index.html`}
              target="_blank"
              rel="noreferrer"
              title="各再構成手法(TSDF/Poisson/平滑化ほか)の比較ギャラリーを開く（verify/run_experiments.sh の出力）"
            >
              🖼 3D品質 比較ギャラリーを開く
            </a>
          )}

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

          {/* GPU メモリ使用量（生成中の挙動確認用） */}
          {gpu?.available && (
            <div className="gpuStat" title={
              `${gpu.name ?? "GPU"} / このプロセス確保: ` +
              `${gpu.torch_reserved_mb ?? 0}MB（割当 ${gpu.torch_allocated_mb ?? 0}MB）`
            }>
              <div className="gpuStatHead">
                <span>
                  🖥 GPU メモリ
                  <button
                    type="button"
                    className="gpuFreeBtn"
                    onClick={handleFreeGpu}
                    disabled={freeing || building3d}
                    title="ロード済みモデル(DA3/LaMa等)を解放してGPUメモリを空ける"
                  >
                    {freeing ? "解放中…" : "解放"}
                  </button>
                </span>
                <span>
                  {((gpu.used_mb ?? 0) / 1024).toFixed(1)} /{" "}
                  {((gpu.total_mb ?? 0) / 1024).toFixed(1)} GB（{gpu.used_pct ?? 0}%）
                  {gpu.util_pct != null ? ` · 使用率 ${gpu.util_pct}%` : ""}
                </span>
              </div>
              <div className="gpuBar">
                <div
                  className="gpuBarFill"
                  style={{ width: `${Math.min(100, gpu.used_pct ?? 0)}%` }}
                />
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
              {config?.multiview_available && (
                <>
                  <h2 className="settingsHead">★ 高精度3D化（マルチビュー）</h2>
                  <label className="field">
                    深度モデル（カメラ対応必須）
                    <select
                      value={multi.depthModel}
                      onChange={(e) => setMultiParam("depthModel", e.target.value)}
                      disabled={building3d}
                    >
                      {(config?.multiview_model_presets ?? []).map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="checkField">
                    <input
                      type="checkbox"
                      checked={multi.enhanceInput}
                      onChange={(e) => setMultiParam("enhanceInput", e.target.checked)}
                      disabled={building3d}
                    />
                    入力画像を高精細化（DA3前）
                  </label>
                  {multi.enhanceInput && (
                    <label className="field">
                      高精細化の方式
                      <select
                        value={multi.enhanceMode}
                        onChange={(e) =>
                          setMultiParam(
                            "enhanceMode",
                            e.target.value as "light" | "esrgan",
                          )
                        }
                        disabled={building3d}
                      >
                        <option value="light">軽量（ノイズ除去＋シャープ・高速）</option>
                        <option value="esrgan">
                          Real-ESRGAN（超解像・要process_res↑・重い）
                        </option>
                      </select>
                    </label>
                  )}
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
                      上下の段数 (1–5)
                      <input
                        type="number"
                        min={1}
                        max={5}
                        step={1}
                        value={multi.pitchCount}
                        onChange={(e) =>
                          setMultiParam("pitchCount", Number(e.target.value))
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
                      遠方クリップ (m, 0=無効)
                      <input
                        type="number"
                        min={0}
                        max={500}
                        step={5}
                        value={multi.farClipM}
                        onChange={(e) =>
                          setMultiParam("farClipM", Number(e.target.value))
                        }
                        disabled={building3d}
                      />
                    </label>
                    <label className="field">
                      頭上クリップ (m, 0=無効)
                      <input
                        type="number"
                        min={0}
                        max={200}
                        step={5}
                        value={multi.heightClipM}
                        onChange={(e) =>
                          setMultiParam("heightClipM", Number(e.target.value))
                        }
                        disabled={building3d}
                      />
                    </label>
                    <label className="field">
                      スパイク除去 辺÷深度 (0=無効)
                      <input
                        type="number"
                        min={0}
                        max={3}
                        step={0.05}
                        value={multi.edgeFactor}
                        onChange={(e) =>
                          setMultiParam("edgeFactor", Number(e.target.value))
                        }
                        disabled={building3d}
                      />
                    </label>
                    <label className="field">
                      不連続しきい値
                      <input
                        type="number"
                        min={0.01}
                        max={1}
                        step={0.01}
                        value={multi.discontinuityRatio}
                        onChange={(e) =>
                          setMultiParam("discontinuityRatio", Number(e.target.value))
                        }
                        disabled={building3d}
                      />
                    </label>
                    <label className="checkField">
                      <input
                        type="checkbox"
                        checked={multi.cutStretch}
                        onChange={(e) =>
                          setMultiParam("cutStretch", e.target.checked)
                        }
                        disabled={building3d}
                      />
                      引き伸ばし三角をカット（パノラマ・縁の伸び防止）
                    </label>
                    <label className="field">
                      カット強度 front (小=強く切る)
                      <input
                        type="number"
                        min={0.05}
                        max={2}
                        step={0.05}
                        value={multi.frontDiscontinuity}
                        onChange={(e) =>
                          setMultiParam("frontDiscontinuity", Number(e.target.value))
                        }
                        disabled={building3d || !multi.cutStretch}
                      />
                    </label>
                    <label className="checkField">
                      <input
                        type="checkbox"
                        checked={multi.groundCap}
                        onChange={(e) =>
                          setMultiParam("groundCap", e.target.checked)
                        }
                        disabled={building3d}
                      />
                      床平面キャップ（RANSAC・床のせり上がり/傾きを除去）
                    </label>
                    <label className="field">
                      カメラ高さ m (標尺=実寸基準)
                      <input
                        type="number"
                        min={0.5}
                        max={5}
                        step={0.1}
                        value={multi.cameraHeightM}
                        onChange={(e) =>
                          setMultiParam("cameraHeightM", Number(e.target.value))
                        }
                        disabled={building3d || !multi.groundCap}
                      />
                    </label>
                    <label className="field">
                      画角 fov (60–120)
                      <input
                        type="number"
                        min={60}
                        max={120}
                        step={5}
                        value={multi.fov}
                        onChange={(e) =>
                          setMultiParam("fov", Number(e.target.value))
                        }
                        disabled={building3d}
                      />
                    </label>
                    <label className="field">
                      TSDFボクセル (m)
                      <input
                        type="number"
                        min={0.04}
                        max={0.4}
                        step={0.01}
                        value={multi.tsdfVoxel}
                        onChange={(e) =>
                          setMultiParam("tsdfVoxel", Number(e.target.value))
                        }
                        disabled={building3d}
                      />
                    </label>
                    <label className="field">
                      処理解像度 (168–1008)
                      <div className="presetChips">
                        {[280, 364, 504, 728].map((r) => (
                          <button
                            key={r}
                            type="button"
                            className={
                              multi.processRes === r ? "chip chipOn" : "chip"
                            }
                            onClick={() => setMultiParam("processRes", r)}
                            disabled={building3d}
                          >
                            {r}
                          </button>
                        ))}
                      </div>
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
                        checked={multi.rawMode}
                        onChange={(e) =>
                          setMultiParam("rawMode", e.target.checked)
                        }
                        disabled={building3d}
                      />
                      生データ（全フィルタ無効）
                    </label>
                    <label className="checkField">
                      <input
                        type="checkbox"
                        checked={multi.removeObjects}
                        onChange={(e) =>
                          setMultiParam("removeObjects", e.target.checked)
                        }
                        disabled={building3d}
                      />
                      物体を除去（YOLO）
                    </label>
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
                  {multi.removeObjects && (
                    <>
                      <label className="field">
                        除去クラス（カンマ区切り: person, car, bicycle …）
                        <input
                          type="text"
                          value={multi.removeClasses}
                          onChange={(e) =>
                            setMultiParam("removeClasses", e.target.value)
                          }
                          disabled={building3d}
                        />
                      </label>
                      <label className="checkField">
                        <input
                          type="checkbox"
                          checked={multi.inpaint}
                          onChange={(e) =>
                            setMultiParam("inpaint", e.target.checked)
                          }
                          disabled={building3d}
                        />
                        消した跡を生成AIで埋める（LaMa）
                      </label>
                    </>
                  )}
                  {(() => {
                    const MAX_IMG = 80; // バックエンドの自動制限と一致
                    const perVp = multi.headingCount * multi.pitchCount;
                    const budgetViews = Math.max(1, Math.floor(MAX_IMG / perVp));
                    const effViews = Math.min(multi.maxViews, budgetViews);
                    const capped = effViews < multi.maxViews;
                    return (
                      <p className={capped ? "hint warn" : "hint"}>
                        予定: <b>{effViews}地点</b> × {multi.headingCount}方向 ×{" "}
                        {multi.pitchCount}段 = <b>{effViews * perVp}枚</b>
                        {capped
                          ? ` ／ ⚠️ 地点数${multi.maxViews}は上限${MAX_IMG}枚を超えるため自動で${effViews}地点に制限されます`
                          : ` （上限${MAX_IMG}枚）`}
                        。実際の地点数は周辺のStreet View数により更に少なくなることがあります。
                      </p>
                    );
                  })()}
                  <p className="hint">
                    地点数×方向数×上下段数 の画像をDA3に一括投入して整合。<b>地点数=1</b>なら
                    今いる1地点の全周だけで作ります（最もキレイ）。<b>上下の段数</b>は天地の抜けを
                    埋めるピッチ方向の枚数（3=下/水平/上）。<b>収集半径</b>は地点数≥2のときだけ効きます。
                    <b>視点位置をGPSで固定</b>は複数視点時のズレ（ぐちゃぐちゃ）を防ぐ最重要オプション。
                    <b>レイベースのポーズ推定</b>は各画素レイから RANSAC でカメラを解く別法。
                    <b>遠方クリップ＝0で無効</b>（奥まで表示）。値を入れると中心からその距離より
                    遠い面を除去、頭上クリップは天井側を削ります（0で無効）。
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
          <SceneViewer
            glbUrl={meshGlb}
            splatUrl={meshSplat}
            initialHeadingDeg={sceneHeadingDeg}
            panos={scenePanos}
          />
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
