#!/usr/bin/env node
/**
 * 3D化 検証ハーネス（Claude が繰り返し回せる用）。
 *
 * フロントUIを Playwright で操作して 3D化 を実行 → 生成された GLB を orbit.html で
 * 全方位レンダリング → ターンテーブル GIF と summary.json を残す。
 *
 * 使い方:
 *   cd verify && npm install && npx playwright install chromium   # 初回のみ
 *   node verify3d.mjs <lat> <lng> [--mode multi|simple] [--frames 8] [--out DIR]
 *
 * 前提: フロント(:3000) と バックエンド(:8000) が起動中（バックエンドは編集による
 *   --reload 再起動で実行中ジョブが落ちるので、検証中はコードを触らないこと）。
 * 環境変数: FRONTEND, BACKEND, GEN_TIMEOUT(秒) で上書き可。
 */
import { chromium } from "playwright";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import path from "node:path";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = process.env.FRONTEND || "http://localhost:3000";
const BACKEND = process.env.BACKEND || "http://localhost:8000";
const GEN_TIMEOUT = Number(process.env.GEN_TIMEOUT || 240);

function parseArgs() {
  const a = process.argv.slice(2);
  const pos = a.filter((x) => !x.startsWith("--"));
  const get = (k, d) => { const i = a.indexOf("--" + k); return i >= 0 ? a[i + 1] : d; };
  return {
    lat: Number(pos[0] ?? 35.659350),
    lng: Number(pos[1] ?? 139.700962),
    mode: get("mode", "multi"),
    frames: Number(get("frames", 8)),
    el: Number(get("el", 15)),
    out: get("out", path.join(HERE, "out")),
  };
}

const log = (m) => console.log(`[verify] ${m}`);

async function main() {
  const args = parseArgs();
  rmSync(args.out, { recursive: true, force: true });
  mkdirSync(args.out, { recursive: true });
  const t0 = Date.now();
  const summary = { ok: false, args, errors: [] };

  const browser = await chromium.launch({
    headless: true,
    args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"],
  });
  const ctx = await browser.newContext({ viewport: { width: 1024, height: 700 } });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => summary.errors.push("PAGEERR " + String(e).slice(0, 200)));
  page.on("console", (m) => { if (m.type() === "error") summary.errors.push("CONSOLE " + m.text().slice(0, 200)); });

  try {
    // 1) フロントを開いて地点ドロップを待つ
    const url = `${FRONTEND}/?lat=${args.lat}&lng=${args.lng}&h=0`;
    log("open " + url);
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60000 });
    const name = args.mode === "simple" ? /この地点を3D化/ : /高精度3D化/;
    const btn = page.getByRole("button", { name });
    let ready = false;
    for (let i = 0; i < 45; i++) { await page.waitForTimeout(1000); if (await btn.isEnabled().catch(() => false)) { ready = true; break; } }
    if (!ready) throw new Error("地点が読み込めず3D化ボタンが有効化されませんでした（Maps/Street View）");

    // 2) 3D化 実行 → 完了/失敗を待つ
    const before = (await fetchScenes()).length;
    log(`generate (mode=${args.mode}) ...`);
    await btn.click();
    const status = () => page.locator("p.status").first().innerText().catch(() => "");
    const gt0 = Date.now();
    let finalStatus = "";
    while ((Date.now() - gt0) / 1000 < GEN_TIMEOUT) {
      await page.waitForTimeout(2000);
      finalStatus = await status();
      if (/完了/.test(finalStatus)) break;
      if (/失敗|エラー/.test(finalStatus)) throw new Error("生成失敗: " + finalStatus);
    }
    summary.status = finalStatus;
    summary.genSeconds = Math.round((Date.now() - gt0) / 1000);
    if (!/完了/.test(finalStatus)) throw new Error(`生成がタイムアウト(${GEN_TIMEOUT}s): ${finalStatus}`);
    log(`generated in ${summary.genSeconds}s: ${finalStatus}`);

    // 3) 生成された最新シーンを取得
    const scenes = await fetchScenes();
    if (scenes.length <= before && scenes.length === 0) throw new Error("シーンが見つかりません");
    const scene = scenes[scenes.length - 1];
    Object.assign(summary, {
      sceneId: scene.id,
      vertexCount: scene.vertex_count,
      faceCount: scene.face_count,
      viewpoints: scene.viewpoints,
      imagesUsed: scene.images_used,
      scaleSource: scene.scale_source,
    });
    const glb = `${BACKEND}${scene.glb_url}`;
    log(`scene ${scene.id} verts=${scene.vertex_count} viewpoints=${scene.viewpoints}`);

    // 4) orbit.html で全方位レンダリング
    await page.goto(`${FRONTEND}/orbit.html?glb=${encodeURIComponent(glb)}`, { waitUntil: "domcontentloaded", timeout: 60000 });
    let ok = false;
    for (let i = 0; i < 90; i++) {
      await page.waitForTimeout(1000);
      const st = await page.evaluate(() => ({ ready: window.ready, err: window.loadError }));
      if (st.err) throw new Error("GLB読込失敗: " + st.err);
      if (st.ready) { ok = true; break; }
    }
    if (!ok) throw new Error("orbit.html が GLB を読み込めませんでした");
    summary.radius = await page.evaluate(() => window.radius);

    // 外周から回す(orbit=立体の証明) → 内部から見回す(interior=歩く体験)
    let idx = 0;
    const frames = [];
    const grab = async (tag) => {
      const f = path.join(args.out, `frame_${String(idx++).padStart(2, "0")}.png`);
      try { await page.screenshot({ path: f, timeout: 20000 }); frames.push(f); } catch { summary.errors.push("shot fail " + tag); }
    };
    for (let i = 0; i < args.frames; i++) {
      await page.evaluate(([a, e]) => window.setView(a, e, 1.3), [(360 / args.frames) * i, args.el]);
      await page.waitForTimeout(350); await grab("orbit" + i);
    }
    for (let i = 0; i < args.frames; i++) {
      await page.evaluate((a) => window.setInterior(a, -3), (360 / args.frames) * i);
      await page.waitForTimeout(350); await grab("interior" + i);
    }
    summary.frames = frames.length;
    log(`rendered ${frames.length} frames (orbit + interior)`);

    // 5) GIF 生成（ffmpeg があれば）。前半=外周, 後半=内部見回し。
    const gif = path.join(args.out, "turntable.gif");
    const ff = spawnSync("ffmpeg", ["-y", "-framerate", "2", "-i", path.join(args.out, "frame_%02d.png"), "-vf", "scale=520:-1", gif], { encoding: "utf8" });
    if (ff.status === 0) { summary.gif = gif; log("gif: " + gif); }
    else summary.errors.push("ffmpeg unavailable or failed");

    summary.ok = frames.length > 0;
  } catch (e) {
    summary.error = String(e.message || e);
    log("FAILED: " + summary.error);
  } finally {
    summary.durationSec = Math.round((Date.now() - t0) / 1000);
    writeFileSync(path.join(args.out, "summary.json"), JSON.stringify(summary, null, 2));
    await browser.close();
  }
  log("SUMMARY " + JSON.stringify({ ok: summary.ok, status: summary.status, verts: summary.vertexCount, viewpoints: summary.viewpoints, gif: summary.gif, error: summary.error, errors: summary.errors.length }));
  process.exit(summary.ok ? 0 : 1);
}

async function fetchScenes() {
  try { const r = await fetch(`${BACKEND}/api/scenes`); const j = await r.json(); return j.scenes || []; }
  catch { return []; }
}

main();
