#!/usr/bin/env node
/**
 * 比較ギャラリー生成: manifest の各バリアントGLBを orbit.html で
 * 「外周＋内部見回し」レンダリング→GIF化し、HTMLギャラリーにまとめる。
 *
 * 使い方: node gallery.mjs <manifest.json> [outDir]
 * manifest.json: [{ scene_id, label, glb_url, faces, verts, build_seconds, method }]
 *   glb_url は BACKEND からの相対(/scenes/...)。
 */
import { chromium } from "playwright";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = process.env.FRONTEND || "http://localhost:3000";
const BACKEND = process.env.BACKEND || "http://localhost:8000";

const manifestPath = process.argv[2];
const outDir = process.argv[3] || path.join(HERE, "gallery");
const items = JSON.parse(readFileSync(manifestPath, "utf8"));
mkdirSync(outDir, { recursive: true });

async function renderItem(page, it) {
  const glb = `${BACKEND}${it.glb_url}`;
  await page.goto(`${FRONTEND}/orbit.html?glb=${encodeURIComponent(glb)}`, { waitUntil: "domcontentloaded", timeout: 60000 });
  let ok = false;
  for (let i = 0; i < 90; i++) {
    await page.waitForTimeout(1000);
    const st = await page.evaluate(() => ({ r: window.ready, e: window.loadError }));
    if (st.e) { it.error = "load: " + st.e; return; }
    if (st.r) { ok = true; break; }
  }
  if (!ok) { it.error = "not ready"; return; }
  it.radius = await page.evaluate(() => window.radius);
  let idx = 0;
  const F = 8;
  const shot = async () => {
    const f = path.join(outDir, `${it.id}_${String(idx++).padStart(2, "0")}.png`);
    try { await page.screenshot({ path: f, timeout: 20000 }); } catch { it.shotFail = (it.shotFail || 0) + 1; }
  };
  for (let i = 0; i < F; i++) { await page.evaluate((a) => window.setView(a, 14, 1.25), (360 / F) * i); await page.waitForTimeout(300); await shot(); }
  for (let i = 0; i < F; i++) { await page.evaluate((a) => window.setInterior(a, -3), (360 / F) * i); await page.waitForTimeout(300); await shot(); }
  const gif = path.join(outDir, `${it.id}.gif`);
  const ff = spawnSync("ffmpeg", ["-y", "-framerate", "2", "-i", path.join(outDir, `${it.id}_%02d.png`), "-vf", "scale=440:-1", gif], { encoding: "utf8" });
  it.gif = ff.status === 0 ? path.basename(gif) : null;
}

(async () => {
  const browser = await chromium.launch({ headless: true, args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"] });
  const page = await (await browser.newContext({ viewport: { width: 900, height: 600 } })).newPage();
  for (const it of items) {
    process.stdout.write(`render ${it.label} (${it.id}) ... `);
    try { await renderItem(page, it); console.log(it.gif ? "ok" : "FAIL " + (it.error || "")); }
    catch (e) { it.error = String(e).split("\n")[0]; console.log("ERR " + it.error); }
  }
  await browser.close();

  const cards = items.map((it) => `
    <div class="card">
      <h3>${it.label}</h3>
      ${it.gif ? `<img src="${it.gif}" />` : `<div class="err">${it.error || "no gif"}</div>`}
      <div class="meta">method: ${it.method || it.variant || "-"} / faces: ${it.faces ?? "-"} / verts: ${it.verts ?? "-"}<br/>build: ${it.build_seconds ?? "-"}s${it.ground_tilt_corrected_deg != null ? ` / tilt直し:${it.ground_tilt_corrected_deg}°` : ""}${it.spike_faces_removed ? ` / spike除去:${it.spike_faces_removed}` : ""}</div>
    </div>`).join("\n");
  const html = `<!doctype html><meta charset="utf-8"><title>3D品質 比較ギャラリー</title>
<style>body{font-family:sans-serif;background:#111;color:#eee;margin:16px}h1{font-size:18px}
.grid{display:flex;flex-wrap:wrap;gap:14px}.card{background:#1c1c22;border-radius:8px;padding:10px;width:460px}
.card h3{margin:4px 0;font-size:15px;color:#7fd}.card img{width:100%;border-radius:4px;background:#aac4d8}
.meta{font-size:12px;color:#9aa;margin-top:6px;line-height:1.5}.err{color:#f88;padding:30px;text-align:center}</style>
<h1>3D品質 比較ギャラリー — 前半=外周(立体の証明) / 後半=内部見回し(歩く体験)</h1>
<div class="grid">${cards}</div>`;
  writeFileSync(path.join(outDir, "index.html"), html);
  console.log("gallery: " + path.join(outDir, "index.html"));
})();
