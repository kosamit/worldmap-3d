import * as THREE from "three";
import { PlayerController } from "./PlayerController.js";
import { loadGLB } from "./SceneLoader.js";
import {
  fetchScenes,
  reconstructFromUpload,
  reconstructPanorama,
} from "./api.js";

// ---------- Three.js セットアップ ----------
const canvasWrap = document.getElementById("canvas-wrap");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
canvasWrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101418);

const camera = new THREE.PerspectiveCamera(70, 1, 0.05, 2000);

scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 1.4));
const dir = new THREE.DirectionalLight(0xffffff, 1.0);
dir.position.set(5, 10, 7);
scene.add(dir);

// 向き確認用の薄いグリッド
const grid = new THREE.GridHelper(60, 60, 0x335577, 0x223344);
grid.material.opacity = 0.3;
grid.material.transparent = true;
scene.add(grid);

const player = new PlayerController(camera, renderer.domElement);
player.object.position.set(0, 1.7, 3);
scene.add(player.object);

let currentScene = null; // 現在ロード中の glb グループ

function resize() {
  const w = canvasWrap.clientWidth;
  const h = canvasWrap.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// ---------- メインループ ----------
const clock = new THREE.Clock();
const coordsEl = document.getElementById("coords");
renderer.setAnimationLoop(() => {
  const delta = Math.min(clock.getDelta(), 0.1);
  player.update(delta);
  const p = player.object.position;
  coordsEl.textContent = `x:${p.x.toFixed(1)} y:${p.y.toFixed(1)} z:${p.z.toFixed(1)}`;
  renderer.render(scene, camera);
});

// ---------- PointerLock UI ----------
const blocker = document.getElementById("blocker");
canvasWrap.addEventListener("click", () => player.lock());
player.controls.addEventListener("lock", () => blocker.classList.add("hidden"));
player.controls.addEventListener("unlock", () =>
  blocker.classList.remove("hidden"),
);

// ---------- DOM 参照 ----------
const baseInput = document.getElementById("backend-url");
const statusEl = document.getElementById("status");
const sceneListEl = document.getElementById("scene-list");

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.style.color = isError ? "#ff6b6b" : "#9fe6a0";
}

function base() {
  return baseInput.value.trim() || "http://localhost:8000";
}

// ---------- シーンのロード ----------
async function loadScene(meta) {
  setStatus(`読み込み中: ${meta.id} ...`);
  try {
    const url = `${base().replace(/\/+$/, "")}${meta.glb_url}`;
    const group = await loadGLB(url);
    if (currentScene) scene.remove(currentScene);
    currentScene = group;
    scene.add(group);
    // パノラマは原点を取り囲むので中心に立つ
    player.object.position.set(0, 1.7, 0);
    setStatus(`表示中: ${meta.id} (${meta.source}, ${meta.vertex_count} 頂点)`);
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function refreshScenes() {
  setStatus("シーン一覧を取得中 ...");
  try {
    const scenes = await fetchScenes(base());
    renderSceneList(scenes);
    setStatus(`${scenes.length} 件のシーン`);
  } catch (err) {
    setStatus(err.message, true);
  }
}

function renderSceneList(scenes) {
  sceneListEl.innerHTML = "";
  if (scenes.length === 0) {
    const li = document.createElement("li");
    li.textContent = "（まだありません）";
    li.className = "empty";
    sceneListEl.appendChild(li);
    return;
  }
  for (const meta of scenes) {
    const li = document.createElement("li");
    const label = meta.location
      ? `${meta.location.lat.toFixed(4)}, ${meta.location.lng.toFixed(4)}`
      : meta.source;
    li.textContent = `${meta.id} — ${label}`;
    li.addEventListener("click", () => loadScene(meta));
    sceneListEl.appendChild(li);
  }
}

// ---------- アップロードから3D化 ----------
document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = document.getElementById("upload-file").files[0];
  const fov = document.getElementById("upload-fov").value || 75;
  if (!file) {
    setStatus("画像ファイルを選択してください", true);
    return;
  }
  setStatus("3D化中（初回はモデルDLで時間がかかります）...");
  try {
    const meta = await reconstructFromUpload(base(), file, fov);
    await refreshScenes();
    await loadScene(meta);
  } catch (err) {
    setStatus(err.message, true);
  }
});

// ---------- Street View から 360° パノラマ3D化 ----------
document
  .getElementById("streetview-form")
  .addEventListener("submit", async (e) => {
    e.preventDefault();
    const params = {
      lat: document.getElementById("sv-lat").value,
      lng: document.getElementById("sv-lng").value,
      num_views: document.getElementById("sv-views").value || 8,
      pitch: document.getElementById("sv-pitch").value,
      fov: document.getElementById("sv-fov").value,
    };
    if (!params.lat || !params.lng) {
      setStatus("緯度・経度を入力してください", true);
      return;
    }
    setStatus(`360°撮影して3D化中（${params.num_views}視点ぶん推論、数十秒〜）...`);
    try {
      const meta = await reconstructPanorama(base(), params);
      await refreshScenes();
      await loadScene(meta);
    } catch (err) {
      setStatus(err.message, true);
    }
  });

document
  .getElementById("refresh-btn")
  .addEventListener("click", () => refreshScenes());

// 初回ロード
refreshScenes();
