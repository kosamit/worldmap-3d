import * as THREE from "three";
import { PlayerController } from "./PlayerController.js";
import { buildTestCity } from "./testCity.js";

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  70,
  window.innerWidth / window.innerHeight,
  0.1,
  1000,
);

buildTestCity(scene);

const player = new PlayerController(camera, renderer.domElement);
scene.add(player.object);

// --- PointerLock の開始/終了 UI ---
const blocker = document.getElementById("blocker");
const instructions = document.getElementById("instructions");

instructions.addEventListener("click", () => player.lock());
player.controls.addEventListener("lock", () => blocker.classList.add("hidden"));
player.controls.addEventListener("unlock", () =>
  blocker.classList.remove("hidden"),
);

// --- HUD ---
const coordsEl = document.getElementById("coords");
const fpsEl = document.getElementById("fps");
let frameCount = 0;
let fpsTimer = 0;

// --- リサイズ対応 ---
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- メインループ ---
const clock = new THREE.Clock();

function animate() {
  const delta = Math.min(clock.getDelta(), 0.1); // 大きなフレーム飛びを抑制

  player.update(delta);

  // HUD 更新
  const p = player.object.position;
  coordsEl.textContent = `x: ${p.x.toFixed(1)}  z: ${p.z.toFixed(1)}`;
  frameCount++;
  fpsTimer += delta;
  if (fpsTimer >= 0.5) {
    fpsEl.textContent = `${Math.round(frameCount / fpsTimer)} fps`;
    frameCount = 0;
    fpsTimer = 0;
  }

  renderer.render(scene, camera);
}

renderer.setAnimationLoop(animate);
