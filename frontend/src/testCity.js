import * as THREE from "three";

const GROUND_SIZE = 200;
const BUILDING_COLORS = [0xb0bec5, 0x90a4ae, 0xcfd8dc, 0xa1887f, 0xbcaaa4];

/**
 * テスト用の街を組み立てる。
 * フラットな地面 + グリッド + ランダム配置のビル(箱) + 簡単なライティング。
 * フェーズ3で生成済み3Dデータに置き換える想定の、プレースホルダ的シーン。
 */
export function buildTestCity(scene) {
  scene.background = new THREE.Color(0x87ceeb);
  scene.fog = new THREE.Fog(0x87ceeb, 60, 180);

  // ライト
  const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
  hemi.position.set(0, 50, 0);
  scene.add(hemi);

  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(30, 60, 20);
  scene.add(sun);

  // 地面
  const groundGeo = new THREE.PlaneGeometry(GROUND_SIZE, GROUND_SIZE);
  const groundMat = new THREE.MeshLambertMaterial({ color: 0x55683a });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  // グリッド（歩いている感を出す）
  const grid = new THREE.GridHelper(GROUND_SIZE, GROUND_SIZE / 4, 0x000000, 0x222222);
  grid.material.opacity = 0.25;
  grid.material.transparent = true;
  scene.add(grid);

  // ビル群（決定論的な疑似乱数で配置 → 毎回同じ街）
  const buildings = new THREE.Group();
  let seed = 1337;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };

  for (let i = 0; i < 60; i++) {
    const w = 4 + rand() * 8;
    const d = 4 + rand() * 8;
    const h = 6 + rand() * 30;
    const geo = new THREE.BoxGeometry(w, h, d);
    const color = BUILDING_COLORS[Math.floor(rand() * BUILDING_COLORS.length)];
    const mat = new THREE.MeshLambertMaterial({ color });
    const mesh = new THREE.Mesh(geo, mat);

    // 中央付近(プレイヤー初期位置)は空けておく
    let x, z;
    do {
      x = (rand() - 0.5) * GROUND_SIZE * 0.85;
      z = (rand() - 0.5) * GROUND_SIZE * 0.85;
    } while (Math.hypot(x, z) < 12);

    mesh.position.set(x, h / 2, z);
    buildings.add(mesh);
  }
  scene.add(buildings);

  return { ground, buildings };
}
