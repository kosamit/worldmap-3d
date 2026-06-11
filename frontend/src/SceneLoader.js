import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const loader = new GLTFLoader();

/**
 * glb/gltf を読み込み THREE.Group を返す。
 * 頂点カラー付きメッシュを両面表示にし、ライト不要で見えるようにする。
 */
export function loadGLB(url) {
  return new Promise((resolve, reject) => {
    loader.load(
      url,
      (gltf) => {
        const root = gltf.scene;
        root.traverse((obj) => {
          if (obj.isMesh) {
            obj.material.side = THREE.DoubleSide;
            obj.material.vertexColors = true;
            obj.frustumCulled = false;
          }
        });
        resolve(root);
      },
      undefined,
      (err) => reject(new Error(`glb の読み込みに失敗: ${err?.message ?? err}`)),
    );
  });
}
