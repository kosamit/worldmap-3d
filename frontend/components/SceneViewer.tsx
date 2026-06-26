"use client";

// 生成した glb を読み込み、PointerLock + WASD で歩ける R3F ビューワー。
// 写実性のため無光沢(MeshBasic)で写真色をそのまま表示し、three-mesh-bvh による
// 衝突判定（地面追従＋壁すり抜け防止）を行う。

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { PointerLockControls, useGLTF } from "@react-three/drei";
import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from "react";
import * as THREE from "three";
import {
  computeBoundsTree,
  disposeBoundsTree,
  acceleratedRaycast,
} from "three-mesh-bvh";

// three-mesh-bvh を three に組み込み（高速レイキャスト）。
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(THREE.BufferGeometry.prototype as any).computeBoundsTree = computeBoundsTree;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(THREE.BufferGeometry.prototype as any).disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

const EYE_HEIGHT = 1.7;
const WALK_SPEED = 40;
const RUN_MULTIPLIER = 2.2;
const DAMPING = 8.0;
const GRAVITY = 25.0;
const JUMP_VELOCITY = 8.5;
const PLAYER_RADIUS = 0.45; // 壁との最小距離 (m)
const STEP_UP = 0.6; // 乗り越えられる段差 (m)

// 頂点色(sRGB バイト)を linear に変換して、出力時の linear→sRGB と往復させ、
// 写真そのままの色で表示する（glTF の COLOR_0 は linear 前提のため要補正）。
function convertVertexColorsToLinear(geom: THREE.BufferGeometry) {
  const attr = geom.getAttribute("color") as THREE.BufferAttribute | undefined;
  if (!attr) return;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if ((attr as any)._srgbFixed) return;
  const a = attr.array as ArrayLike<number> & { [i: number]: number };
  const norm = attr.normalized;
  for (let i = 0; i < a.length; i++) {
    let c = a[i] as number;
    if (norm) c /= 255;
    c = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    (a as { [i: number]: number })[i] = norm ? Math.round(c * 255) : c;
  }
  attr.needsUpdate = true;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (attr as any)._srgbFixed = true;
}

function Model({
  url,
  onReady,
}: {
  url: string;
  onReady: (meshes: THREE.Mesh[]) => void;
}) {
  const { scene } = useGLTF(url);
  const object = useMemo(() => {
    const cloned = scene.clone(true);
    const meshes: THREE.Mesh[] = [];
    cloned.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (mesh.isMesh) {
        convertVertexColorsToLinear(mesh.geometry);
        // 無光沢マテリアル：ライティングせず写真色をそのまま出す。
        mesh.material = new THREE.MeshBasicMaterial({
          vertexColors: true,
          side: THREE.DoubleSide,
        });
        mesh.frustumCulled = false;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (mesh.geometry as any).computeBoundsTree();
        meshes.push(mesh);
      }
    });
    onReady(meshes);
    return cloned;
  }, [scene, onReady]);
  return <primitive object={object} />;
}

interface KeyState {
  forward: boolean;
  backward: boolean;
  left: boolean;
  right: boolean;
  run: boolean;
}

function Player({
  onLockChange,
  coordsRef,
  colliderRef,
  initialHeadingDeg,
}: {
  onLockChange: (locked: boolean) => void;
  coordsRef: RefObject<HTMLDivElement | null>;
  colliderRef: RefObject<THREE.Mesh[]>;
  initialHeadingDeg?: number | null;
}) {
  const { camera } = useThree();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const controls = useRef<any>(null);

  // GPSアンカー世界は真北=-Z, 東=+X。パノラマで見ていた方位(時計回り,北=0)へ初期視線を向ける。
  const faceHeading = useCallback(() => {
    if (initialHeadingDeg == null) return;
    const h = (initialHeadingDeg * Math.PI) / 180;
    camera.lookAt(
      camera.position.x + Math.sin(h),
      camera.position.y,
      camera.position.z - Math.cos(h),
    );
  }, [camera, initialHeadingDeg]);
  const velocity = useRef(new THREE.Vector3());
  const direction = useRef(new THREE.Vector3());
  const canJump = useRef(false);
  const spawned = useRef(false);
  const lastGroundY = useRef<number | null>(null);
  const rayDown = useRef(new THREE.Raycaster());
  const rayMove = useRef(new THREE.Raycaster());
  const keys = useRef<KeyState>({
    forward: false,
    backward: false,
    left: false,
    right: false,
    run: false,
  });

  useEffect(() => {
    camera.position.set(0, EYE_HEIGHT, 3);
    faceHeading();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (rayDown.current as any).firstHitOnly = true;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (rayMove.current as any).firstHitOnly = true;

    const onKeyDown = (e: KeyboardEvent) => {
      switch (e.code) {
        case "KeyW":
        case "ArrowUp":
          keys.current.forward = true;
          break;
        case "KeyS":
        case "ArrowDown":
          keys.current.backward = true;
          break;
        case "KeyA":
        case "ArrowLeft":
          keys.current.left = true;
          break;
        case "KeyD":
        case "ArrowRight":
          keys.current.right = true;
          break;
        case "ShiftLeft":
        case "ShiftRight":
          keys.current.run = true;
          break;
        case "Space":
          if (canJump.current) {
            velocity.current.y = JUMP_VELOCITY;
            canJump.current = false;
          }
          break;
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      switch (e.code) {
        case "KeyW":
        case "ArrowUp":
          keys.current.forward = false;
          break;
        case "KeyS":
        case "ArrowDown":
          keys.current.backward = false;
          break;
        case "KeyA":
        case "ArrowLeft":
          keys.current.left = false;
          break;
        case "KeyD":
        case "ArrowRight":
          keys.current.right = false;
          break;
        case "ShiftLeft":
        case "ShiftRight":
          keys.current.run = false;
          break;
      }
    };
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("keyup", onKeyUp);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("keyup", onKeyUp);
    };
  }, [camera, faceHeading]);

  // 真下の地面の高さを返す（無ければ null）。
  const groundAt = (x: number, y: number, z: number): number | null => {
    const colliders = colliderRef.current;
    if (!colliders || colliders.length === 0) return null;
    rayDown.current.set(
      new THREE.Vector3(x, y + STEP_UP, z),
      new THREE.Vector3(0, -1, 0),
    );
    rayDown.current.far = 1000;
    const hits = rayDown.current.intersectObjects(colliders, false);
    return hits.length ? hits[0].point.y : null;
  };

  useFrame((_, rawDelta) => {
    const c = controls.current;
    if (!c) return;

    // 初回：地面の上にスポーンさせる。
    if (!spawned.current && colliderRef.current && colliderRef.current.length) {
      const g = groundAt(0, 200, 0);
      if (g != null) {
        camera.position.set(0, g + EYE_HEIGHT, 0);
        lastGroundY.current = g;
      }
      faceHeading();
      spawned.current = true;
    }

    if (!c.isLocked) return;
    const delta = Math.min(rawDelta, 0.1);
    const v = velocity.current;

    v.x -= v.x * DAMPING * delta;
    v.z -= v.z * DAMPING * delta;

    const k = keys.current;
    direction.current.z = Number(k.forward) - Number(k.backward);
    direction.current.x = Number(k.right) - Number(k.left);
    direction.current.normalize();

    const speed = WALK_SPEED * (k.run ? RUN_MULTIPLIER : 1);
    if (k.forward || k.backward) v.z += direction.current.z * speed * delta;
    if (k.left || k.right) v.x += direction.current.x * speed * delta;

    // --- 水平移動（壁との衝突判定つき） ---
    const before = camera.position.clone();
    c.moveRight(v.x * delta);
    c.moveForward(v.z * delta);
    const move = camera.position.clone().sub(before);
    move.y = 0;
    camera.position.copy(before); // いったん戻す

    const dist = move.length();
    if (dist > 1e-5) {
      const dir = move.clone().normalize();
      const colliders = colliderRef.current;
      let allowed = dist;
      if (colliders && colliders.length) {
        rayMove.current.set(before, dir);
        rayMove.current.far = dist + PLAYER_RADIUS;
        const hits = rayMove.current.intersectObjects(colliders, false);
        if (hits.length && hits[0].distance < dist + PLAYER_RADIUS) {
          allowed = Math.max(0, hits[0].distance - PLAYER_RADIUS);
          v.x *= 0.3;
          v.z *= 0.3;
        }
      }
      camera.position.addScaledVector(dir, allowed);
    }
    camera.position.y = before.y;

    // --- 垂直移動（重力＋地面追従） ---
    v.y -= GRAVITY * delta;
    camera.position.y += v.y * delta;

    const g = groundAt(camera.position.x, camera.position.y, camera.position.z);
    const groundY = g != null ? g : lastGroundY.current;
    if (groundY != null) {
      lastGroundY.current = groundY;
      const floorY = groundY + EYE_HEIGHT;
      if (camera.position.y < floorY) {
        v.y = 0;
        camera.position.y = floorY;
        canJump.current = true;
      }
    } else if (camera.position.y < EYE_HEIGHT) {
      // 地面が全く見つからない場合のフォールバック（落下防止）。
      v.y = 0;
      camera.position.y = EYE_HEIGHT;
      canJump.current = true;
    }

    if (coordsRef.current) {
      const p = camera.position;
      coordsRef.current.textContent = `x:${p.x.toFixed(1)} y:${p.y.toFixed(
        1,
      )} z:${p.z.toFixed(1)}`;
    }
  });

  return (
    <PointerLockControls
      ref={controls}
      selector="#walk-area"
      onLock={() => onLockChange(true)}
      onUnlock={() => onLockChange(false)}
    />
  );
}

export default function SceneViewer({
  glbUrl,
  initialHeadingDeg,
}: {
  glbUrl: string | null;
  initialHeadingDeg?: number | null;
}) {
  const [locked, setLocked] = useState(false);
  const coordsRef = useRef<HTMLDivElement | null>(null);
  const colliderRef = useRef<THREE.Mesh[]>([]);

  return (
    <div id="walk-area" className="canvasWrap">
      <Canvas
        camera={{ fov: 70, near: 0.05, far: 2000, position: [0, EYE_HEIGHT, 3] }}
        style={{ background: "#aac4d8" }}
      >
        <Suspense fallback={null}>
          {glbUrl && (
            <Model
              key={glbUrl}
              url={glbUrl}
              onReady={(meshes) => {
                colliderRef.current = meshes;
              }}
            />
          )}
        </Suspense>
        <Player
          onLockChange={setLocked}
          coordsRef={coordsRef}
          colliderRef={colliderRef}
          initialHeadingDeg={initialHeadingDeg}
        />
      </Canvas>

      {!locked && (
        <div className="blocker">
          <div className="instructions">
            <p className="big">クリックで歩行開始</p>
            <p>
              <strong>WASD</strong> 移動 / <strong>マウス</strong> 視点 /{" "}
              <strong>Shift</strong> ダッシュ / <strong>Esc</strong> 解除
            </p>
          </div>
        </div>
      )}
      <div className="hud">
        <div ref={coordsRef}>x:0 y:0 z:0</div>
      </div>
    </div>
  );
}
