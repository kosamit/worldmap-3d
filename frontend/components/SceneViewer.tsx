"use client";

// 生成した glb を読み込み、PointerLock + WASD で歩ける R3F ビューワー。
// 旧 PlayerController.js の物理（減衰・重力・ジャンプ・ダッシュ）を移植。

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { PointerLockControls, useGLTF } from "@react-three/drei";
import {
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from "react";
import * as THREE from "three";

const EYE_HEIGHT = 1.7;
const WALK_SPEED = 40;
const RUN_MULTIPLIER = 2.2;
const DAMPING = 8.0;
const GRAVITY = 25.0;
const JUMP_VELOCITY = 8.5;

function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  const object = useMemo(() => {
    const cloned = scene.clone(true);
    cloned.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (mesh.isMesh) {
        const material = (mesh.material as THREE.Material).clone();
        material.side = THREE.DoubleSide;
        (material as THREE.MeshStandardMaterial).vertexColors = true;
        mesh.material = material;
        mesh.frustumCulled = false;
      }
    });
    return cloned;
  }, [scene]);
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
}: {
  onLockChange: (locked: boolean) => void;
  coordsRef: RefObject<HTMLDivElement | null>;
}) {
  const { camera } = useThree();
  // drei は three の PointerLockControls インスタンスを ref に渡す。
  // moveForward / moveRight / isLocked を使うため any で受ける。
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const controls = useRef<any>(null);
  const velocity = useRef(new THREE.Vector3());
  const direction = useRef(new THREE.Vector3());
  const canJump = useRef(false);
  const keys = useRef<KeyState>({
    forward: false,
    backward: false,
    left: false,
    right: false,
    run: false,
  });

  useEffect(() => {
    camera.position.set(0, EYE_HEIGHT, 3);

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
  }, [camera]);

  useFrame((_, rawDelta) => {
    const c = controls.current;
    if (!c || !c.isLocked) return;
    const delta = Math.min(rawDelta, 0.1);
    const v = velocity.current;

    v.x -= v.x * DAMPING * delta;
    v.z -= v.z * DAMPING * delta;
    v.y -= GRAVITY * delta;

    const k = keys.current;
    direction.current.z = Number(k.forward) - Number(k.backward);
    direction.current.x = Number(k.right) - Number(k.left);
    direction.current.normalize();

    const speed = WALK_SPEED * (k.run ? RUN_MULTIPLIER : 1);
    if (k.forward || k.backward) v.z += direction.current.z * speed * delta;
    if (k.left || k.right) v.x += direction.current.x * speed * delta;

    c.moveRight(v.x * delta);
    c.moveForward(v.z * delta);

    camera.position.y += v.y * delta;
    if (camera.position.y < EYE_HEIGHT) {
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

export default function SceneViewer({ glbUrl }: { glbUrl: string | null }) {
  const [locked, setLocked] = useState(false);
  const coordsRef = useRef<HTMLDivElement | null>(null);

  return (
    <div id="walk-area" className="canvasWrap">
      <Canvas
        camera={{ fov: 70, near: 0.05, far: 2000, position: [0, EYE_HEIGHT, 3] }}
        style={{ background: "#101418" }}
      >
        <hemisphereLight args={[0xffffff, 0x404040, 1.4]} />
        <directionalLight position={[5, 10, 7]} intensity={1.0} />
        <gridHelper args={[60, 60, 0x335577, 0x223344]} />
        <Suspense fallback={null}>
          {glbUrl && <Model key={glbUrl} url={glbUrl} />}
        </Suspense>
        <Player onLockChange={setLocked} coordsRef={coordsRef} />
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
