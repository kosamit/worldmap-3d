"use client";

// 1パノラマ分のキューブ6面を「内側から見る箱」のスカイボックスとして表示する。
// scene.background のキューブは左右反転して見えるため、箱を BackSide + 各面を
// 水平反転して正しい向きにする。マウスドラッグで見回し、方位はコンパスで表示。
// W=見ている方向へ進む / S=戻る、画面の矢印で隣接ノードへ移動する。

import { Canvas, useThree } from "@react-three/fiber";
import { useCallback, useEffect, useMemo, useState } from "react";
import * as THREE from "three";

export interface PanoLink {
  heading: number;
  pano: string;
}

const LOOK_SPEED = 0.0025;

const CARDINALS = ["北", "北東", "東", "南東", "南", "南西", "西", "北西"];
function cardinal(bearing: number): string {
  return CARDINALS[Math.round(bearing / 45) % 8];
}

// equirectangular を内側から見る球体スカイボックス。
// scale(-1,1,1) で球を裏返し、内側から見ても左右反転しないようにする。
// lon=0(北) を -Z(カメラ初期方向=コンパス北) に合わせるため Y 回転で調整。
function Skybox({ url }: { url: string }) {
  const object = useMemo(() => {
    const loader = new THREE.TextureLoader();
    loader.setCrossOrigin("anonymous");
    const texture = loader.load(url);
    texture.colorSpace = THREE.SRGBColorSpace;
    const geometry = new THREE.SphereGeometry(500, 64, 48);
    geometry.scale(-1, 1, 1);
    geometry.rotateY(Math.PI);
    const material = new THREE.MeshBasicMaterial({ map: texture });
    return new THREE.Mesh(geometry, material);
  }, [url]);
  return <primitive object={object} />;
}

function DragLook({ onFacing }: { onFacing: (bearing: number) => void }) {
  const { camera, gl } = useThree();
  useEffect(() => {
    camera.rotation.order = "YXZ";
    const state = { dragging: false, x: 0, y: 0, yaw: 0, pitch: 0 };
    const emit = () => {
      const deg = (state.yaw * 180) / Math.PI;
      onFacing(((-deg) % 360 + 360) % 360);
    };
    emit();
    const onDown = (e: PointerEvent) => {
      state.dragging = true;
      state.x = e.clientX;
      state.y = e.clientY;
      gl.domElement.style.cursor = "grabbing";
    };
    const onUp = () => {
      state.dragging = false;
      gl.domElement.style.cursor = "grab";
    };
    const onMove = (e: PointerEvent) => {
      if (!state.dragging) return;
      state.yaw -= (e.clientX - state.x) * LOOK_SPEED;
      state.pitch -= (e.clientY - state.y) * LOOK_SPEED;
      state.x = e.clientX;
      state.y = e.clientY;
      const limit = Math.PI / 2 - 0.02;
      state.pitch = Math.max(-limit, Math.min(limit, state.pitch));
      camera.rotation.y = state.yaw;
      camera.rotation.x = state.pitch;
      emit();
    };
    gl.domElement.style.cursor = "grab";
    gl.domElement.addEventListener("pointerdown", onDown);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointermove", onMove);
    return () => {
      gl.domElement.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointermove", onMove);
    };
  }, [camera, gl, onFacing]);
  return null;
}

export default function TourViewer({
  base,
  equirect,
  links,
  canBack,
  onForward,
  onBack,
  onStepLink,
  onFacingChange,
}: {
  base: string;
  equirect: string | null;
  links: PanoLink[];
  canBack: boolean;
  onForward: () => void;
  onBack: () => void;
  onStepLink: (pano: string, heading: number) => void;
  onFacingChange?: (bearing: number) => void;
}) {
  const [facing, setFacing] = useState(0);

  const handleFacing = useCallback(
    (bearing: number) => {
      setFacing(bearing);
      onFacingChange?.(bearing);
    },
    [onFacingChange],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "KeyW" || e.code === "ArrowUp") onForward();
      if (e.code === "KeyS" || e.code === "ArrowDown") onBack();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onForward, onBack]);

  const url = useMemo(
    () => (equirect ? `${base.replace(/\/+$/, "")}${equirect}` : null),
    [equirect, base],
  );

  return (
    <div id="tour-area" className="canvasWrap">
      <Canvas camera={{ fov: 75, near: 0.1, far: 1000, position: [0, 0, 0] }}>
        <DragLook onFacing={handleFacing} />
        {url && <Skybox url={url} />}
      </Canvas>

      {!equirect && (
        <div className="blocker">
          <div className="instructions">
            <p className="big">地図をクリックして降り立つ地点を選んでください</p>
            <p>そこから両隣のパノラマへ歩いていけます</p>
          </div>
        </div>
      )}

      {equirect && (
        <>
          <div className="compass" aria-label="方位">
            <div
              className="compassDial"
              style={{ transform: `rotate(${-facing}deg)` }}
            >
              <span className="cN">N</span>
              <span className="cE">E</span>
              <span className="cS">S</span>
              <span className="cW">W</span>
            </div>
            <div className="compassNeedle" />
            <div className="compassLabel">
              {cardinal(facing)} {Math.round(facing)}°
            </div>
          </div>

          <div className="panoArrows">
            {links.map((link) => (
              <button
                key={link.pano}
                type="button"
                title={`${cardinal(link.heading)}へ進む`}
                style={{ transform: `rotate(${link.heading - facing}deg)` }}
                onClick={() => onStepLink(link.pano, link.heading)}
              >
                ↑
              </button>
            ))}
            {canBack && (
              <button type="button" className="backBtn" onClick={onBack}>
                ↩
              </button>
            )}
          </div>

          <div className="hud">ドラッグで見回し / W 進む・S 戻る</div>
        </>
      )}
    </div>
  );
}
