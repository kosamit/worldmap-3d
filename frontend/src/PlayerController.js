import * as THREE from "three";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";

const EYE_HEIGHT = 1.7; // 目線の高さ (m)
const WALK_SPEED = 40; // 加速度 (m/s^2 相当)
const RUN_MULTIPLIER = 2.2;
const DAMPING = 8.0; // 速度減衰係数
const GRAVITY = 25.0;
const JUMP_VELOCITY = 8.5;

/**
 * PointerLock によるマウスルックと WASD 物理移動をまとめたコントローラ。
 * フェーズ1ではフラットな地面 (y=0) を前提に、ジャンプと重力のみ扱う。
 */
export class PlayerController {
  constructor(camera, domElement) {
    this.controls = new PointerLockControls(camera, domElement);
    this.camera = camera;

    this.velocity = new THREE.Vector3();
    this.direction = new THREE.Vector3();
    this.keys = {
      forward: false,
      backward: false,
      left: false,
      right: false,
      run: false,
    };
    this.canJump = false;

    this._onKeyDown = this._onKeyDown.bind(this);
    this._onKeyUp = this._onKeyUp.bind(this);
    document.addEventListener("keydown", this._onKeyDown);
    document.addEventListener("keyup", this._onKeyUp);

    // 初期位置
    this.controls.object.position.set(0, EYE_HEIGHT, 8);
  }

  get object() {
    return this.controls.object;
  }

  lock() {
    this.controls.lock();
  }

  unlock() {
    this.controls.unlock();
  }

  _onKeyDown(event) {
    switch (event.code) {
      case "KeyW":
      case "ArrowUp":
        this.keys.forward = true;
        break;
      case "KeyS":
      case "ArrowDown":
        this.keys.backward = true;
        break;
      case "KeyA":
      case "ArrowLeft":
        this.keys.left = true;
        break;
      case "KeyD":
      case "ArrowRight":
        this.keys.right = true;
        break;
      case "ShiftLeft":
      case "ShiftRight":
        this.keys.run = true;
        break;
      case "Space":
        if (this.canJump) {
          this.velocity.y = JUMP_VELOCITY;
          this.canJump = false;
        }
        break;
    }
  }

  _onKeyUp(event) {
    switch (event.code) {
      case "KeyW":
      case "ArrowUp":
        this.keys.forward = false;
        break;
      case "KeyS":
      case "ArrowDown":
        this.keys.backward = false;
        break;
      case "KeyA":
      case "ArrowLeft":
        this.keys.left = false;
        break;
      case "KeyD":
      case "ArrowRight":
        this.keys.right = false;
        break;
      case "ShiftLeft":
      case "ShiftRight":
        this.keys.run = false;
        break;
    }
  }

  /**
   * 毎フレーム呼ぶ。delta は秒。
   */
  update(delta) {
    if (!this.controls.isLocked) return;

    // 水平方向の減衰
    this.velocity.x -= this.velocity.x * DAMPING * delta;
    this.velocity.z -= this.velocity.z * DAMPING * delta;

    // 重力
    this.velocity.y -= GRAVITY * delta;

    // 入力方向 (カメラローカル基準)
    this.direction.z = Number(this.keys.forward) - Number(this.keys.backward);
    this.direction.x = Number(this.keys.right) - Number(this.keys.left);
    this.direction.normalize();

    const speed = WALK_SPEED * (this.keys.run ? RUN_MULTIPLIER : 1);
    if (this.keys.forward || this.keys.backward) {
      this.velocity.z += this.direction.z * speed * delta;
    }
    if (this.keys.left || this.keys.right) {
      this.velocity.x += this.direction.x * speed * delta;
    }

    // 水平移動を適用 (カメラの向きに沿って)
    this.controls.moveRight(this.velocity.x * delta);
    this.controls.moveForward(this.velocity.z * delta);

    // 垂直移動 (ジャンプ/重力) を適用し、地面でクランプ
    const obj = this.controls.object;
    obj.position.y += this.velocity.y * delta;
    if (obj.position.y < EYE_HEIGHT) {
      this.velocity.y = 0;
      obj.position.y = EYE_HEIGHT;
      this.canJump = true;
    }
  }

  dispose() {
    document.removeEventListener("keydown", this._onKeyDown);
    document.removeEventListener("keyup", this._onKeyUp);
    this.controls.dispose();
  }
}
