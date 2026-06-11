"""生成した 3D シーン(glb + meta.json)をディスクに蓄積・列挙する。"""

import json
import uuid
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scenes"


def new_scene_id() -> str:
    return uuid.uuid4().hex[:12]


def scene_dir(scene_id: str) -> Path:
    return DATA_DIR / scene_id


def save_scene(mesh, meta: dict) -> Path:
    """mesh を glb として、meta を json として保存する。glb のパスを返す。"""
    sid = meta["id"]
    d = scene_dir(sid)
    d.mkdir(parents=True, exist_ok=True)

    glb_path = d / "scene.glb"
    mesh.export(str(glb_path))

    (d / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return glb_path


def list_scenes() -> list[dict]:
    if not DATA_DIR.exists():
        return []
    scenes = []
    for d in DATA_DIR.iterdir():
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                scenes.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
    return scenes
