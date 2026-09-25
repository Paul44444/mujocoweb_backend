"""Authenticated, allowlisted source editor for the simulation web demo.

This is intentionally disabled unless EDITOR_TOKEN is configured. Editing Python
remains equivalent to running arbitrary code as the backend's OS user.
"""

from __future__ import annotations

import ast
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
MUJOCO_REPOSITORIES = {
    "dapg": ROOT,
    "live-robohive": Path("/home/paul/robohive"),
}
ISAAC_REPOSITORIES = {
    "isaac-web": ROOT,
    "isaac-lift-task": Path("/home/paul/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift"),
    "isaac-franka-robot": Path("/home/paul/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots"),
    "isaac-training": Path("/home/paul/IsaacLab/scripts/reinforcement_learning"),
}
REPOSITORIES = {**MUJOCO_REPOSITORIES, **ISAAC_REPOSITORIES}
ENGINE_REPOSITORIES = {
    "mujoco": tuple(MUJOCO_REPOSITORIES),
    "isaaclab": tuple(ISAAC_REPOSITORIES),
}
# These roots deliberately expose only the files used by this web demo rather
# than the unrelated contents of their parent directories.
LIMITED_ROOT_FILES = {
    "isaac-web": ("isaac_web_worker.py",),
    "isaac-franka-robot": ("franka.py",),
}
ISAAC_RUNTIME_ROOTS = {"isaac-web", "isaac-lift-task", "isaac-franka-robot"}
BACKUP_DIR = Path(os.environ.get("EDITOR_BACKUP_DIR", "/home/paul/.local/share/mujocoweb-editor-backups"))
SCENE_DIR = Path(os.environ.get("MUJOCOWEB_SCENE_DIR", "/home/paul/.local/share/mujocoweb-scenes"))
SCENE_ASSETS = [
    {"id": "cube", "name": "Cube", "asset": "box", "scale": [0.04, 0.04, 0.04]},
    {"id": "sphere", "name": "Sphere", "asset": "sphere", "scale": [0.04, 0.04, 0.04]},
    {"id": "cylinder", "name": "Cylinder", "asset": "cylinder", "scale": [0.03, 0.03, 0.06]},
    {"id": "hammer", "name": "Hammer", "asset": "hammer", "scale": [1.0, 1.0, 1.0]},
]
MAX_CONTENT_BYTES = 150_000
MAX_TREE_FILES = 2_000
EDITABLE_SUFFIXES = {".cfg", ".ini", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".xml", ".yaml", ".yml"}
IGNORED_DIRECTORIES = {".git", ".mypy_cache", ".pytest_cache", ".venv", "__pycache__", "logs", "iterations"}
REVISION_PATTERN = re.compile(r"^[0-9]{14}-[0-9a-f]{12}$")
USER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,31}$")
PUBLIC_LOG_SECRET_PATTERN = re.compile(
    r"(?i)(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._~-]{16,}|(?:OPENAI_API_KEY|EDITOR_TOKEN)\s*=\s*\S+)"
)
router = APIRouter(prefix="/api/editor", tags=["editor"])
write_lock = threading.Lock()


class SaveRequest(BaseModel):
    content: str
    expected_sha256: str


class RestoreRequest(BaseModel):
    revision: str
    expected_sha256: str


class SceneAsset(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,47}$")
    asset: str = Field(pattern=r"^(box|sphere|cylinder|hammer)$")
    position: List[float] = Field(min_length=3, max_length=3)
    rotation: List[float] = Field(min_length=3, max_length=3)
    scale: List[float] = Field(min_length=3, max_length=3)


class SceneRequest(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,47}$")
    assets: List[SceneAsset] = Field(default_factory=list, max_length=100)


def _authorize(authorization: Optional[str]) -> None:
    token = os.environ.get("EDITOR_TOKEN", "")
    if len(token) < 32:
        raise HTTPException(status_code=503, detail="Remote editor is not configured")
    supplied = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
    if not hmac.compare_digest(supplied, token):
        raise HTTPException(status_code=401, detail="Invalid editor password", headers={"WWW-Authenticate": "Bearer"})


def _scene_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,47}", name):
        raise HTTPException(status_code=400, detail="Invalid scene name")
    return SCENE_DIR / f"{name}.json"


def _user_scene_dir(user: str) -> Path:
    if not USER_PATTERN.fullmatch(user):
        raise HTTPException(status_code=400, detail="User name must be 1-32 letters, numbers, spaces, _ or -")
    # User names are deliberately case-insensitive in this password-free test.
    return SCENE_DIR / "users" / user.casefold()


def _user_scene_path(user: str, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,47}", name):
        raise HTTPException(status_code=400, detail="Invalid scene name")
    return _user_scene_dir(user) / f"{name}.json"


def _ensure_user_starter_scene(user: str) -> None:
    path = _user_scene_path(user, "DAPG Relocate Start")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    profile = path.parent / ".profile"
    if not profile.exists():
        profile.write_text(user, encoding="utf-8")
        profile.chmod(0o600)
    if path.exists():
        return
    path.write_text(json.dumps(SceneRequest(name="DAPG Relocate Start", assets=[
        SceneAsset(id="training-cube", asset="box", position=[0.0, 0.0, 0.035], rotation=[0.0, 0.0, 0.0], scale=[0.03, 0.03, 0.03]),
    ]).model_dump(), indent=2), encoding="utf-8")
    path.chmod(0o600)


def _ensure_starter_scene() -> None:
    path = _scene_path("DAPG Relocate Start")
    if path.exists():
        return
    SCENE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(SceneRequest(name="DAPG Relocate Start", assets=[
        SceneAsset(id="training-cube", asset="box", position=[0.0, 0.0, 0.035], rotation=[0.0, 0.0, 0.0], scale=[0.03, 0.03, 0.03]),
    ]).model_dump(), indent=2), encoding="utf-8")


def _file(file_id: str) -> Tuple[Path, str]:
    root_id, separator, relative_path = file_id.partition(":")
    root = REPOSITORIES.get(root_id)
    if not separator or root is None or not relative_path:
        raise HTTPException(status_code=404, detail="Unknown editor file")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Unknown editor file") from exc
    if not candidate.is_file() or candidate.suffix.lower() not in EDITABLE_SUFFIXES:
        raise HTTPException(status_code=404, detail="Unknown or non-editable file")
    if candidate.stat().st_size > MAX_CONTENT_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 150 KB editor limit")
    return candidate, f"{root_id}/{relative_path}"


def _tree_node(root_id: str, directory: Path, relative_path: Path, file_counter: list[int]) -> Optional[dict]:
    if directory.name in IGNORED_DIRECTORIES or directory.name.startswith("."):
        return None
    children = []
    try:
        entries = sorted(directory.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower()))
    except OSError:
        return None
    for entry in entries:
        if entry.is_symlink():
            continue
        child_relative_path = relative_path / entry.name
        if entry.is_dir():
            child = _tree_node(root_id, entry, child_relative_path, file_counter)
            if child and child["children"]:
                children.append(child)
        elif (
            entry.suffix.lower() in EDITABLE_SUFFIXES
            and entry.stat().st_size <= MAX_CONTENT_BYTES
            and file_counter[0] < MAX_TREE_FILES
        ):
            file_counter[0] += 1
            children.append({
                "kind": "file",
                "name": entry.name,
                "id": f"{root_id}:{child_relative_path.as_posix()}",
            })
    return {"kind": "directory", "name": relative_path.name or root_id, "children": children}


def _repository_tree(engine: str) -> list[dict]:
    repository_ids = ENGINE_REPOSITORIES.get(engine)
    if repository_ids is None:
        raise HTTPException(status_code=400, detail="Unknown simulation engine")
    roots = []
    for root_id in repository_ids:
        root = REPOSITORIES[root_id]
        if not root.is_dir():
            continue
        limited_files = LIMITED_ROOT_FILES.get(root_id)
        if limited_files is None:
            node = _tree_node(root_id, root, Path(), [0])
        else:
            children = []
            for relative_name in limited_files:
                candidate = root / relative_name
                if candidate.is_file() and candidate.stat().st_size <= MAX_CONTENT_BYTES:
                    children.append({
                        "kind": "file",
                        "name": candidate.name,
                        "id": f"{root_id}:{relative_name}",
                    })
            node = {"kind": "directory", "name": root_id, "children": children}
        if node and node["children"]:
            roots.append(node)
    return roots


def _find_definition(file_id: str, symbol: str) -> Optional[dict]:
    path, _ = _file(file_id)
    root_id, _, _ = file_id.partition(":")
    root = REPOSITORIES[root_id]
    candidates = [path]
    if root_id in LIMITED_ROOT_FILES:
        candidates.extend(
            root / relative_name for relative_name in LIMITED_ROOT_FILES[root_id]
            if (root / relative_name) != path and (root / relative_name).suffix == ".py"
        )
    else:
        candidates.extend(
            candidate for candidate in root.rglob("*.py")
            if candidate != path
            and not candidate.is_symlink()
            and not any(part in IGNORED_DIRECTORIES or part.startswith(".") for part in candidate.relative_to(root).parts)
            and candidate.stat().st_size <= MAX_CONTENT_BYTES
        )
    for candidate in candidates:
        try:
            tree = ast.parse(candidate.read_text(encoding="utf-8"), filename=str(candidate))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
                return {
                    "id": f"{root_id}:{candidate.relative_to(root).as_posix()}",
                    "line": node.lineno,
                }
    return None


def _sha(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read {path.name}") from exc


def _validate(path: Path, content: str) -> None:
    if not content.strip() or len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
        raise HTTPException(status_code=400, detail="File is empty or exceeds 150 KB")
    try:
        if path.suffix == ".py":
            compile(ast.parse(content, filename=path.name), path.name, "exec")
        elif path.suffix == ".xml":
            ET.fromstring(content)
    except (SyntaxError, ET.ParseError) as exc:
        raise HTTPException(status_code=400, detail=f"Syntax error: {exc}") from exc


def _backup_path(file_id: str, revision: str) -> Path:
    return BACKUP_DIR / hashlib.sha256(file_id.encode("utf-8")).hexdigest() / revision


def _save_backup(file_id: str, current: str) -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    BACKUP_DIR.chmod(0o700)
    folder = BACKUP_DIR / hashlib.sha256(file_id.encode("utf-8")).hexdigest()
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    folder.chmod(0o700)
    original = folder / "original"
    if not original.exists():
        original.write_text(current, encoding="utf-8")
        original.chmod(0o600)
    revision = time.strftime("%Y%m%d%H%M%S", time.gmtime()) + "-" + _sha(current)[:12]
    snapshot = folder / revision
    if not snapshot.exists():
        snapshot.write_text(current, encoding="utf-8")
        snapshot.chmod(0o600)


def _replace(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    try:
        temporary_path.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _runtime_for_file(file_id: str) -> Optional[str]:
    root_id, _, _ = file_id.partition(":")
    if root_id in ISAAC_RUNTIME_ROOTS:
        return "isaaclab"
    if root_id == "isaac-training":
        return None
    return "mujoco"


def _restart_runtime(runtime: str) -> None:
    # Run only after the HTTP response has been sent, so the editor knows that
    # its atomic save and backup completed before a process disappears.
    if os.environ.get("EDITOR_RESTART_ON_SAVE", "1") == "0":
        return
    time.sleep(1)
    if runtime == "isaaclab":
        try:
            subprocess.run(
                ["systemctl", "--user", "restart", "mujocoweb-isaac.service"],
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        return
    # The web backend service has Restart=on-failure.
    os._exit(75)


@router.get("/tree")
def get_repository_tree(
    engine: str = "mujoco",
) -> dict:
    return {"engine": engine, "roots": _repository_tree(engine)}


@router.get("/auth")
def authenticate_owner(authorization: Optional[str] = Header(default=None)) -> dict:
    _authorize(authorization)
    return {"authenticated": True}


@router.get("/scenes")
def list_scenes(authorization: Optional[str] = Header(default=None)) -> dict:
    _authorize(authorization)
    _ensure_starter_scene()
    if not SCENE_DIR.is_dir():
        return {"scenes": []}
    return {"scenes": sorted(path.stem for path in SCENE_DIR.glob("*.json"))}


@router.get("/scene-assets")
def list_scene_assets(authorization: Optional[str] = Header(default=None)) -> dict:
    _authorize(authorization)
    return {"assets": SCENE_ASSETS}


@router.get("/users/{user}/scenes")
def list_user_scenes(user: str) -> dict:
    """List scenes in a name-only test account; no password is required."""
    _ensure_user_starter_scene(user)
    folder = _user_scene_dir(user)
    return {
        "user": user,
        "scenes": sorted(path.stem for path in folder.glob("*.json")),
        "authentication": "none",
    }


@router.get("/users")
def list_scene_users() -> dict:
    """List the public name-only test accounts that currently exist."""
    users_root = SCENE_DIR / "users"
    if not users_root.is_dir():
        return {"users": [], "authentication": "none"}
    users = []
    for folder in users_root.iterdir():
        if not folder.is_dir() or not USER_PATTERN.fullmatch(folder.name):
            continue
        profile = folder / ".profile"
        try:
            display_name = profile.read_text(encoding="utf-8").strip() if profile.is_file() else folder.name
        except OSError:
            display_name = folder.name
        users.append(display_name if USER_PATTERN.fullmatch(display_name) else folder.name)
    return {"users": sorted(users, key=str.casefold), "authentication": "none"}


@router.get("/users/{user}/scenes/{name}")
def get_user_scene(user: str, name: str) -> dict:
    path = _user_scene_path(user, name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Scene not found")
    try:
        return SceneRequest.model_validate_json(path.read_text(encoding="utf-8")).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Stored scene is invalid") from exc


@router.put("/users/{user}/scenes/{name}")
def save_user_scene(user: str, name: str, request: SceneRequest) -> dict:
    if request.name != name:
        raise HTTPException(status_code=400, detail="Scene name does not match URL")
    if len({asset.id for asset in request.assets}) != len(request.assets):
        raise HTTPException(status_code=400, detail="Asset IDs must be unique")
    path = _user_scene_path(user, name)
    with write_lock:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text(json.dumps(request.model_dump(), indent=2), encoding="utf-8")
        path.chmod(0o600)
    return {"user": user, "name": name, "assets": len(request.assets)}


@router.get("/scenes/{name}")
def get_scene(name: str, authorization: Optional[str] = Header(default=None)) -> dict:
    _authorize(authorization)
    path = _scene_path(name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Scene not found")
    try:
        return SceneRequest.parse_raw(path.read_text(encoding="utf-8")).dict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Stored scene is invalid") from exc


@router.put("/scenes/{name}")
def save_scene(name: str, request: SceneRequest, authorization: Optional[str] = Header(default=None)) -> dict:
    _authorize(authorization)
    if request.name != name:
        raise HTTPException(status_code=400, detail="Scene name does not match URL")
    if len({asset.id for asset in request.assets}) != len(request.assets):
        raise HTTPException(status_code=400, detail="Asset IDs must be unique")
    path = _scene_path(name)
    SCENE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(request.model_dump(), indent=2), encoding="utf-8")
    path.chmod(0o600)
    return {"name": name, "assets": len(request.assets)}


@router.get("/definitions")
def get_definition(
    file_id: str,
    symbol: str,
) -> dict:
    if not re.fullmatch(r"[A-Za-z_]\w*", symbol):
        raise HTTPException(status_code=400, detail="Invalid Python symbol")
    result = _find_definition(file_id, symbol)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No Python definition found for {symbol}")
    return result


@router.get("/logs")
def get_backend_logs(
    engine: str = "mujoco",
) -> dict:
    unit = "mujocoweb-isaac.service" if engine == "isaaclab" else "mujocoweb-backend.service"
    try:
        result = subprocess.run(
            ["journalctl", "--user", f"--unit={unit}",
             "--lines=2000", "--no-pager", "--output=short-iso"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(status_code=503, detail="Backend logs are unavailable") from exc
    if result.returncode != 0:
        raise HTTPException(status_code=503, detail="Backend logs are unavailable")
    public_logs = PUBLIC_LOG_SECRET_PATTERN.sub("[redacted secret]", result.stdout[-500_000:])
    return {"logs": public_logs}


@router.get("/files/{file_id:path}")
def get_file(file_id: str) -> dict:
    path, description = _file(file_id)
    content = _read(path)
    folder = BACKUP_DIR / hashlib.sha256(file_id.encode("utf-8")).hexdigest()
    revisions = ["original"] if (folder / "original").is_file() else []
    if folder.is_dir():
        revisions += sorted(
            (p.name for p in folder.iterdir() if REVISION_PATTERN.fullmatch(p.name)),
            reverse=True,
        )[:30]
    return {
        "id": file_id,
        "name": path.name,
        "description": description,
        "content": content,
        "sha256": _sha(content),
        "revisions": revisions,
    }


@router.put("/files/{file_id:path}")
def save_file(
    file_id: str,
    request: SaveRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
) -> dict:
    _authorize(authorization)
    path, _ = _file(file_id)
    _validate(path, request.content)
    with write_lock:
        current = _read(path)
        if _sha(current) != request.expected_sha256:
            raise HTTPException(status_code=409, detail="File changed; reload before saving")
        if current == request.content:
            return {"sha256": _sha(current), "restarting": False, "runtime": None}
        _save_backup(file_id, current)
        _replace(path, request.content)
    runtime = _runtime_for_file(file_id)
    if runtime is not None:
        background_tasks.add_task(_restart_runtime, runtime)
    return {"sha256": _sha(request.content), "restarting": runtime is not None, "runtime": runtime}


@router.post("/files/restore/{file_id:path}")
def restore_file(
    file_id: str,
    request: RestoreRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
) -> dict:
    _authorize(authorization)
    path, _ = _file(file_id)
    if request.revision != "original" and not REVISION_PATTERN.fullmatch(request.revision):
        raise HTTPException(status_code=400, detail="Invalid revision")
    snapshot = _backup_path(file_id, request.revision)
    if not snapshot.is_file():
        raise HTTPException(status_code=404, detail="Backup not found")
    restored = _read(snapshot)
    _validate(path, restored)
    with write_lock:
        current = _read(path)
        if _sha(current) != request.expected_sha256:
            raise HTTPException(status_code=409, detail="File changed; reload before restoring")
        _save_backup(file_id, current)
        _replace(path, restored)
    runtime = _runtime_for_file(file_id)
    if runtime is not None:
        background_tasks.add_task(_restart_runtime, runtime)
    return {"sha256": _sha(restored), "restarting": runtime is not None, "runtime": runtime}
