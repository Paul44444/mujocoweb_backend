"""Authenticated, allowlisted source editor for the personal MuJoCo demo.

This is intentionally disabled unless EDITOR_TOKEN is configured. Editing Python
remains equivalent to running arbitrary code as the backend's OS user.
"""

from __future__ import annotations

import ast
import hashlib
import hmac
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parent
REPOSITORIES = {
    "dapg": ROOT,
    "live-robohive": Path("/home/paul/robohive"),
}
BACKUP_DIR = Path(os.environ.get("EDITOR_BACKUP_DIR", "/home/paul/.local/share/mujocoweb-editor-backups"))
MAX_CONTENT_BYTES = 150_000
MAX_TREE_FILES = 2_000
EDITABLE_SUFFIXES = {".cfg", ".ini", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".xml", ".yaml", ".yml"}
IGNORED_DIRECTORIES = {".git", ".mypy_cache", ".pytest_cache", ".venv", "__pycache__", "logs", "iterations"}
REVISION_PATTERN = re.compile(r"^[0-9]{14}-[0-9a-f]{12}$")
router = APIRouter(prefix="/api/editor", tags=["editor"])
write_lock = threading.Lock()


class SaveRequest(BaseModel):
    content: str
    expected_sha256: str


class RestoreRequest(BaseModel):
    revision: str
    expected_sha256: str


def _authorize(authorization: Optional[str]) -> None:
    token = os.environ.get("EDITOR_TOKEN", "")
    if len(token) < 32:
        raise HTTPException(status_code=503, detail="Remote editor is not configured")
    supplied = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
    if not hmac.compare_digest(supplied, token):
        raise HTTPException(status_code=401, detail="Invalid editor password", headers={"WWW-Authenticate": "Bearer"})


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


def _repository_tree() -> list[dict]:
    return [
        _tree_node(root_id, root, Path(), [0])
        for root_id, root in REPOSITORIES.items()
        if root.is_dir()
    ]


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


def _restart_service() -> None:
    # The local systemd user service has Restart=on-failure. Do this only after
    # the HTTP response has been sent, so the caller knows the save succeeded.
    if os.environ.get("EDITOR_RESTART_ON_SAVE", "1") != "0":
        time.sleep(1)
        os._exit(75)


@router.get("/tree")
def get_repository_tree(authorization: Optional[str] = Header(default=None)) -> dict:
    _authorize(authorization)
    return {"roots": _repository_tree()}


@router.get("/logs")
def get_backend_logs(authorization: Optional[str] = Header(default=None)) -> dict:
    _authorize(authorization)
    try:
        result = subprocess.run(
            ["journalctl", "--user", "--unit=mujocoweb-backend.service",
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
    return {"logs": result.stdout[-500_000:]}


@router.get("/files/{file_id:path}")
def get_file(file_id: str, authorization: Optional[str] = Header(default=None)) -> dict:
    _authorize(authorization)
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
            return {"sha256": _sha(current), "restarting": False}
        _save_backup(file_id, current)
        _replace(path, request.content)
    background_tasks.add_task(_restart_service)
    return {"sha256": _sha(request.content), "restarting": True}


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
    background_tasks.add_task(_restart_service)
    return {"sha256": _sha(restored), "restarting": True}
