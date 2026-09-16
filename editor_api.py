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
FILES = {
    "live-relocate-environment": (
        Path("/home/paul/robohive/robohive/envs/hands/relocate_v1.py"),
        "Live Relocate environment and rewards",
    ),
    "live-relocate-scene": (
        Path("/home/paul/robohive/robohive/envs/hands/assets/DAPG_relocate.xml"),
        "Live default MuJoCo scene and robot model",
    ),
    "custom-object-template": (
        ROOT / "robohive/robohive/envs/hands/assets/DAPG_relocate.xml",
        "Scene template used when generating a custom object",
    ),
    "simulation": (
        ROOT / "muj1.py",
        "Web simulation and rendering",
    ),
    "training": (
        ROOT / "mjrlpaul/utils/train_agent.py",
        "Training code; does not change the running demo",
    ),
}
BACKUP_DIR = Path(os.environ.get("EDITOR_BACKUP_DIR", "/home/paul/.local/share/mujocoweb-editor-backups"))
MAX_CONTENT_BYTES = 150_000
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
    result = FILES.get(file_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown editor file")
    return result


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
        else:
            ET.fromstring(content)
    except (SyntaxError, ET.ParseError) as exc:
        raise HTTPException(status_code=400, detail=f"Syntax error: {exc}") from exc


def _backup_path(file_id: str, revision: str) -> Path:
    return BACKUP_DIR / file_id / revision


def _save_backup(file_id: str, current: str) -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    BACKUP_DIR.chmod(0o700)
    folder = BACKUP_DIR / file_id
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


@router.get("/files")
def list_files(authorization: Optional[str] = Header(default=None)) -> dict:
    _authorize(authorization)
    return {
        "files": [
            {"id": file_id, "name": path.name, "description": description}
            for file_id, (path, description) in FILES.items()
        ]
    }


@router.get("/logs")
def get_backend_logs(authorization: Optional[str] = Header(default=None)) -> dict:
    _authorize(authorization)
    try:
        result = subprocess.run(
            ["journalctl", "--user", "--unit=mujocoweb-backend.service",
             "--lines=120", "--no-pager", "--output=short-iso"],
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
    return {"logs": result.stdout[-60_000:]}


@router.get("/files/{file_id}")
def get_file(file_id: str, authorization: Optional[str] = Header(default=None)) -> dict:
    _authorize(authorization)
    path, description = _file(file_id)
    content = _read(path)
    folder = BACKUP_DIR / file_id
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


@router.put("/files/{file_id}")
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


@router.post("/files/{file_id}/restore")
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
