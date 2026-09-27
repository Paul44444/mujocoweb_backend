"""Bounded process manager and read-only metrics API for web DAPG training."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, List
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
RUNS_DIRECTORY = ROOT / "web_training_runs"
WORKER = ROOT / "web_training_worker.py"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
router = APIRouter(prefix="/api/training", tags=["training"])
manager_lock = threading.Lock()


class StartTrainingRequest(BaseModel):
    user: str = Field(default="Guest", min_length=1, max_length=32)
    iterations: int = Field(default=10, ge=1, le=25)
    trajectories: int = Field(default=3, ge=1, le=8)
    horizon: int = Field(default=200, ge=20, le=500)
    seed: int = Field(default=123, ge=0, le=2_147_483_647)


def _read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return fallback


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _run_payload(run_directory: Path) -> Dict[str, object]:
    status = _read_json(run_directory / "status.json", {"status": "unknown", "iteration": 0})
    config = _read_json(run_directory / "config.json", {})
    metrics = _read_json(run_directory / "metrics.json", {"metrics": []}).get("metrics", [])
    pid_data = _read_json(run_directory / "process.json", {})
    pid = int(pid_data.get("pid", 0) or 0)
    if status.get("status") in {"starting", "training"} and pid and not _is_alive(pid):
        status = {**status, "status": "failed", "error": "Training process exited unexpectedly."}
    checkpoints = sorted((run_directory / "checkpoints").glob("policy_*.pickle"))
    return {
        "id": run_directory.name,
        "status": status,
        "config": config,
        "metrics": metrics[-250:],
        "checkpoints": [path.name for path in checkpoints],
    }


def _active_run() -> Dict[str, object] | None:
    if not RUNS_DIRECTORY.exists():
        return None
    for run_directory in sorted(RUNS_DIRECTORY.iterdir(), reverse=True):
        if not run_directory.is_dir():
            continue
        payload = _run_payload(run_directory)
        if payload["status"].get("status") in {"starting", "training"}:
            return payload
    return None


@router.get("/runs")
def list_training_runs() -> Dict[str, List[Dict[str, object]]]:
    RUNS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    runs = [_run_payload(path) for path in sorted(RUNS_DIRECTORY.iterdir(), reverse=True) if path.is_dir()]
    return {"runs": runs[:20]}


@router.get("/runs/{run_id}")
def get_training_run(run_id: str) -> Dict[str, object]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="Invalid training run ID.")
    run_directory = RUNS_DIRECTORY / run_id
    if not run_directory.is_dir():
        raise HTTPException(status_code=404, detail="Training run not found.")
    return _run_payload(run_directory)


@router.post("/start")
def start_training(request: StartTrainingRequest) -> Dict[str, object]:
    with manager_lock:
        active = _active_run()
        if active:
            raise HTTPException(status_code=409, detail=f"Training run {active['id']} is already active.")
        RUNS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        user = re.sub(r"[^A-Za-z0-9_-]", "-", request.user).strip("-")[:32] or "Guest"
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{user}-{uuid.uuid4().hex[:6]}"
        run_directory = RUNS_DIRECTORY / run_id
        run_directory.mkdir(mode=0o750)
        (run_directory / "status.json").write_text(
            json.dumps({"status": "starting", "started_at": time.time(), "iteration": 0}),
            encoding="utf-8",
        )
        log_file = (run_directory / "training.log").open("ab", buffering=0)
        command = [
            sys.executable,
            str(WORKER),
            "--run-directory", str(run_directory),
            "--iterations", str(request.iterations),
            "--trajectories", str(request.trajectories),
            "--horizon", str(request.horizon),
            "--seed", str(request.seed),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise HTTPException(status_code=500, detail="Could not start the training process.") from exc
        finally:
            log_file.close()
        (run_directory / "process.json").write_text(json.dumps({"pid": process.pid}), encoding="utf-8")
        return _run_payload(run_directory)


@router.post("/runs/{run_id}/stop")
def stop_training(run_id: str) -> Dict[str, object]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="Invalid training run ID.")
    run_directory = RUNS_DIRECTORY / run_id
    if not run_directory.is_dir():
        raise HTTPException(status_code=404, detail="Training run not found.")
    payload = _run_payload(run_directory)
    if payload["status"].get("status") not in {"starting", "training"}:
        return payload
    pid = int(_read_json(run_directory / "process.json", {}).get("pid", 0) or 0)
    if pid and _is_alive(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            pass
    return _run_payload(run_directory)
