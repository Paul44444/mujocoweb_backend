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
from typing import Dict, List, Literal, Optional
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
RUNS_DIRECTORY = ROOT / "web_training_runs"
WORKER = ROOT / "web_training_worker.py"
ISAAC_WORKER = ROOT / "isaac_training_worker.py"
ISAAC_OUTPUT_DIRECTORY = Path(os.environ.get("ISAAC_OUTPUT_DIRECTORY", "/tmp/mujocoweb-isaac"))
ISAAC_POLICY_SELECTION = ISAAC_OUTPUT_DIRECTORY / "selected_policy.json"
ISAAC_STATUS_PATH = ISAAC_OUTPUT_DIRECTORY / "status.json"
ISAAC_FRAME_PATH = ISAAC_OUTPUT_DIRECTORY / "frame.jpg"
ISAAC_METADATA_PATH = ISAAC_OUTPUT_DIRECTORY / "metadata.json"
ISAAC_CONTROL_DIRECTORY = ISAAC_OUTPUT_DIRECTORY / "commands"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
CHECKPOINT_PATTERN = re.compile(r"^model_\d+\.pt$")
router = APIRouter(prefix="/api/training", tags=["training"])
manager_lock = threading.Lock()


class StartTrainingRequest(BaseModel):
    engine: Literal["mujoco", "isaaclab"] = "mujoco"
    user: str = Field(default="Guest", min_length=1, max_length=32)
    name: Optional[str] = Field(default=None, max_length=48)
    iterations: int = Field(default=10, ge=1, le=500)
    trajectories: int = Field(default=3, ge=1, le=8)
    horizon: int = Field(default=200, ge=20, le=500)
    seed: int = Field(default=123, ge=0, le=2_147_483_647)
    num_envs: Literal[16, 32, 64] = 32
    resume_checkpoint: Optional[str] = Field(default=None, max_length=160)


class SelectCheckpointRequest(BaseModel):
    checkpoint: Optional[str] = Field(default=None, max_length=160)


class DeleteCheckpointRequest(BaseModel):
    checkpoint: str = Field(min_length=1, max_length=160)


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


def _checkpoint_path(checkpoint_id: str) -> Path:
    run_id, separator, checkpoint_name = checkpoint_id.partition("/")
    if (
        not separator
        or not RUN_ID_PATTERN.fullmatch(run_id)
        or not CHECKPOINT_PATTERN.fullmatch(checkpoint_name)
    ):
        raise HTTPException(status_code=400, detail="Invalid checkpoint ID.")
    path = RUNS_DIRECTORY / run_id / "checkpoints" / checkpoint_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Checkpoint not found.")
    return path.resolve()


def _run_display_name(run_directory: Path, config: Dict[str, object]) -> str:
    configured_name = str(config.get("name") or "").strip()
    if configured_name:
        return configured_name
    legacy_match = re.fullmatch(r"\d{8}-\d{6}-(.+)-[0-9a-f]{6}", run_directory.name)
    return legacy_match.group(1) if legacy_match else run_directory.name


def _checkpoint_payloads() -> List[Dict[str, object]]:
    checkpoints = []
    if not RUNS_DIRECTORY.is_dir():
        return checkpoints
    for run_directory in sorted(RUNS_DIRECTORY.iterdir(), reverse=True):
        config = _read_json(run_directory / "config.json", {})
        if not run_directory.is_dir() or config.get("engine") != "isaaclab":
            continue
        for path in sorted(
            (run_directory / "checkpoints").glob("model_*.pt"),
            key=lambda item: int(item.stem.split("_")[-1]),
            reverse=True,
        ):
            checkpoints.append({
                "id": f"{run_directory.name}/{path.name}",
                "run_id": run_directory.name,
                "name": path.name,
                "label": f"{_run_display_name(run_directory, config)} · {path.name}",
                "modified_at": path.stat().st_mtime,
                "deletable": True,
            })
    return checkpoints[:100]


def _restart_isaac_runtime() -> None:
    try:
        subprocess.run(
            ["systemctl", "--user", "stop", "mujocoweb-isaac.service"],
            timeout=40,
            check=False,
        )
        ISAAC_FRAME_PATH.unlink(missing_ok=True)
        ISAAC_METADATA_PATH.unlink(missing_ok=True)
        subprocess.run(
            ["systemctl", "--user", "start", "mujocoweb-isaac.service"],
            timeout=40,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _publish_isaac_command(command: Dict[str, object]) -> None:
    ISAAC_CONTROL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    name = f"{time.time_ns()}-{os.getpid()}-policy.json"
    path = ISAAC_CONTROL_DIRECTORY / name
    temporary = ISAAC_CONTROL_DIRECTORY / f".{name}.next"
    temporary.write_text(json.dumps(command), encoding="utf-8")
    os.replace(temporary, path)


def _sample_metrics(metrics: List[Dict[str, object]], maximum: int = 200) -> List[Dict[str, object]]:
    """Return a bounded, whole-run sample while preserving both endpoints."""
    if len(metrics) <= maximum:
        return metrics
    return [
        metrics[round(index * (len(metrics) - 1) / (maximum - 1))]
        for index in range(maximum)
    ]


def _run_payload(run_directory: Path) -> Dict[str, object]:
    status = _read_json(run_directory / "status.json", {"status": "unknown", "iteration": 0})
    config = _read_json(run_directory / "config.json", {})
    config = {**config, "name": _run_display_name(run_directory, config)}
    metrics = _read_json(run_directory / "metrics.json", {"metrics": []}).get("metrics", [])
    pid_data = _read_json(run_directory / "process.json", {})
    pid = int(pid_data.get("pid", 0) or 0)
    if status.get("status") in {"starting", "training"} and pid and not _is_alive(pid):
        status = {**status, "status": "failed", "error": "Training process exited unexpectedly."}
    if (
        config.get("engine") == "isaaclab"
        and status.get("status") == "completed"
        and not metrics
    ):
        status = {
            **status,
            "status": "failed",
            "error": "Isaac Lab exited before producing a training iteration. Check the training log for the GPU error.",
        }
    checkpoints = sorted(
        list((run_directory / "checkpoints").glob("policy_*.pickle"))
        + list((run_directory / "checkpoints").glob("model_*.pt"))
    )
    return {
        "id": run_directory.name,
        "status": status,
        "config": config,
        "metrics": _sample_metrics(metrics),
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
def list_training_runs(
    engine: Optional[Literal["mujoco", "isaaclab"]] = Query(default=None),
) -> Dict[str, List[Dict[str, object]]]:
    RUNS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    runs = [_run_payload(path) for path in sorted(RUNS_DIRECTORY.iterdir(), reverse=True) if path.is_dir()]
    if engine:
        runs = [run for run in runs if run.get("config", {}).get("engine", "mujoco") == engine]
    return {"runs": runs[:20]}


@router.get("/runs/{run_id}")
def get_training_run(run_id: str) -> Dict[str, object]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="Invalid training run ID.")
    run_directory = RUNS_DIRECTORY / run_id
    if not run_directory.is_dir():
        raise HTTPException(status_code=404, detail="Training run not found.")
    return _run_payload(run_directory)


@router.get("/checkpoints")
def list_checkpoints() -> Dict[str, object]:
    selected = _read_json(ISAAC_POLICY_SELECTION, {}).get("id")
    return {"checkpoints": _checkpoint_payloads(), "selected": selected}


@router.get("/checkpoints/status")
def checkpoint_status() -> Dict[str, object]:
    selected = _read_json(ISAAC_POLICY_SELECTION, {}).get("id")
    status = _read_json(ISAAC_STATUS_PATH, {"status": "unknown"})
    return {"selected": selected, "worker": status}


@router.delete("/checkpoints")
def delete_checkpoint(request: DeleteCheckpointRequest) -> Dict[str, object]:
    selected = _read_json(ISAAC_POLICY_SELECTION, {}).get("id")
    if request.checkpoint == selected:
        raise HTTPException(
            status_code=409,
            detail="The active checkpoint cannot be deleted. Load another policy first.",
        )
    path = _checkpoint_path(request.checkpoint)
    path.unlink()
    return {"deleted": request.checkpoint}


@router.post("/checkpoints/select")
def select_checkpoint(request: SelectCheckpointRequest, background_tasks: BackgroundTasks) -> Dict[str, object]:
    ISAAC_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if request.checkpoint:
        path = _checkpoint_path(request.checkpoint)
        payload = {"id": request.checkpoint, "path": str(path), "selected_at": time.time()}
    else:
        payload = {"id": None, "path": None, "selected_at": time.time()}
    temporary = ISAAC_POLICY_SELECTION.with_suffix(".json.next")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, ISAAC_POLICY_SELECTION)
    current_status = _read_json(ISAAC_STATUS_PATH, {})
    hot_swap = bool(
        payload["id"]
        and current_status.get("status") == "ready"
        and current_status.get("mode") == "trained_policy"
    )
    if hot_swap:
        status_temporary = ISAAC_STATUS_PATH.with_suffix(".json.next")
        status_temporary.write_text(json.dumps({
            **current_status,
            "status": "switching_policy",
            "requested_checkpoint": payload["id"],
        }), encoding="utf-8")
        os.replace(status_temporary, ISAAC_STATUS_PATH)
        _publish_isaac_command({
            "type": "policy_load",
            "id": payload["id"],
            "path": payload["path"],
        })
        return {"selected": payload["id"], "status": "switching_policy", "hot_swap": True}
    status_temporary = ISAAC_STATUS_PATH.with_suffix(".json.next")
    status_temporary.write_text(json.dumps({
        "status": "restarting",
        "checkpoint": payload["id"],
        "requested_at": time.time(),
    }), encoding="utf-8")
    os.replace(status_temporary, ISAAC_STATUS_PATH)
    background_tasks.add_task(_restart_isaac_runtime)
    return {"selected": payload["id"], "status": "restarting", "hot_swap": False, "estimated_seconds": 45}


@router.post("/start")
def start_training(request: StartTrainingRequest) -> Dict[str, object]:
    with manager_lock:
        if request.engine == "mujoco" and request.iterations > 25:
            raise HTTPException(status_code=422, detail="MuJoCo web training is limited to 25 iterations per run.")
        resume_path = None
        if request.resume_checkpoint:
            if request.engine != "isaaclab":
                raise HTTPException(status_code=422, detail="Checkpoint resume currently supports Isaac Lab only.")
            resume_path = _checkpoint_path(request.resume_checkpoint)
        active = _active_run()
        if active:
            raise HTTPException(status_code=409, detail=f"Training run {active['id']} is already active.")
        RUNS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        user = re.sub(r"[^A-Za-z0-9_-]", "-", request.user).strip("-")[:32] or "Guest"
        default_name = "Isaac training" if request.engine == "isaaclab" else "MuJoCo training"
        display_name = (request.name or "").strip()[:48] or f"{default_name} {time.strftime('%Y-%m-%d %H:%M')}"
        name_slug = re.sub(r"[^A-Za-z0-9_-]", "-", display_name).strip("-")[:28] or "training"
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{name_slug}-{uuid.uuid4().hex[:6]}"
        run_directory = RUNS_DIRECTORY / run_id
        run_directory.mkdir(mode=0o750)
        initial_config = {
            "engine": request.engine,
            "name": display_name,
            "user": user,
            "iterations": request.iterations,
            "seed": request.seed,
            **({"num_envs": request.num_envs} if request.engine == "isaaclab" else {
                "trajectories": request.trajectories,
                "horizon": request.horizon,
            }),
            "resume_checkpoint": request.resume_checkpoint,
        }
        (run_directory / "config.json").write_text(json.dumps(initial_config), encoding="utf-8")
        (run_directory / "status.json").write_text(
            json.dumps({"status": "starting", "started_at": time.time(), "iteration": 0}),
            encoding="utf-8",
        )
        log_file = (run_directory / "training.log").open("ab", buffering=0)
        worker = ISAAC_WORKER if request.engine == "isaaclab" else WORKER
        command = [
            sys.executable,
            str(worker),
            "--run-directory", str(run_directory),
            "--iterations", str(request.iterations),
            "--seed", str(request.seed),
        ]
        if request.engine == "isaaclab":
            command.extend(["--num-envs", str(request.num_envs)])
            if resume_path:
                command.extend([
                    "--resume-checkpoint", str(resume_path),
                    "--resume-checkpoint-id", request.resume_checkpoint,
                ])
        else:
            command.extend(["--trajectories", str(request.trajectories), "--horizon", str(request.horizon)])
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
