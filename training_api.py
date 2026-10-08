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
ISAAC_TASK_SELECTION = ISAAC_OUTPUT_DIRECTORY / "selected_task.json"
ISAAC_STATUS_PATH = ISAAC_OUTPUT_DIRECTORY / "status.json"
ISAAC_FRAME_PATH = ISAAC_OUTPUT_DIRECTORY / "frame.jpg"
ISAAC_METADATA_PATH = ISAAC_OUTPUT_DIRECTORY / "metadata.json"
ISAAC_CONTROL_DIRECTORY = ISAAC_OUTPUT_DIRECTORY / "commands"
ISAAC_DEMO_DIRECTORY = Path(os.environ.get("ISAAC_DEMO_DIRECTORY", Path.home() / ".local/share/mujocoweb/demos"))
ISAAC_DESKTOP_DIRECTORY = Path("/tmp/mujocoweb-isaac-desktop")
ISAAC_DESKTOP_STATUS = ISAAC_DESKTOP_DIRECTORY / "launcher-status.json"
ISAACLAB_PYTHON = Path(os.environ.get("ISAACLAB_PYTHON", "/home/paul/miniconda3/envs/env_isaaclab1/bin/python"))
ISAAC_WEB_WORKER = ROOT / "isaac_web_worker.py"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
CHECKPOINT_PATTERN = re.compile(r"^model_\d+\.pt$")
router = APIRouter(prefix="/api/training", tags=["training"])
manager_lock = threading.Lock()


class StartTrainingRequest(BaseModel):
    engine: Literal["mujoco", "isaaclab"] = "mujoco"
    user: str = Field(default="Guest", min_length=1, max_length=32)
    name: Optional[str] = Field(default=None, max_length=48)
    iterations: int = Field(default=10, ge=1)
    trajectories: int = Field(default=3, ge=1, le=8)
    horizon: int = Field(default=200, ge=20, le=500)
    seed: int = Field(default=123, ge=0, le=2_147_483_647)
    num_envs: Literal[16, 32, 64, 128, 256, 512] = 256
    resume_checkpoint: Optional[str] = Field(default=None, max_length=160)
    isaac_task: Literal["state", "vision", "labware_lift", "labware"] = "state"
    demo_ids: List[str] = Field(default_factory=list, max_length=8)
    bc_epochs: int = Field(default=200, ge=1, le=2000)


class SelectIsaacTaskRequest(BaseModel):
    task: Literal["state", "vision", "labware_lift", "labware"]


class SelectCheckpointRequest(BaseModel):
    checkpoint: Optional[str] = Field(default=None, max_length=160)


class DeleteCheckpointRequest(BaseModel):
    checkpoint: str = Field(min_length=1, max_length=160)


def _read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return fallback


def _write_json(path: Path, value: Dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    os.replace(temporary, path)


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


def _demonstration_path(demo_id: str, user: str, isaac_task: str) -> Path:
    demo_user, separator, demo_name = demo_id.partition("/")
    if (
        not separator
        or demo_user != user
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", demo_name)
    ):
        raise HTTPException(status_code=400, detail="Invalid demonstration ID.")
    path = (ISAAC_DEMO_DIRECTORY / demo_user / f"{demo_name}.npz").resolve()
    if ISAAC_DEMO_DIRECTORY.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Demonstration {demo_id} was not found.")
    metadata = _read_json(path.with_suffix(".json"), {})
    if metadata.get("task") != isaac_task:
        raise HTTPException(
            status_code=422,
            detail=f"Demonstration {demo_id} belongs to task {metadata.get('task', 'unknown')}, not {isaac_task}.",
        )
    return path


def _run_display_name(run_directory: Path, config: Dict[str, object]) -> str:
    configured_name = str(config.get("name") or "").strip()
    if configured_name:
        return configured_name
    legacy_match = re.fullmatch(r"\d{8}-\d{6}-(.+)-[0-9a-f]{6}", run_directory.name)
    return legacy_match.group(1) if legacy_match else run_directory.name


def _checkpoint_payloads(isaac_task: Optional[str] = None) -> List[Dict[str, object]]:
    checkpoints = []
    if not RUNS_DIRECTORY.is_dir():
        return checkpoints
    for run_directory in sorted(RUNS_DIRECTORY.iterdir(), reverse=True):
        config = _read_json(run_directory / "config.json", {})
        if not run_directory.is_dir() or config.get("engine") != "isaaclab":
            continue
        run_task = str(config.get("isaac_task") or "state")
        if isaac_task == "state" and run_task != "state":
            continue
        if isaac_task == "vision" and run_task not in {"state", "vision"}:
            continue
        if isaac_task == "labware_lift" and run_task != "labware_lift":
            continue
        if isaac_task == "labware" and run_task not in {"labware_lift", "labware"}:
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
                "isaac_task": run_task,
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
    if status.get("status") in {"starting", "training", "paused"} and pid and not _is_alive(pid):
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
        if payload["status"].get("status") in {"starting", "training", "paused"}:
            return payload
    return None


@router.get("/runs")
def list_training_runs(
    engine: Optional[Literal["mujoco", "isaaclab"]] = Query(default=None),
    isaac_task: Optional[Literal["state", "vision", "labware_lift", "labware"]] = Query(default=None),
) -> Dict[str, List[Dict[str, object]]]:
    RUNS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    runs = [_run_payload(path) for path in sorted(RUNS_DIRECTORY.iterdir(), reverse=True) if path.is_dir()]
    if engine:
        runs = [run for run in runs if run.get("config", {}).get("engine", "mujoco") == engine]
    if isaac_task:
        runs = [run for run in runs if run.get("config", {}).get("isaac_task", "state") == isaac_task]
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
def list_checkpoints(
    isaac_task: Optional[Literal["state", "vision", "labware_lift", "labware"]] = Query(default=None),
) -> Dict[str, object]:
    selected = _read_json(ISAAC_POLICY_SELECTION, {}).get("id")
    return {"checkpoints": _checkpoint_payloads(isaac_task), "selected": selected}


@router.get("/isaac-task")
def get_isaac_task() -> Dict[str, object]:
    task = _read_json(ISAAC_TASK_SELECTION, {}).get("task", "state")
    status = _read_json(ISAAC_STATUS_PATH, {"status": "unknown"})
    return {"task": task, "worker": status}


@router.post("/isaac-task")
def select_isaac_task(request: SelectIsaacTaskRequest, background_tasks: BackgroundTasks) -> Dict[str, object]:
    ISAAC_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    current = _read_json(ISAAC_TASK_SELECTION, {}).get("task", "state")
    if current == request.task and _read_json(ISAAC_STATUS_PATH, {}).get("status") == "ready":
        return {"task": request.task, "status": "ready", "restarting": False}
    _write_json(ISAAC_TASK_SELECTION, {"task": request.task, "selected_at": time.time()})
    _write_json(ISAAC_POLICY_SELECTION, {"id": None, "path": None, "isaac_task": request.task})
    _write_json(ISAAC_STATUS_PATH, {"status": "restarting", "task": request.task})
    background_tasks.add_task(_restart_isaac_runtime)
    return {"task": request.task, "status": "restarting", "restarting": True, "estimated_seconds": 45}


def _run_isaac_desktop(task: str) -> None:
    ISAAC_DESKTOP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    log_path = ISAAC_DESKTOP_DIRECTORY / "desktop.log"
    process = None
    try:
        subprocess.run(["systemctl", "--user", "stop", "mujocoweb-isaac.service"], timeout=45, check=False)
        environment = os.environ.copy()
        environment.setdefault("DISPLAY", ":1")
        environment.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")
        environment.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
        with log_path.open("ab", buffering=0) as log_file:
            process = subprocess.Popen([
                str(ISAACLAB_PYTHON), str(ISAAC_WEB_WORKER),
                "--device", "cuda:0", "--desktop", "--task-mode", task,
                "--selection-directory", str(ISAAC_OUTPUT_DIRECTORY),
                "--output-directory", str(ISAAC_DESKTOP_DIRECTORY),
            ], cwd="/home/paul/IsaacLab", env=environment, stdin=subprocess.DEVNULL,
               stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        _write_json(ISAAC_DESKTOP_STATUS, {"status": "running", "pid": process.pid, "task": task})
        return_code = process.wait()
        _write_json(ISAAC_DESKTOP_STATUS, {"status": "closed", "return_code": return_code, "task": task})
    except Exception as exc:
        _write_json(ISAAC_DESKTOP_STATUS, {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "task": task})
    finally:
        subprocess.run(["systemctl", "--user", "start", "mujocoweb-isaac.service"], timeout=45, check=False)


@router.post("/isaac-desktop")
def open_isaac_desktop(background_tasks: BackgroundTasks) -> Dict[str, object]:
    active = _active_run()
    if active:
        raise HTTPException(status_code=409, detail="Pause or cancel the active training run before opening Isaac Lab Desktop.")
    current = _read_json(ISAAC_DESKTOP_STATUS, {})
    pid = int(current.get("pid", 0) or 0)
    if current.get("status") in {"starting", "running"} and pid and _is_alive(pid):
        return {"status": "running", "task": current.get("task", "state")}
    task = _read_json(ISAAC_TASK_SELECTION, {}).get("task", "state")
    ISAAC_DESKTOP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(ISAAC_DESKTOP_STATUS, {"status": "starting", "task": task})
    background_tasks.add_task(_run_isaac_desktop, task)
    return {"status": "starting", "task": task, "estimated_seconds": 30}


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
    current_task = _read_json(ISAAC_TASK_SELECTION, {}).get("task", "state")
    if request.checkpoint:
        path = _checkpoint_path(request.checkpoint)
        run_config = _read_json(path.parent.parent / "config.json", {})
        checkpoint_task = run_config.get("isaac_task", "state")
        compatible = (
            checkpoint_task == current_task
            or (current_task == "vision" and checkpoint_task == "state")
            or (current_task == "labware" and checkpoint_task == "labware_lift")
        )
        if not compatible:
            raise HTTPException(status_code=409, detail="This checkpoint is not observation-compatible with the active Isaac task.")
        payload = {"id": request.checkpoint, "path": str(path), "isaac_task": checkpoint_task, "selected_at": time.time()}
    else:
        payload = {"id": None, "path": None, "isaac_task": current_task, "selected_at": time.time()}
    temporary = ISAAC_POLICY_SELECTION.with_suffix(".json.next")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, ISAAC_POLICY_SELECTION)
    current_status = _read_json(ISAAC_STATUS_PATH, {})
    # A compatible actor can be installed in the warm playback environment
    # regardless of whether it currently runs the scripted preview or another
    # trained actor. Restarting Isaac Sim here needlessly competes with an
    # active trainer for several gigabytes of GPU memory.
    hot_swap = bool(
        payload["id"]
        and current_status.get("status") == "ready"
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
        user = re.sub(r"[^A-Za-z0-9_-]", "-", request.user).strip("-")[:32] or "Guest"
        if request.demo_ids and request.engine != "isaaclab":
            raise HTTPException(status_code=422, detail="Demonstration warm-start currently supports Isaac Lab only.")
        demo_paths = [
            _demonstration_path(demo_id, user, request.isaac_task)
            for demo_id in request.demo_ids
        ]
        resume_path = None
        if request.resume_checkpoint:
            if request.engine != "isaaclab":
                raise HTTPException(status_code=422, detail="Checkpoint resume currently supports Isaac Lab only.")
            resume_path = _checkpoint_path(request.resume_checkpoint)
            resume_config = _read_json(resume_path.parent.parent / "config.json", {})
            resume_task = resume_config.get("isaac_task", "state")
            compatible = (
                resume_task == request.isaac_task
                or (request.isaac_task == "vision" and resume_task == "state")
                or (request.isaac_task == "labware" and resume_task == "labware_lift")
            )
            if not compatible:
                raise HTTPException(status_code=422, detail="The resume checkpoint is not observation-compatible with this Isaac task.")
        active = _active_run()
        if active:
            raise HTTPException(status_code=409, detail=f"Training run {active['id']} is already active.")
        RUNS_DIRECTORY.mkdir(parents=True, exist_ok=True)
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
            "isaac_task": request.isaac_task if request.engine == "isaaclab" else None,
            "demo_ids": request.demo_ids,
            "bc_epochs": request.bc_epochs if request.demo_ids else 0,
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
            command.extend(["--num-envs", str(request.num_envs), "--isaac-task", request.isaac_task])
            if demo_paths:
                command.extend(["--bc-epochs", str(request.bc_epochs)])
                for demo_path in demo_paths:
                    command.extend(["--demo-path", str(demo_path)])
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
    """Freeze the complete trainer process tree so it can resume exactly in place."""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="Invalid training run ID.")
    run_directory = RUNS_DIRECTORY / run_id
    if not run_directory.is_dir():
        raise HTTPException(status_code=404, detail="Training run not found.")
    with manager_lock:
        payload = _run_payload(run_directory)
        if payload["status"].get("status") == "paused":
            return payload
        if payload["status"].get("status") not in {"starting", "training"}:
            return payload
        pid = int(_read_json(run_directory / "process.json", {}).get("pid", 0) or 0)
        if not pid or not _is_alive(pid):
            raise HTTPException(status_code=409, detail="The training process is no longer running.")
        try:
            os.killpg(pid, signal.SIGSTOP)
        except OSError as exc:
            raise HTTPException(status_code=409, detail="Could not pause the training process.") from exc
        current = _read_json(run_directory / "status.json", {})
        _write_json(run_directory / "status.json", {
            **current,
            "status": "paused",
            "paused_at": time.time(),
        })
        return _run_payload(run_directory)


@router.post("/runs/{run_id}/cancel")
def cancel_training(run_id: str) -> Dict[str, object]:
    """Terminate an active trainer while retaining metrics and checkpoints."""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="Invalid training run ID.")
    run_directory = RUNS_DIRECTORY / run_id
    if not run_directory.is_dir():
        raise HTTPException(status_code=404, detail="Training run not found.")
    with manager_lock:
        payload = _run_payload(run_directory)
        state = payload["status"].get("status")
        if state not in {"starting", "training", "paused"}:
            return payload
        pid = int(_read_json(run_directory / "process.json", {}).get("pid", 0) or 0)
        if pid and _is_alive(pid):
            try:
                # A SIGSTOP-paused process cannot handle SIGTERM until it is
                # continued. The complete process group contains the wrapper
                # and Isaac/RSL-RL child, but no playback process.
                if state == "paused":
                    os.killpg(pid, signal.SIGCONT)
                os.killpg(pid, signal.SIGTERM)
                deadline = time.monotonic() + 8.0
                while _is_alive(pid) and time.monotonic() < deadline:
                    time.sleep(0.1)
                if _is_alive(pid):
                    os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                raise HTTPException(status_code=409, detail="Could not cancel the training process.") from exc
        current = _read_json(run_directory / "status.json", {})
        _write_json(run_directory / "status.json", {
            **current,
            "status": "cancelled",
            "cancelled_at": time.time(),
            "error": None,
        })
        return _run_payload(run_directory)


@router.post("/runs/{run_id}/continue")
def continue_training(run_id: str) -> Dict[str, object]:
    """Resume a SIGSTOP-paused trainer without replacing its state or metrics."""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="Invalid training run ID.")
    run_directory = RUNS_DIRECTORY / run_id
    if not run_directory.is_dir():
        raise HTTPException(status_code=404, detail="Training run not found.")
    with manager_lock:
        payload = _run_payload(run_directory)
        if payload["status"].get("status") != "paused":
            return payload
        pid = int(_read_json(run_directory / "process.json", {}).get("pid", 0) or 0)
        if not pid or not _is_alive(pid):
            raise HTTPException(status_code=409, detail="The paused training process is no longer available. Resume from its latest checkpoint instead.")
        current = _read_json(run_directory / "status.json", {})
        _write_json(run_directory / "status.json", {
            **current,
            "status": "training",
            "continued_at": time.time(),
        })
        try:
            os.killpg(pid, signal.SIGCONT)
        except OSError as exc:
            _write_json(run_directory / "status.json", current)
            raise HTTPException(status_code=409, detail="Could not continue the training process.") from exc
        return _run_payload(run_directory)
