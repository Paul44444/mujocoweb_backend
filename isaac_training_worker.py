"""Bounded wrapper around Isaac Lab's official RSL-RL Franka trainer.

The FastAPI backend intentionally stays in its lightweight Python environment.
This worker launches Isaac Lab with its own interpreter, mirrors the console to
the per-run log, and translates RSL-RL progress into the common web metrics
format used by the training dashboard.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parent
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_ROOT", "/home/paul/IsaacLab"))
ISAACLAB_PYTHON = Path(
    os.environ.get("ISAACLAB_PYTHON", "/home/paul/miniconda3/envs/env_isaaclab1/bin/python")
)
TRAIN_SCRIPT = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
VISION_TRAIN_SCRIPT = ROOT / "isaac_vision_train.py"
TASKS = {
    "state": "Isaac-Lift-Cube-Franka-v0",
    "vision": "Isaac-Lift-Cube-Franka-Vision-v0",
    "labware_lift": "Isaac-Franka-Test-Tube-Lift-v0",
    "labware": "Isaac-Franka-Labware-Placement-v0",
}
EXPERIMENTS = {
    "state": "franka_lift",
    "vision": "franka_lift_vision",
    "labware_lift": "franka_test_tube_lift",
    "labware": "franka_labware_placement",
}
# This installed Isaac Lab release parses ``--experiment_name`` but keeps the
# task's registered experiment name. Keep discovery aligned with that output.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
stopping = False
child: subprocess.Popen[str] | None = None


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def stop_worker(_signum: int, _frame: object) -> None:
    global stopping
    stopping = True
    if child and child.poll() is None:
        child.terminate()


def find_output_directory(run_name: str, experiment: str) -> Path | None:
    root = ISAACLAB_ROOT / "logs" / "rsl_rl" / experiment
    matches = sorted(root.glob(f"*_{run_name}"), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def sync_checkpoints(output_directory: Path | None, destination: Path) -> list[str]:
    if output_directory is None:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    for source in output_directory.glob("model_*.pt"):
        target = destination / source.name
        if not target.exists() or target.stat().st_mtime_ns != source.stat().st_mtime_ns:
            shutil.copy2(source, target)
    return sorted(path.name for path in destination.glob("model_*.pt"))


def number(line: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line, re.I)
    return float(match.group(1)) if match else None


def main() -> None:
    global child
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--num-envs", type=int, choices=(16, 32, 64, 128, 256, 512), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--isaac-task", choices=("state", "vision", "labware_lift", "labware"), default="state")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--resume-checkpoint-id")
    args = parser.parse_args()
    task = TASKS[args.isaac_task]
    experiment = EXPERIMENTS[args.isaac_task]
    train_script = VISION_TRAIN_SCRIPT if args.isaac_task in {"vision", "labware_lift", "labware"} else TRAIN_SCRIPT

    run_directory = Path(args.run_directory).resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoints_directory = run_directory / "checkpoints"
    status_path = run_directory / "status.json"
    metrics_path = run_directory / "metrics.json"
    started_at = time.time()
    run_name = f"web_{run_directory.name}"
    try:
        existing_config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        existing_config = {}
    config = {
        **existing_config,
        "engine": "isaaclab",
        "environment": task,
        "isaac_task": args.isaac_task,
        "algorithm": "RSL-RL PPO",
        "iterations": args.iterations,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "resume_checkpoint": args.resume_checkpoint_id,
    }
    atomic_json(run_directory / "config.json", config)
    atomic_json(metrics_path, {"metrics": []})
    atomic_json(status_path, {"status": "starting", "started_at": started_at, "iteration": 0})
    signal.signal(signal.SIGTERM, stop_worker)
    signal.signal(signal.SIGINT, stop_worker)

    metrics: list[dict] = []
    current: dict = {}
    first_raw_iteration: int | None = None
    output_directory: Path | None = None
    fatal_error: str | None = None

    def publish_current() -> None:
        nonlocal current, output_directory
        if not current.get("iteration"):
            return
        iteration = int(current["iteration"])
        previous = metrics[-1] if metrics else {}
        metric = {
            "iteration": iteration,
            "reward_mean": float(current.get("reward_mean", previous.get("reward_mean", 0.0))),
            "reward_std": 0.0,
            "reward_min": 0.0,
            "reward_max": 0.0,
            "kl_distance": 0.0,
            "surrogate_improvement": 0.0,
            "vf_error_before": 0.0,
            "vf_error_after": float(current.get("loss", 0.0)),
            "success_rate": float(current.get("success_rate", 0.0)),
            "samples": int(current.get("samples", iteration * args.num_envs * 24)),
            "seconds": float(current.get("seconds", 0.0)),
            "fps": int(current.get("fps", 0)),
        }
        if metrics and metrics[-1]["iteration"] == iteration:
            metrics[-1] = metric
        else:
            metrics.append(metric)
        atomic_json(metrics_path, {"metrics": metrics})
        output_directory = output_directory or find_output_directory(run_name, experiment)
        checkpoint_names = sync_checkpoints(output_directory, checkpoints_directory)
        atomic_json(status_path, {
            "status": "training",
            "started_at": started_at,
            "iteration": iteration,
            "latest_checkpoint": checkpoint_names[-1] if checkpoint_names else None,
        })

    try:
        if not ISAACLAB_PYTHON.is_file():
            raise FileNotFoundError(f"Isaac Lab Python not found: {ISAACLAB_PYTHON}")
        if not train_script.is_file():
            raise FileNotFoundError(f"Isaac Lab training script not found: {train_script}")
        command = [
            str(ISAACLAB_PYTHON), "-u", str(train_script),
            "--task", task,
            "--num_envs", str(args.num_envs),
            "--max_iterations", str(args.iterations),
            "--seed", str(args.seed),
            "--experiment_name", experiment,
            "--run_name", run_name,
            "--headless",
        ]
        if args.resume_checkpoint:
            resume_checkpoint = Path(args.resume_checkpoint).resolve()
            if not resume_checkpoint.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
            resume_directory = ISAACLAB_ROOT / "logs" / "rsl_rl" / experiment / f"resume_{run_name}"
            resume_directory.mkdir(parents=True, exist_ok=True)
            staged_checkpoint = resume_directory / resume_checkpoint.name
            shutil.copy2(resume_checkpoint, staged_checkpoint)
            command.extend([
                "--resume",
                "--load_run", f"^{re.escape(resume_directory.name)}$",
                "--checkpoint", f"^{re.escape(staged_checkpoint.name)}$",
            ])
        environment = os.environ.copy()
        environment["PYTHONPATH"] = f"{ROOT}:{environment.get('PYTHONPATH', '')}"
        environment["PYTHONUNBUFFERED"] = "1"
        # run-local.sh deliberately leaves this empty for the MuJoCo process.
        # An empty CUDA_VISIBLE_DEVICES hides every GPU from the child process.
        if not environment.get("CUDA_VISIBLE_DEVICES", "").strip():
            environment["CUDA_VISIBLE_DEVICES"] = "0"
        child = subprocess.Popen(
            command,
            cwd=str(ISAACLAB_ROOT),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        atomic_json(status_path, {"status": "starting", "started_at": started_at, "iteration": 0, "pid": child.pid})
        assert child.stdout is not None
        for raw_line in child.stdout:
            print(raw_line, end="", flush=True)
            line = ANSI_ESCAPE.sub("", raw_line).strip()
            if any(marker in line for marker in (
                "No CUDA GPUs are available",
                "no CUDA-capable device is detected",
                "No CUDA devices found",
                "Error executing job with overrides",
            )):
                fatal_error = line
            iteration_match = re.search(r"Learning iteration\s+(\d+)\s*/\s*(\d+)", line)
            if iteration_match:
                publish_current()
                raw_iteration = int(iteration_match.group(1))
                if first_raw_iteration is None:
                    first_raw_iteration = raw_iteration
                current = {"iteration": raw_iteration - first_raw_iteration + 1}
            fps_match = re.search(r"Computation:\s*(\d+)\s+steps/s", line)
            if fps_match:
                current["fps"] = int(fps_match.group(1))
            for label in ("Mean reward", "Mean value_function loss", "Mean value loss", "value_function"):
                value = number(line, label)
                if value is not None:
                    current["reward_mean" if label == "Mean reward" else "loss"] = value
            success = number(line, "Episode_Reward/lifting")
            if success is not None:
                current["success_rate"] = success
            elapsed = number(line, "Iteration time")
            if elapsed is not None:
                current["seconds"] = elapsed
            if line.startswith("Total timesteps:"):
                value = number(line, "Total timesteps")
                if value is not None:
                    current["samples"] = int(value)
                publish_current()
            if stopping:
                break
        return_code = child.wait()
        publish_current()
        output_directory = output_directory or find_output_directory(run_name, experiment)
        checkpoint_names = sync_checkpoints(output_directory, checkpoints_directory)
        if stopping:
            final_status = "stopped"
        elif fatal_error:
            raise RuntimeError(fatal_error)
        elif return_code == 0:
            final_status = "completed"
        else:
            raise RuntimeError(f"Isaac Lab trainer exited with status {return_code}")
        atomic_json(status_path, {
            "status": final_status,
            "started_at": started_at,
            "finished_at": time.time(),
            "iteration": len(metrics),
            "latest_checkpoint": checkpoint_names[-1] if checkpoint_names else None,
            "output_directory": str(output_directory) if output_directory else None,
        })
    except Exception as exc:
        atomic_json(status_path, {
            "status": "failed",
            "started_at": started_at,
            "finished_at": time.time(),
            "iteration": len(metrics),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


if __name__ == "__main__":
    main()
