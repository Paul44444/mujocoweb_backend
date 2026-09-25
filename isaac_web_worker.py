"""Persistent Isaac Lab renderer for the public web demo.

The worker deliberately runs in Isaac Lab's own Python environment.  It keeps
Isaac Sim warm and publishes the newest JPEG plus lightweight metadata through
atomic files.  The FastAPI process can therefore remain in the DAPG/MuJoCo
environment and relay either engine through the same browser WebSocket.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Warm Isaac Lab web renderer")
parser.add_argument("--output-directory", default="/tmp/mujocoweb-isaac")
parser.add_argument("--task", default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--jpeg-quality", type=int, default=86)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True

launcher = AppLauncher(args)
simulation_app = launcher.app

# Isaac/Omniverse modules must be imported only after AppLauncher.
import gymnasium as gym
import numpy as np
from PIL import Image
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


output_directory = Path(args.output_directory)
output_directory.mkdir(parents=True, exist_ok=True)
frame_path = output_directory / "frame.jpg"
metadata_path = output_directory / "metadata.json"
status_path = output_directory / "status.json"
stopping = False


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def atomic_json(path: Path, data: dict) -> None:
    atomic_write(path, json.dumps(data, separators=(",", ":")).encode("utf-8"))


def stop_worker(_signum: int, _frame: object) -> None:
    global stopping
    stopping = True


signal.signal(signal.SIGTERM, stop_worker)
signal.signal(signal.SIGINT, stop_worker)
atomic_json(status_path, {"status": "starting", "task": args.task})

env = None
started_at = time.monotonic()
try:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = 42
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "object_pose"):
        env_cfg.commands.object_pose.debug_vis = False
    env_cfg.scene.web_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/WebCamera",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.4, 1.8, 1.2),
            rot=(-0.1393, 0.2025, 0.8185, -0.5192),
            convention="ros",
        ),
    )
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()
    atomic_json(
        status_path,
        {
            "status": "ready",
            "task": args.task,
            "startup_seconds": round(time.monotonic() - started_at, 3),
        },
    )

    step = 0
    episode = 0
    with torch.inference_mode():
        while simulation_app.is_running() and not stopping:
            phase = step * 0.018
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # A slow, bounded demonstration motion keeps the first web version
            # visibly alive while a trained checkpoint is wired in next.
            actions[:, 0] = 0.22 * math.sin(phase)
            actions[:, 1] = 0.16 * math.sin(phase * 0.73 + 0.8)
            actions[:, 3] = 0.18 * math.sin(phase * 0.51)
            actions[:, 5] = 0.12 * math.cos(phase * 0.67)
            actions[:, 7] = 1.0 if math.sin(phase * 0.35) > 0 else -1.0
            _, rewards, terminated, truncated, _ = env.step(actions)

            frame = (
                env.unwrapped.scene["web_camera"]
                .data.output["rgb"][0]
                .detach()
                .cpu()
                .numpy()
            )
            image_path = frame_path.with_suffix(".jpg.next")
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                image_path,
                format="JPEG",
                quality=max(50, min(95, args.jpeg_quality)),
            )
            os.replace(image_path, frame_path)

            simulation_time = step * float(env.unwrapped.step_dt)
            atomic_json(
                metadata_path,
                {
                    "engine": "isaaclab",
                    "task": args.task,
                    "episode": episode,
                    "step": step,
                    "reward": float(rewards[0].item()),
                    "simulation_time": simulation_time,
                    "mode": "scripted_preview",
                },
            )
            step += 1
            if bool(terminated[0].item() or truncated[0].item()):
                env.reset()
                step = 0
                episode += 1
finally:
    atomic_json(status_path, {"status": "stopped", "task": args.task})
    if env is not None:
        env.close()
    simulation_app.close()
