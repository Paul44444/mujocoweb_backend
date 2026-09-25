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
import omni.usd
from PIL import Image
import torch

import isaaclab.sim as sim_utils
from isaaclab_assets.robots import KUKA_ALLEGRO_CFG
from isaaclab.sensors import CameraCfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


output_directory = Path(args.output_directory)
output_directory.mkdir(parents=True, exist_ok=True)
frame_path = output_directory / "frame.jpg"
metadata_path = output_directory / "metadata.json"
status_path = output_directory / "status.json"
control_directory = output_directory / "commands"
control_directory.mkdir(parents=True, exist_ok=True)
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
    camera = env.unwrapped.scene["web_camera"]
    default_camera_target = torch.tensor(
        [[0.45, 0.0, 0.45]], dtype=torch.float32, device=env.unwrapped.device
    )
    camera_target = default_camera_target.clone()
    default_camera = (62.0, 20.0, 2.15)
    camera_state = {
        "azimuth": default_camera[0],
        "elevation": default_camera[1],
        "distance": default_camera[2],
        "position": [1.4, 1.8, 1.2],
    }
    web_assets = []

    def euler_degrees_to_quaternion(rotation: list[float]) -> tuple[float, float, float, float]:
        roll, pitch, yaw = (math.radians(float(value)) * 0.5 for value in rotation)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )

    def update_camera_pose() -> None:
        azimuth = math.radians(camera_state["azimuth"])
        elevation = math.radians(camera_state["elevation"])
        horizontal_distance = camera_state["distance"] * math.cos(elevation)
        camera_position = torch.tensor(
            [[
                float(camera_target[0, 0]) + horizontal_distance * math.cos(azimuth),
                float(camera_target[0, 1]) + horizontal_distance * math.sin(azimuth),
                float(camera_target[0, 2]) + camera_state["distance"] * math.sin(elevation),
            ]],
            dtype=torch.float32,
            device=env.unwrapped.device,
        )
        camera.set_world_poses_from_view(camera_position, camera_target)
        camera_state["position"] = camera_position[0].tolist()

    def render_camera_metadata() -> dict:
        position = np.asarray(camera_state["position"], dtype=np.float64)
        target = np.asarray(camera_target[0].tolist(), dtype=np.float64)
        forward = target - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        half_width = 20.955 / (2.0 * 24.0)
        half_height = half_width * 720.0 / 1280.0
        return {
            "position": position.tolist(),
            "forward": forward.tolist(),
            "up": up.tolist(),
            "near": 1.0,
            "top": half_height,
            "bottom": -half_height,
            "center": 0.0,
        }

    def spawn_web_asset(command: dict) -> None:
        if len(web_assets) >= 32:
            print("Ignoring Isaac web asset: the 32-object limit was reached", flush=True)
            return
        asset = command["asset"]
        position = [float(value) for value in command["position"]]
        rotation = [float(value) for value in command.get("rotation", [0.0, 0.0, 0.0])]
        scale = [float(value) for value in command.get("scale", [0.04, 0.04, 0.04])]
        if asset != "kuka_allegro":
            position[2] = max(position[2], scale[2] + 0.006)
        default_colors = {
            "box": (0.15, 0.55, 0.95),
            "sphere": (0.95, 0.35, 0.18),
            "cylinder": (0.35, 0.8, 0.35),
        }
        color = tuple(float(value) for value in command.get("color", default_colors.get(asset, (0.8, 0.45, 0.12))))
        prim_path = f"/World/envs/env_0/WebAsset_{len(web_assets) + 1}"
        orientation = euler_degrees_to_quaternion(rotation)
        if asset == "kuka_allegro":
            spawn_cfg = KUKA_ALLEGRO_CFG.spawn
            spawn_cfg.func(
                prim_path,
                spawn_cfg,
                translation=tuple(position),
                orientation=orientation,
            )
            web_assets.append({
                "id": command["id"], "asset": asset, "prim_path": prim_path,
                "position": position, "rotation": rotation, "scale": [1.0, 1.0, 1.0],
            })
            print(f"Spawned Isaac scene robot {command['id']} ({asset}) at {position}", flush=True)
            return
        common = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=3.0,
                disable_gravity=False,
            ),
            "mass_props": sim_utils.MassPropertiesCfg(mass=0.12),
            "collision_props": sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.004,
                rest_offset=0.0,
            ),
            "visual_material": sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, metallic=0.05, roughness=0.35
            ),
            "physics_material": sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.45,
                restitution=0.12,
            ),
        }
        if asset == "box":
            spawn_cfg = sim_utils.CuboidCfg(size=tuple(2.0 * value for value in scale), **common)
        elif asset == "sphere":
            spawn_cfg = sim_utils.SphereCfg(radius=scale[0], **common)
        else:
            spawn_cfg = sim_utils.CylinderCfg(radius=scale[0], height=2.0 * scale[2], axis="Z", **common)
        spawn_cfg.func(prim_path, spawn_cfg, translation=tuple(position), orientation=orientation)
        web_assets.append({
            "id": command["id"], "asset": asset, "prim_path": prim_path,
            "position": position, "rotation": rotation, "scale": scale,
        })
        print(
            f"Spawned Isaac web asset {command['id']} ({asset}) at {position}",
            flush=True,
        )

    def replace_web_scene(commands: list[dict]) -> None:
        stage = omni.usd.get_context().get_stage()
        for item in reversed(web_assets):
            stage.RemovePrim(item["prim_path"])
        web_assets.clear()
        for command in commands:
            try:
                spawn_web_asset(command)
            except Exception as exc:
                print(
                    f"Could not spawn Isaac scene asset {command.get('id', '?')}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        if any(command.get("asset") == "kuka_allegro" for command in commands):
            camera_target[0] = torch.tensor(
                [0.15, 0.0, 0.72], dtype=torch.float32, device=env.unwrapped.device
            )
            camera_state["distance"] = max(camera_state["distance"], 3.4)
            update_camera_pose()
        print(f"Replaced Isaac web scene with {len(web_assets)} assets", flush=True)

    def apply_camera_commands() -> None:
        camera_changed = False
        for command_path in sorted(control_directory.glob("*.json")):
            try:
                command = json.loads(command_path.read_text(encoding="utf-8"))
                command_type = command.get("type")
                if command_type == "camera_orbit":
                    camera_state["azimuth"] -= float(command["delta_x"]) * 0.25
                    camera_state["elevation"] = max(
                        -10.0,
                        min(85.0, camera_state["elevation"] + float(command["delta_y"]) * 0.2),
                    )
                    camera_changed = True
                elif command_type == "camera_pan":
                    camera_metadata = render_camera_metadata()
                    forward = np.asarray(camera_metadata["forward"], dtype=np.float64)
                    up = np.asarray(camera_metadata["up"], dtype=np.float64)
                    right = np.cross(forward, up)
                    scale = camera_state["distance"] * 0.0012
                    translation = (
                        -right * float(command["delta_x"]) * scale
                        + up * float(command["delta_y"]) * scale
                    )
                    target = np.asarray(camera_target[0].tolist()) + translation
                    target = np.clip(target, (-1.5, -2.0, -0.5), (2.5, 2.0, 2.0))
                    camera_target[0] = torch.as_tensor(
                        target, dtype=torch.float32, device=env.unwrapped.device
                    )
                    camera_changed = True
                elif command_type == "camera_zoom":
                    camera_state["distance"] = max(
                        0.65,
                        min(5.0, camera_state["distance"] * (1.12 ** float(command["delta"]))),
                    )
                    camera_changed = True
                elif command_type == "camera_reset":
                    camera_state["azimuth"], camera_state["elevation"], camera_state["distance"] = default_camera
                    camera_target.copy_(default_camera_target)
                    camera_changed = True
                elif command_type == "scene_spawn":
                    spawn_web_asset(command)
                elif command_type == "scene_replace":
                    replace_web_scene(command.get("assets", []))
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                print(f"Ignoring invalid Isaac web command: {exc}", flush=True)
            except Exception as exc:
                print(f"Isaac web command failed: {type(exc).__name__}: {exc}", flush=True)
            finally:
                try:
                    command_path.unlink()
                except FileNotFoundError:
                    pass
        if camera_changed:
            update_camera_pose()

    update_camera_pose()
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
            apply_camera_commands()
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
                    "camera": {
                        "azimuth": camera_state["azimuth"],
                        "elevation": camera_state["elevation"],
                        "distance": camera_state["distance"],
                        "lookat": camera_target[0].tolist(),
                    },
                    "render_camera": render_camera_metadata(),
                    "web_assets": web_assets,
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
