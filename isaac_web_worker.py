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
parser.add_argument("--max-fps", type=float, default=20.0)
parser.add_argument("--training-max-fps", type=float, default=5.0)
parser.add_argument("--desktop", action="store_true", help="Open Isaac Sim as a local desktop window")
parser.add_argument("--task-mode", choices=("state", "vision", "labware_lift", "labware"))
parser.add_argument("--selection-directory")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = not args.desktop

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
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import CameraCfg
import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
import isaac_vision_task
import isaac_labware_task  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


output_directory = Path(args.output_directory)
output_directory.mkdir(parents=True, exist_ok=True)
frame_path = output_directory / "frame.jpg"
vision_frame_path = output_directory / "vision_frame.jpg"
metadata_path = output_directory / "metadata.json"
status_path = output_directory / "status.json"
control_directory = output_directory / "commands"
training_runs_directory = Path(__file__).resolve().parent / "web_training_runs"
control_directory.mkdir(parents=True, exist_ok=True)
selection_directory = Path(args.selection_directory) if args.selection_directory else output_directory
policy_selection_path = selection_directory / "selected_policy.json"
task_selection_path = selection_directory / "selected_task.json"
task_mode = args.task_mode or "state"
if args.task_mode is None:
    try:
        selected_task = json.loads(task_selection_path.read_text(encoding="utf-8"))
        if selected_task.get("task") in {"vision", "labware_lift", "labware"}:
            task_mode = selected_task["task"]
    except (OSError, ValueError, TypeError):
        pass
policy_selection = {}
try:
    policy_selection = json.loads(policy_selection_path.read_text(encoding="utf-8"))
except (OSError, ValueError, TypeError):
    pass
checkpoint_path = Path(policy_selection.get("path", "")) if policy_selection.get("path") else None
checkpoint_id = policy_selection.get("id") if checkpoint_path and checkpoint_path.is_file() else None
stopping = False
simulation_paused = False
demo_directory = Path(os.environ.get("ISAAC_DEMO_DIRECTORY", Path.home() / ".local/share/mujocoweb/demos"))
demo_directory.mkdir(parents=True, exist_ok=True)


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def atomic_json(path: Path, data: dict) -> None:
    atomic_write(path, json.dumps(data, separators=(",", ":")).encode("utf-8"))


def stop_worker(_signum: int, _frame: object) -> None:
    global stopping
    stopping = True


def load_actor(checkpoint: Path) -> torch.nn.Module:
    """Load only the compact playback actor, avoiding the training-time RSL wrapper."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_state = payload["model_state_dict"]
    actor_state = {key.removeprefix("actor."): value.cpu() for key, value in model_state.items() if key.startswith("actor.")}
    linear_keys = sorted(
        (key for key in actor_state if key.endswith(".weight")),
        key=lambda key: int(key.split(".", 1)[0]),
    )
    layers = []
    for layer_index, key in enumerate(linear_keys):
        output_size, input_size = actor_state[key].shape
        layers.append(torch.nn.Linear(input_size, output_size))
        if layer_index < len(linear_keys) - 1:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers)
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor


signal.signal(signal.SIGTERM, stop_worker)
signal.signal(signal.SIGINT, stop_worker)
atomic_json(status_path, {"status": "starting", "task": task_mode})

env = None
started_at = time.monotonic()
try:
    if task_mode == "vision":
        effective_task = "Isaac-Lift-Cube-Franka-Vision-Play-v0"
    elif task_mode == "labware_lift":
        effective_task = "Isaac-Franka-Test-Tube-Lift-Play-v0"
    elif task_mode == "labware":
        effective_task = "Isaac-Franka-Labware-Placement-Play-v0"
    else:
        effective_task = "Isaac-Lift-Cube-Franka-Play-v0" if checkpoint_id else args.task
    env_cfg = parse_env_cfg(effective_task, device=args.device, num_envs=1)
    env_cfg.seed = 42
    # The public viewer is an interactive session, not a fixed-horizon RL
    # rollout. It ends only through an explicit reset (apart from physical
    # failure terminations such as dropping the task object).
    env_cfg.episode_length_s = 24.0 * 60.0 * 60.0
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
    env = gym.make(effective_task, cfg=env_cfg)
    reset_observations, _ = env.reset()
    camera = env.unwrapped.scene["web_camera"]
    if task_mode == "vision":
        vision_camera = env.unwrapped.scene["vision_camera"]
        vision_camera_side = env.unwrapped.scene["vision_camera_side"]
        perception_target = torch.tensor([[0.48, 0.0, 0.16]], dtype=torch.float32, device=env.unwrapped.device)
        vision_camera.set_world_poses_from_view(
            torch.tensor([[1.25, 0.90, 0.90]], dtype=torch.float32, device=env.unwrapped.device),
            perception_target,
        )
        vision_camera_side.set_world_poses_from_view(
            torch.tensor([[1.0, -0.90, 0.85]], dtype=torch.float32, device=env.unwrapped.device),
            perception_target,
        )
        # Camera images follow new Xforms immediately, while CameraData keeps
        # initialization poses unless explicitly reset/synchronized.
        vision_camera.reset()
        vision_camera_side.reset()
    # Keep camera framing independent per environment.  In particular, the
    # long-standing cube preset stays byte-for-byte unchanged while labware is
    # framed closer and lower around the tube/rack workspace.
    camera_presets = {
        "state": {"target": (0.45, 0.0, 0.45), "orbit": (62.0, 20.0, 2.15)},
        "vision": {"target": (0.45, 0.0, 0.45), "orbit": (62.0, 20.0, 2.15)},
        "labware_lift": {"target": (0.51, 0.04, 0.24), "orbit": (55.0, 24.0, 1.65)},
        "labware": {"target": (0.51, 0.04, 0.24), "orbit": (55.0, 24.0, 1.65)},
    }
    camera_preset = camera_presets.get(task_mode, camera_presets["state"])
    default_camera_target = torch.tensor(
        [camera_preset["target"]], dtype=torch.float32, device=env.unwrapped.device
    )
    camera_target = default_camera_target.clone()
    default_camera = camera_preset["orbit"]
    camera_state = {
        "azimuth": default_camera[0],
        "elevation": default_camera[1],
        "distance": default_camera[2],
        "position": [1.4, 1.8, 1.2],
    }
    inference_policy = None
    policy_observations = reset_observations.get("policy") if isinstance(reset_observations, dict) else reset_observations

    def apply_vision_estimate(observations):
        if task_mode != "vision" or observations is None:
            return observations
        observations[:, 18:21] = isaac_vision_task.camera_object_position(env.unwrapped)
        return observations

    if task_mode == "vision":
        # Render the newly positioned perception camera before the first policy
        # action.  Otherwise Isaac exposes one stale frame from its configured
        # spawn pose and the actor can react to a wildly incorrect coordinate.
        warmup_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(3):
            warmup_observations, _, _, _, _ = env.step(warmup_actions)
        policy_observations = (
            warmup_observations.get("policy")
            if isinstance(warmup_observations, dict)
            else warmup_observations
        )
        policy_observations = apply_vision_estimate(policy_observations)

    if checkpoint_id and checkpoint_path:
        inference_policy = load_actor(checkpoint_path)
        print(f"Loaded Isaac web policy checkpoint {checkpoint_id}", flush=True)
    web_assets = []
    # Never reuse a USD prim path during the worker lifetime. Hydra/RTX keeps
    # renderer-side mesh caches, and removing then recreating (for example)
    # WebAsset_1 with a different geometry can corrupt unrelated visuals.
    web_asset_serial = [0]

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
        web_asset_serial[0] += 1
        prim_path = f"/World/envs/env_0/WebAsset_{web_asset_serial[0]}"
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

    demo_state = {
        "active": False, "recording": False, "name": "Demo", "user": "Guest",
        "movement": np.zeros(3, dtype=np.float32), "gripper": 1.0,
        "observations": [], "actions": [], "ee_positions": [], "object_positions": [],
        "playback": None, "playback_index": 0, "playback_last_action": None,
        "initial_state": None,
        "normal_episode_length_s": float(env.unwrapped.cfg.episode_length_s),
    }
    robot = env.unwrapped.scene["robot"]
    hand_body_index = robot.body_names.index("panda_hand")
    jacobian_body_index = hand_body_index - 1 if robot.is_fixed_base else hand_body_index
    teleop_ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1,
        device=env.unwrapped.device,
    )
    teleop_target_pose_b = torch.zeros((1, 7), dtype=torch.float32, device=env.unwrapped.device)
    normal_arm_stiffness = robot.data.joint_stiffness[:, :7].clone()
    normal_arm_damping = robot.data.joint_damping[:, :7].clone()
    robot_prim_path = robot.root_physx_view.prim_paths[0]

    def set_teleop_drive_gains(enabled: bool) -> None:
        """Use Isaac's recommended stiff Franka gains only for task-space teleoperation."""
        if enabled:
            robot.write_joint_stiffness_to_sim(400.0, joint_ids=list(range(7)))
            robot.write_joint_damping_to_sim(80.0, joint_ids=list(range(7)))
        else:
            robot.write_joint_stiffness_to_sim(normal_arm_stiffness, joint_ids=list(range(7)))
            robot.write_joint_damping_to_sim(normal_arm_damping, joint_ids=list(range(7)))
        sim_utils.modify_rigid_body_properties(
            robot_prim_path,
            sim_utils.RigidBodyPropertiesCfg(disable_gravity=enabled),
        )

    def reset_teleop_target() -> None:
        root_pose_w = robot.data.root_pose_w[:1]
        hand_pose_w = robot.data.body_pose_w[:1, hand_body_index]
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], hand_pose_w[:, :3], hand_pose_w[:, 3:7]
        )
        teleop_target_pose_b[:, :3] = ee_pos_b
        teleop_target_pose_b[:, 3:7] = ee_quat_b
        teleop_ik.reset()
        teleop_ik.set_command(teleop_target_pose_b)

    reset_teleop_target()

    def capture_demo_initial_state() -> dict[str, np.ndarray]:
        """Capture enough simulator state to replay a demo from its exact start pose."""
        object_asset = env.unwrapped.scene["object"]
        initial_state = {
            "initial_robot_joint_pos": robot.data.joint_pos[0].detach().cpu().numpy().copy(),
            "initial_robot_joint_vel": robot.data.joint_vel[0].detach().cpu().numpy().copy(),
            "initial_robot_root_pose": robot.data.root_pose_w[0].detach().cpu().numpy().copy(),
            "initial_robot_root_velocity": robot.data.root_vel_w[0].detach().cpu().numpy().copy(),
            "initial_object_root_pose": object_asset.data.root_pose_w[0].detach().cpu().numpy().copy(),
            "initial_object_root_velocity": object_asset.data.root_vel_w[0].detach().cpu().numpy().copy(),
            "initial_end_effector_pose": robot.data.body_pose_w[0, hand_body_index].detach().cpu().numpy().copy(),
        }
        if "object_pose" in env.unwrapped.command_manager.active_terms:
            command_term = env.unwrapped.command_manager.get_term("object_pose")
            initial_state["initial_task_target_pose"] = (
                command_term.command[0].detach().cpu().numpy().copy()
            )
        return initial_state

    def freeze_demo_task_target() -> None:
        """Prevent Isaac Lab from choosing a new task target during recording/playback."""
        if "object_pose" in env.unwrapped.command_manager.active_terms:
            command_term = env.unwrapped.command_manager.get_term("object_pose")
            command_term.time_left.fill_(24.0 * 60.0 * 60.0)

    def restore_demo_initial_state(initial_state: dict[str, np.ndarray]) -> None:
        """Restore a modern demonstration's robot, object, and task-target state."""
        device = env.unwrapped.device
        object_asset = env.unwrapped.scene["object"]

        def tensor(name: str) -> torch.Tensor:
            return torch.as_tensor(initial_state[name], dtype=torch.float32, device=device).reshape(1, -1)

        robot.write_root_pose_to_sim(tensor("initial_robot_root_pose"))
        robot.write_root_velocity_to_sim(tensor("initial_robot_root_velocity"))
        robot.write_joint_state_to_sim(
            tensor("initial_robot_joint_pos"), tensor("initial_robot_joint_vel")
        )
        object_asset.write_root_pose_to_sim(tensor("initial_object_root_pose"))
        object_asset.write_root_velocity_to_sim(tensor("initial_object_root_velocity"))
        if "initial_task_target_pose" in initial_state:
            command_term = env.unwrapped.command_manager.get_term("object_pose")
            command_term.command.copy_(tensor("initial_task_target_pose"))
            freeze_demo_task_target()
            env.unwrapped.command_manager.compute(dt=0.0)
        env.unwrapped.episode_length_buf.zero_()
        env.unwrapped.scene.update(dt=0.0)

    def save_demo() -> None:
        if not demo_state["recording"] or not demo_state["actions"]:
            return
        user = str(demo_state["user"])
        user_directory = demo_directory / user
        user_directory.mkdir(parents=True, exist_ok=True)
        slug = "-".join(str(demo_state["name"]).strip().split()) or "Demo"
        slug = "".join(char for char in slug if char.isalnum() or char in "-_.")[:64]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        stem = f"{stamp}-{slug}"
        initial_state = demo_state.get("initial_state") or capture_demo_initial_state()
        np.savez_compressed(
            user_directory / f"{stem}.npz",
            observations=np.asarray(demo_state["observations"], dtype=np.float32),
            actions=np.asarray(demo_state["actions"], dtype=np.float32),
            ee_positions=np.asarray(demo_state["ee_positions"], dtype=np.float32),
            object_positions=np.asarray(demo_state["object_positions"], dtype=np.float32),
            **initial_state,
        )
        atomic_json(user_directory / f"{stem}.json", {
            "name": demo_state["name"], "task": task_mode, "steps": len(demo_state["actions"]),
            "duration": round(len(demo_state["actions"]) * float(env.unwrapped.step_dt), 3),
            "created_at": time.time(), "format": "isaaclab-observation-action-v3",
        })
        print(f"Saved web demonstration {user}/{stem} ({len(demo_state['actions'])} steps)", flush=True)

    def teleop_action() -> torch.Tensor:
        movement = torch.as_tensor(demo_state["movement"], dtype=torch.float32, device=env.unwrapped.device)
        if torch.linalg.vector_norm(movement) > 0:
            # About 9 cm/s at the 20 Hz web loop: responsive while remaining
            # slow enough for precise demonstrations and stable IK tracking.
            teleop_target_pose_b[:, :3] += movement.unsqueeze(0) * 0.0045
            teleop_target_pose_b[:, 0].clamp_(0.20, 0.85)
            teleop_target_pose_b[:, 1].clamp_(-0.55, 0.55)
            teleop_target_pose_b[:, 2].clamp_(0.03, 0.90)
            teleop_ik.set_command(teleop_target_pose_b)

        root_pose_w = robot.data.root_pose_w[:1]
        hand_pose_w = robot.data.body_pose_w[:1, hand_body_index]
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], hand_pose_w[:, :3], hand_pose_w[:, 3:7]
        )
        jacobian = robot.root_physx_view.get_jacobians()[:1, jacobian_body_index, :, :7].clone()
        base_rotation = math_utils.matrix_from_quat(math_utils.quat_inv(root_pose_w[:, 3:7]))
        jacobian[:, :3, :] = torch.bmm(base_rotation, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rotation, jacobian[:, 3:, :])
        desired_joint_pos = teleop_ik.compute(
            ee_pos_b, ee_quat_b, jacobian, robot.data.joint_pos[:1, :7]
        )
        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        # JointPositionActionCfg applies: target = default + action * 0.5.
        # Do not clamp the inverse mapping to policy-style [-1, 1]: a valid
        # current/IK joint pose can be farther than 0.5 rad from the default,
        # and clipping it couples otherwise independent Cartesian axes.
        action[0, :7] = (desired_joint_pos[0] - robot.data.default_joint_pos[0, :7]) / 0.5
        action[0, 7] = float(demo_state["gripper"])
        return action

    def apply_camera_commands() -> None:
        global inference_policy, policy_observations, checkpoint_id, checkpoint_path, simulation_paused, step, episode
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
                elif command_type == "set_paused":
                    simulation_paused = bool(command["paused"])
                    print(
                        "Isaac web simulation paused" if simulation_paused else "Isaac web simulation resumed",
                        flush=True,
                    )
                elif command_type == "reset_episode":
                    if demo_state["recording"]:
                        print("Ignoring episode reset while a demonstration is recording", flush=True)
                    else:
                        demo_state["playback"] = None
                        demo_state["playback_last_action"] = None
                        set_teleop_drive_gains(False)
                        reset = env.reset()
                        policy_observations = reset[0].get("policy") if isinstance(reset, tuple) and isinstance(reset[0], dict) else (reset[0] if isinstance(reset, tuple) else reset)
                        step = 0
                        episode += 1
                        print("Isaac web episode reset manually", flush=True)
                elif command_type == "policy_load":
                    set_teleop_drive_gains(False)
                    requested_path = Path(str(command["path"])).resolve()
                    if not requested_path.is_file():
                        raise FileNotFoundError(f"Policy checkpoint not found: {requested_path}")
                    inference_policy = load_actor(requested_path)
                    checkpoint_path = requested_path
                    checkpoint_id = str(command["id"])
                    atomic_json(status_path, {
                        "status": "ready",
                        "task": task_mode,
                        "mode": "trained_policy",
                        "checkpoint": checkpoint_id,
                        "hot_swapped_at": time.time(),
                    })
                    print(f"Hot-swapped Isaac web policy to {checkpoint_id}", flush=True)
                elif command_type == "policy_evaluate":
                    if inference_policy is None:
                        raise RuntimeError("Load a trained policy before starting demonstration evaluation")
                    with np.load(command["path"]) as payload:
                        state_keys = (
                            "initial_robot_joint_pos", "initial_robot_joint_vel",
                            "initial_robot_root_pose", "initial_robot_root_velocity",
                            "initial_object_root_pose", "initial_object_root_velocity",
                        )
                        if not all(key in payload.files for key in state_keys):
                            raise ValueError("This legacy demonstration has no recorded initial state")
                        initial_state = {
                            key: np.asarray(payload[key], dtype=np.float32).copy()
                            for key in state_keys
                        }
                        if "initial_task_target_pose" in payload.files:
                            initial_state["initial_task_target_pose"] = np.asarray(
                                payload["initial_task_target_pose"], dtype=np.float32
                            ).copy()
                    variation = max(0.0, min(0.15, float(command.get("position_variation", 0.0))))
                    offset = np.zeros(3, dtype=np.float32)
                    if variation > 0.0:
                        offset[:2] = np.random.uniform(-variation, variation, size=2)
                        initial_state["initial_object_root_pose"][:2] += offset[:2]
                    demo_state["playback"] = None
                    demo_state["playback_last_action"] = None
                    demo_state["active"] = False
                    demo_state["recording"] = False
                    set_teleop_drive_gains(False)
                    reset = env.reset()
                    policy_observations = reset[0].get("policy") if isinstance(reset, tuple) and isinstance(reset[0], dict) else (reset[0] if isinstance(reset, tuple) else reset)
                    restore_demo_initial_state(initial_state)
                    observations = env.unwrapped.observation_manager.compute(update_history=True)
                    policy_observations = observations.get("policy") if isinstance(observations, dict) else observations
                    policy_observations = apply_vision_estimate(policy_observations)
                    # Present the restored state before the policy takes its
                    # first action. The normal Start simulation button resumes
                    # from this exact pose.
                    simulation_paused = True
                    step = 0
                    episode += 1
                    print(
                        f"Evaluating policy from demo {command['id']} with object XY offset "
                        f"({offset[0]:+.4f}, {offset[1]:+.4f}) m",
                        flush=True,
                    )
                elif command_type == "demo_start":
                    demo_state.update({"active": True, "recording": True, "name": command["name"], "user": command["user"],
                        "movement": np.zeros(3, dtype=np.float32), "gripper": 1.0, "observations": [], "actions": [],
                        "ee_positions": [], "object_positions": [], "playback": None, "playback_index": 0,
                        "playback_last_action": None})
                    reset_teleop_target()
                    set_teleop_drive_gains(True)
                    demo_state["initial_state"] = capture_demo_initial_state()
                    freeze_demo_task_target()
                    # Human teleoperation must not race the six-second RL
                    # horizon. Keep physical failure terminations active, but
                    # move the timeout far beyond any practical demo length.
                    env.unwrapped.cfg.episode_length_s = 24.0 * 60.0 * 60.0
                    simulation_paused = False
                    print(f"Started web demonstration recording: {command['user']}/{command['name']}", flush=True)
                elif command_type == "demo_control":
                    if demo_state["active"]:
                        demo_state["movement"] = np.asarray(command["movement"], dtype=np.float32)
                        if float(command.get("gripper", 0)) != 0:
                            demo_state["gripper"] = float(command["gripper"])
                elif command_type == "demo_stop":
                    save_demo()
                    set_teleop_drive_gains(False)
                    env.unwrapped.cfg.episode_length_s = demo_state["normal_episode_length_s"]
                    demo_state["active"] = False
                    demo_state["recording"] = False
                    demo_state["movement"] = np.zeros(3, dtype=np.float32)
                    print("Stopped web demonstration recording", flush=True)
                elif command_type == "demo_play":
                    with np.load(command["path"]) as payload:
                        playback = np.asarray(payload["actions"], dtype=np.float32).copy()
                        state_keys = (
                            "initial_robot_joint_pos", "initial_robot_joint_vel",
                            "initial_robot_root_pose", "initial_robot_root_velocity",
                            "initial_object_root_pose", "initial_object_root_velocity",
                        )
                        initial_state = (
                            {key: np.asarray(payload[key], dtype=np.float32).copy() for key in state_keys}
                            if all(key in payload.files for key in state_keys) else None
                        )
                        if initial_state is not None and "initial_task_target_pose" in payload.files:
                            initial_state["initial_task_target_pose"] = np.asarray(
                                payload["initial_task_target_pose"], dtype=np.float32
                            ).copy()
                    set_teleop_drive_gains(True)
                    env.unwrapped.cfg.episode_length_s = 24.0 * 60.0 * 60.0
                    demo_state["playback"] = playback
                    demo_state["playback_index"] = 0
                    demo_state["playback_last_action"] = playback[-1:].copy() if len(playback) else None
                    demo_state["active"] = False
                    demo_state["recording"] = False
                    simulation_paused = False
                    reset = env.reset()
                    freeze_demo_task_target()
                    policy_observations = reset[0].get("policy") if isinstance(reset, tuple) and isinstance(reset[0], dict) else (reset[0] if isinstance(reset, tuple) else reset)
                    if initial_state is not None:
                        restore_demo_initial_state(initial_state)
                        observations = env.unwrapped.observation_manager.compute(update_history=True)
                        policy_observations = observations.get("policy") if isinstance(observations, dict) else observations
                        print(f"Playing web demonstration {command['id']} from its recorded initial state", flush=True)
                    else:
                        print(
                            f"Playing legacy web demonstration {command['id']} from the task reset state; "
                            "record it again for exact start-pose playback",
                            flush=True,
                        )
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                print(f"Ignoring invalid Isaac web command: {exc}", flush=True)
            except Exception as exc:
                print(f"Isaac web command failed: {type(exc).__name__}: {exc}", flush=True)
                if command.get("type") == "policy_load":
                    atomic_json(status_path, {
                        "status": "error",
                        "mode": "trained_policy",
                        "checkpoint": checkpoint_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
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
            "task": task_mode,
            "mode": "trained_policy" if inference_policy else "scripted_preview",
            "checkpoint": checkpoint_id,
            "startup_seconds": round(time.monotonic() - started_at, 3),
        },
    )

    step = 0
    episode = 0
    next_frame_at = time.monotonic()
    training_check_at = [0.0]
    training_is_active = [False]

    def refresh_training_activity(now: float) -> bool:
        if now < training_check_at[0]:
            return training_is_active[0]
        training_check_at[0] = now + 2.0
        training_is_active[0] = False
        try:
            for run_status_path in training_runs_directory.glob("*/status.json"):
                run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
                if run_status.get("status") in {"starting", "training"}:
                    training_is_active[0] = True
                    break
        except (OSError, ValueError, TypeError):
            pass
        return training_is_active[0]

    rewards = torch.zeros(1, dtype=torch.float32, device=env.unwrapped.device)
    terminated = torch.zeros(1, dtype=torch.bool, device=env.unwrapped.device)
    truncated = torch.zeros(1, dtype=torch.bool, device=env.unwrapped.device)
    with torch.inference_mode():
        while simulation_app.is_running() and not stopping:
            apply_camera_commands()
            if not simulation_paused:
                if demo_state["playback"] is not None:
                    index = int(demo_state["playback_index"])
                    if index < len(demo_state["playback"]):
                        actions = torch.as_tensor(demo_state["playback"][index:index + 1], device=env.unwrapped.device)
                        demo_state["playback_index"] = index + 1
                    else:
                        demo_state["playback"] = None
                        set_teleop_drive_gains(False)
                        env.unwrapped.cfg.episode_length_s = demo_state["normal_episode_length_s"]
                        last_action = demo_state.get("playback_last_action")
                        actions = (
                            torch.as_tensor(last_action, device=env.unwrapped.device)
                            if last_action is not None
                            else torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                        )
                        demo_state["playback_last_action"] = None
                    step_observations, rewards, terminated, truncated, _ = env.step(actions)
                    policy_observations = step_observations.get("policy") if isinstance(step_observations, dict) else step_observations
                elif demo_state["active"]:
                    actions = teleop_action()
                    if demo_state["recording"] and policy_observations is not None:
                        demo_state["observations"].append(policy_observations[0].detach().cpu().numpy())
                        demo_state["actions"].append(actions[0].detach().cpu().numpy())
                        demo_state["ee_positions"].append(robot.data.body_pos_w[0, hand_body_index].detach().cpu().numpy())
                        demo_state["object_positions"].append(env.unwrapped.scene["object"].data.root_pos_w[0].detach().cpu().numpy())
                    step_observations, rewards, terminated, truncated, _ = env.step(actions)
                    policy_observations = step_observations.get("policy") if isinstance(step_observations, dict) else step_observations
                elif inference_policy is not None and policy_observations is not None:
                    actions = inference_policy(policy_observations.cpu()).to(env.unwrapped.device)
                    step_observations, rewards, terminated, truncated, _ = env.step(actions)
                    policy_observations = step_observations.get("policy") if isinstance(step_observations, dict) else step_observations
                    policy_observations = apply_vision_estimate(policy_observations)
                else:
                    phase = step * 0.018
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    actions[:, 0] = 0.22 * math.sin(phase)
                    actions[:, 1] = 0.16 * math.sin(phase * 0.73 + 0.8)
                    actions[:, 3] = 0.18 * math.sin(phase * 0.51)
                    actions[:, 5] = 0.12 * math.cos(phase * 0.67)
                    actions[:, 7] = 1.0 if math.sin(phase * 0.35) > 0 else -1.0
                    step_observations, rewards, terminated, truncated, _ = env.step(actions)
                    policy_observations = step_observations.get("policy") if isinstance(step_observations, dict) else step_observations
                    policy_observations = apply_vision_estimate(policy_observations)

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
            if task_mode == "vision" and step % 3 == 0:
                vision_frame = (
                    env.unwrapped.scene["vision_camera"]
                    .data.output["rgb"][0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                vision_image_path = vision_frame_path.with_suffix(".jpg.next")
                Image.fromarray(np.asarray(vision_frame, dtype=np.uint8)).save(
                    vision_image_path,
                    format="JPEG",
                    quality=max(50, min(92, args.jpeg_quality)),
                )
                os.replace(vision_image_path, vision_frame_path)

            simulation_time = step * float(env.unwrapped.step_dt)
            atomic_json(
                metadata_path,
                {
                    "engine": "isaaclab",
                    "task": task_mode,
                    "episode": episode,
                    "step": step,
                    "reward": float(rewards[0].item()),
                    "simulation_time": simulation_time,
                    "paused": simulation_paused,
                    "mode": "demo_recording" if demo_state["recording"] else ("demo_playback" if demo_state["playback"] is not None else ("trained_policy" if inference_policy else "scripted_preview")),
                    "demo_recording": demo_state["recording"],
                    "demo_steps": len(demo_state["actions"]),
                    "demo_playback_step": int(demo_state["playback_index"]) if demo_state["playback"] is not None else None,
                    "demo_playback_steps": len(demo_state["playback"]) if demo_state["playback"] is not None else None,
                    "checkpoint": checkpoint_id,
                    "vision_estimated_position": policy_observations[0, 18:21].tolist() if task_mode == "vision" and policy_observations is not None else None,
                    "object_position": env.unwrapped.scene["object"].data.root_pos_w[0].tolist(),
                    "object_velocity": env.unwrapped.scene["object"].data.root_vel_w[0].tolist(),
                    "end_effector_position": robot.data.body_pos_w[0, hand_body_index].tolist(),
                    "demo_target_position": teleop_target_pose_b[0, :3].tolist() if demo_state["active"] else None,
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
            # RTX rendering used to run without a frame limit and could
            # starve a simultaneous multi-environment trainer on the same GPU.
            # Keep playback usable, but explicitly yield most GPU time while a
            # training run is active. These are separate processes and no
            # trainer state, environment, or metric file is touched here.
            now = time.monotonic()
            frame_rate = args.training_max_fps if refresh_training_activity(now) else args.max_fps
            frame_period = 1.0 / max(1.0, frame_rate)
            next_frame_at = max(next_frame_at + frame_period, now)
            remaining = next_frame_at - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            if simulation_paused:
                continue
            step += 1
            if bool(terminated[0].item() or truncated[0].item()):
                if inference_policy is None:
                    env.reset()
                step = 0
                episode += 1
finally:
    atomic_json(status_path, {"status": "stopped", "task": args.task})
    if env is not None:
        env.close()
    simulation_app.close()
