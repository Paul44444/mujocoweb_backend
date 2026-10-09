"""Isolated GPU smoke test. Only publishes grasps after an actual held lift.

Use the Isaac interpreter, after graspgenx_prepare.py generated predictions.
No grasp attachment, pose teleport, or training/checkpoint writes are used.
"""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--grasps", type=Path, required=True)
parser.add_argument("--validation-output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
import gymnasium as gym
import torch
import isaaclab_tasks
import isaac_graspgen_task
from isaaclab_tasks.utils import parse_env_cfg
from isaac_graspgen_controller import GraspGenTubeController

cfg = parse_env_cfg("Isaac-Franka-GraspGenX-Tube-Pick-Play-v0", device=args.device, num_envs=1)
cfg.seed = 42
cfg.episode_length_s = 120.
for actuator in ("panda_shoulder", "panda_forearm"):
    cfg.scene.robot.actuators[actuator].stiffness = 400.
    cfg.scene.robot.actuators[actuator].damping = 80.
cfg.scene.robot.spawn.rigid_props.disable_gravity = True
env = gym.make("Isaac-Franka-GraspGenX-Tube-Pick-Play-v0", cfg=cfg)
try:
    obs, _ = env.reset()
    controller = GraspGenTubeController(env, args.grasps)
    initial = float(env.unwrapped.scene["object"].data.root_pos_w[0, 2])
    peak = initial
    held_steps = 0
    with torch.no_grad():
        for step in range(1600):
            obs, _, terminated, truncated, _ = env.step(controller(obs["policy"]))
            height = float(env.unwrapped.scene["object"].data.root_pos_w[0, 2])
            peak = max(peak, height)
            held_steps = held_steps + 1 if controller.stage == "hold" and height > initial + .15 else 0
            if step % 100 == 0:
                print("GRASP_TEST", step, controller.stage, height, flush=True)
            if held_steps >= 500 or bool(terminated[0]) or bool(truncated[0]) or controller.stage == "failed":
                break
    result = {"steps": step + 1, "stage": controller.stage, "initial_height_m": initial,
              "peak_height_m": peak, "final_height_m": height, "held_seconds": held_steps * cfg.sim.dt * cfg.decimation,
              "passed": held_steps >= 500, "seed": cfg.seed}
    print("GRASP_RESULT", json.dumps(result), flush=True)
    if not result["passed"]:
        raise RuntimeError("Pickup did not hold a real-contact lift for 10 simulated seconds")
    if args.validation_output:
        payload = json.loads(args.grasps.read_text())
        payload.update(physics_verified=True, pickup_validation=result,
                       environment="Isaac-Franka-GraspGenX-Tube-Pick-Play-v0",
                       limitation="Precomputed grasp candidates for this cylinder; IK and staged arm motion, not an end-to-end neural policy.")
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)
        args.validation_output.write_text(json.dumps(payload, indent=2))
finally:
    env.close()
    launcher.app.close()
