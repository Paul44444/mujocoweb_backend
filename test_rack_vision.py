"""GPU vision regression. Truth is used ONLY in test diagnostics, not control."""
import argparse
import json
from pathlib import Path
import faulthandler
faulthandler.dump_traceback_later(90, repeat=True)
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--execute", action="store_true")
parser.add_argument("--validation-output", type=Path)
parser.add_argument("--rack-x", type=float, default=0.)
parser.add_argument("--rack-y", type=float, default=0.)
parser.add_argument("--rack-yaw", type=float, default=0., help="Degrees")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
print("VISION_TEST_APP_READY", flush=True)
env = None
try:
    import gymnasium as gym
    import torch
    import time
    torch.set_num_threads(1)
    import isaaclab_tasks
    import isaac_rack_vision_task as task
    from isaaclab_tasks.utils import parse_env_cfg
    from isaac_rack_perception import MarkerPerception, sphere_centers
    from isaac_rack_vision_controller import VisionRackController
    from isaac_rack_insert_task import HOLE_CENTER, SEATED_CENTER_Z, insertion_reward
    from PIL import Image
    import numpy as np
    import math

    cfg = parse_env_cfg("Isaac-Franka-Rack-Insert-Vision-Play-v0", device=args.device, num_envs=1)
    cfg.seed = 42
    cfg.episode_length_s = 120
    yaw = math.radians(args.rack_yaw)
    cfg.scene.rack_visual.init_state.pos = (task.RACK_POSITION[0] + args.rack_x, task.RACK_POSITION[1] + args.rack_y, 0.)
    cfg.scene.rack_visual.init_state.rot = (math.cos(yaw / 2.), 0., 0., math.sin(yaw / 2.))
    local = np.array([.0136571, -.0136573])
    rotation_xy = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    expected_xy = rotation_xy @ local + np.array(cfg.scene.rack_visual.init_state.pos[:2])
    for name in ("panda_shoulder", "panda_forearm"):
        cfg.scene.robot.actuators[name].stiffness = 400.
        cfg.scene.robot.actuators[name].damping = 80.
    cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    env = gym.make("Isaac-Franka-Rack-Insert-Vision-Play-v0", cfg=cfg)
    base = env.unwrapped
    env.reset()
    task.aim_cameras(env)
    actions = torch.zeros((1, 8), device=base.device)
    actions[:, 7] = 1.
    for _ in range(40):
        env.step(actions)
    for name in ("rack_camera_a", "rack_camera_b"):
        camera = base.scene[name]
        Image.fromarray(camera.data.output["rgb"][0, ..., :3].cpu().numpy()).save(f"/tmp/{name}.png")
        np.savez(f"/tmp/{name}.npz", rgb=camera.data.output["rgb"][0, ..., :3].cpu().numpy(),
                 depth=camera.data.output["distance_to_image_plane"][0].cpu().numpy(),
                 k=camera.data.intrinsic_matrices[0].cpu().numpy(),
                 position=camera.data.pos_w[0].cpu().numpy(), quaternion=camera.data.quat_w_ros[0].cpu().numpy())
        print("DETECTIONS", name, {color: [p.tolist() for p in sphere_centers(camera, color, .007 if color == 'rack' else .012, debug=True)] for color in ('upper','lower','rack')}, flush=True)
    perception = MarkerPerception(env)
    pose, target = perception.estimate()
    error = torch.linalg.vector_norm(pose[:, :3] - base.scene["object"].data.root_pos_w[:1]).item()
    goal_error = torch.linalg.vector_norm(target[0] - target.new_tensor([*expected_xy, SEATED_CENTER_Z])).item()
    print("VISION_ERROR", error, goal_error, pose.cpu().tolist(), target.cpu().tolist(), flush=True)
    assert error < .0015 and goal_error < .001, "Pose estimates too inaccurate for insertion"
    if args.execute:
        controller = VisionRackController(env)
        previous = ""
        lost = 0
        timings = [0., 0.]
        class TruthForbidden:
            def __getattr__(self, name):
                raise AssertionError("Vision controller attempted to read object truth: " + name)
        with torch.inference_mode():
            for step in range(5000):
                original = base.scene._rigid_objects["object"]
                base.scene._rigid_objects["object"] = TruthForbidden()
                try:
                    start = time.monotonic()
                    action = controller(None)
                    timings[0] += time.monotonic() - start
                finally:
                    base.scene._rigid_objects["object"] = original
                start = time.monotonic()
                env.step(action)
                timings[1] += time.monotonic() - start
                lost = 0 if controller.perception.last_error in {"RGB-D markers detected", "Tube detected; static rack uses prior visual calibration"} else lost + 1
                assert lost < 100, "Persistent marker occlusion: " + controller.perception.last_error
                if step % 100 == 0 or previous != controller.stage:
                    previous = controller.stage
                    print("VISION_STAGE", step, previous, controller.perception.last_error, flush=True)
                    print("TIMINGS", timings, flush=True)
                assert controller.stage != "failed"
                if controller.stage == "done" and controller.ticks >= 300:
                    object_data = base.scene["object"].data
                    assert torch.linalg.vector_norm(object_data.root_pos_w[0, :2] - target.new_tensor(expected_xy)).item() < .0018
                    assert abs(float(object_data.root_pos_w[0, 2]) - SEATED_CENTER_Z) < .008
                    assert torch.linalg.vector_norm(object_data.root_lin_vel_w[0]).item() < .03
                    print("VISION_INSERTION_PASS", flush=True)
                    if args.validation_output:
                        result = {
                            "initial_tube_error_m": error, "initial_goal_error_m": goal_error,
                            "steps": step + 1, "stable_seconds": 6., "truth_access_guard": True,
                            "rack_variation": {"x": args.rack_x, "y": args.rack_y, "yaw_degrees": args.rack_yaw},
                        }
                        previous = json.loads(args.validation_output.read_text()) if args.validation_output.exists() else {}
                        validations = previous.get("validation", [])
                        if previous.get("physics_verified") and not validations and "initial_tube_error_m" in previous:
                            validations = [{key: previous[key] for key in ("initial_tube_error_m", "initial_goal_error_m", "steps", "stable_seconds")}]
                        args.validation_output.write_text(json.dumps({
                            "controller": "rack_insertion_rgbd", "physics_verified": True,
                            "environment": "Isaac-Franka-Rack-Insert-Vision-Play-v0",
                            "validation": [*validations, result],
                            "limitation": "Calibrated RGB-D cameras and known colored spherical markers. Static rack calibration retained during occlusion until reset. IK reference, not neural. Uses no object/rack truth for control; simulated depth has no real sensor noise."
                        }, indent=2))
                    break
            else:
                raise RuntimeError("Vision insertion timed out")
finally:
    if env is not None:
        env.close()
    faulthandler.cancel_dump_traceback_later()
    launcher.app.close(wait_for_replicator=False)
