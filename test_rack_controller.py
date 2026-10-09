"""Contact-only insertion regression; publish reference only after all cases pass."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--validation-output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks
    import isaac_rack_insert_task as task
    from isaaclab_tasks.utils import parse_env_cfg
    from isaac_rack_controller import RackInsertionController

    cfg = parse_env_cfg("Isaac-Franka-Upright-Tube-Rack-Insert-Play-v0", device=args.device, num_envs=1)
    cfg.seed = 42
    cfg.episode_length_s = 120
    for name in ("panda_shoulder", "panda_forearm"):
        cfg.scene.robot.actuators[name].stiffness = 400.
        cfg.scene.robot.actuators[name].damping = 80.
    cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    env = gym.make("Isaac-Franka-Upright-Tube-Rack-Insert-Play-v0", cfg=cfg)
    base = env.unwrapped
    controller = RackInsertionController(env)
    results = []
    with torch.inference_mode():
        for position in (None, (0.415, -0.15, 0.047), (0.445, -0.17, 0.047)):
            env.reset()
            obj = base.scene["object"]
            if position is not None:
                pose = obj.data.default_root_state[:, :7].clone()
                pose[:, :3] = pose.new_tensor(position) + base.scene.env_origins
                pose[:, 3:] = pose.new_tensor((1., 0., 0., 0.))
                # Initial state only: the controller never writes object poses.
                obj.write_root_pose_to_sim(pose)
                obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=base.device))
                base.sim.forward()
                base.scene.update(0.)
            initial = obj.data.root_pose_w[0].cpu().tolist()
            controller.reset()
            previous = ""
            peak = initial[2]
            for step in range(5000):
                _, reward, term, trunc, _ = env.step(controller(None))
                peak = max(peak, float(obj.data.root_pos_w[0, 2]))
                assert torch.isfinite(reward).all() and not term.any() and not trunc.any()
                if previous != controller.stage:
                    previous = controller.stage
                    print("STAGE", len(results), step, previous, obj.data.root_pose_w.cpu().tolist(), flush=True)
                assert previous != "failed", "Insertion controller failed"
                if previous == "done" and controller.ticks >= 300:
                    assert task.insertion_reward(base, "release").item() > .9, "Tube not stably seated after release"
                    assert peak > initial[2] + .10, "No real pickup occurred"
                    result = {"initial_pose": initial, "final_pose": obj.data.root_pose_w[0].cpu().tolist(),
                              "peak_height_m": peak, "steps": step + 1, "stable_seconds": 6.0}
                    results.append(result)
                    print("INSERTION_RESULT", json.dumps(result), flush=True)
                    break
            else:
                raise RuntimeError("Insertion timed out")
    if args.validation_output:
        args.validation_output.write_text(json.dumps({
            "controller": "rack_insertion_ik", "physics_verified": True,
            "environment": "Isaac-Franka-Upright-Tube-Rack-Insert-Play-v0", "validation": results,
            "limitation": "Analytic upright grasp, Lula IK, measured object-pose feedback and staged insertion. Not a pretrained neural policy or GraspGenX prediction. Uses simulator poses; no object attachment."
        }, indent=2))
    print("RACK_CONTROLLER_PASS", len(results), flush=True)
    env.close()
finally:
    launcher.app.close()
