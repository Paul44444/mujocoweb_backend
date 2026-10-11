"""Validate real contact grasp and loading, release, closed door and stability."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--validation-output", type=Path)
parser.add_argument("--diagnostic-image", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
try:
    import gymnasium as gym
    import torch
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api="blas")
    import isaaclab_tasks
    import isaac_incubator_task
    from isaaclab_tasks.utils import parse_env_cfg
    from isaac_incubator_controller import IncubatorLoadingController
    cfg = parse_env_cfg("Isaac-Franka-Tube-Incubator-Play-v0", device=args.device, num_envs=1)
    cfg.seed = 42
    cfg.sim.render_interval = 1000  # Physics regression needs no camera frames.
    if args.diagnostic_image:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg
        cfg.scene.diagnostic_camera = CameraCfg(prim_path="{ENV_REGEX_NS}/DiagnosticCamera",
            update_period=0., width=960, height=720, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=20.955,
                clipping_range=(.01, 100.)))
    env = gym.make("Isaac-Franka-Tube-Incubator-Play-v0", cfg=cfg)
    env.reset()
    if args.diagnostic_image:
        env.unwrapped.scene["diagnostic_camera"].set_world_poses_from_view(
            torch.tensor([[1.0, -.70, .65]], device=env.unwrapped.device),
            torch.tensor([[.48, .30, .24]], device=env.unwrapped.device))
    controller = IncubatorLoadingController(env)
    previous = ""
    peak = 0.
    with torch.inference_mode():
        for step in range(8000):
            _, reward, term, trunc, _ = env.step(controller(None))
            obj = env.unwrapped.scene["object"]
            peak = max(peak, float(obj.data.root_pos_w[0, 2]))
            assert torch.isfinite(reward).all() and not term.any() and not trunc.any()
            if previous != controller.stage:
                previous = controller.stage
                print("INCUBATOR_STAGE", step, previous, obj.data.root_pose_w[0].cpu().tolist(), flush=True)
            if args.diagnostic_image and (step == 1800 or controller.stage == "failed"
                    or (controller.stage == "done" and controller.ticks == 1)):
                from PIL import Image
                env.unwrapped.sim.render()
                env.unwrapped.scene["diagnostic_camera"].update(.02, force_recompute=True)
                Image.fromarray(env.unwrapped.scene["diagnostic_camera"].data.output["rgb"][0].cpu().numpy()).save(args.diagnostic_image)
            assert controller.stage != "failed", "Loading controller failed"
            if step % 300 == 0:
                print("LOADING_DIAGNOSTIC", step, controller.stage,
                    env.unwrapped.scene["robot"].data.body_pose_w[0, controller.hand].cpu().tolist(),
                    obj.data.root_pose_w[0].cpu().tolist(), flush=True)
            if controller.stage == "done" and controller.ticks > 300:
                assert controller.deposited(), "Tube not stably deposited inside"
                assert peak > .20, "No physical lift"
                assert abs(float(env.unwrapped.scene["incubator"].data.joint_pos[0, 0])) < .025
                result = {"controller": "incubator_loading_ik", "physics_verified": True,
                    "final_pose": obj.data.root_pose_w[0].cpu().tolist(), "peak_height_m": peak,
                    "stable_seconds": 6., "steps": step + 1,
                    "limitation": "Analytic simulator-pose contact grasp + Lula IK; automatic door. Not neural or camera-based."}
                if args.validation_output:
                    args.validation_output.write_text(json.dumps(result, indent=2))
                print("INCUBATOR_LOADING_PASS", json.dumps(result), flush=True)
                break
        else:
            raise RuntimeError("Loading timed out")
    env.close()
except Exception:
    import traceback
    traceback.print_exc()
finally:
    launcher.app.close()
