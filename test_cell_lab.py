"""GPU integration: generated laboratory, fixed furniture, real contact cube lift."""
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
    import torch
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api="blas")
    import gymnasium as gym
    import isaaclab_tasks
    import isaac_cell_lab_task
    from isaac_cell_lab_controller import CellLabCubeController
    from isaaclab_tasks.utils import parse_env_cfg
    cfg = parse_env_cfg("Isaac-Franka-Cell-Lab-Play-v0", device=args.device, num_envs=1)
    cfg.seed = 42
    cfg.sim.render_interval = 1000
    if args.diagnostic_image:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg
        cfg.scene.diagnostic_camera = CameraCfg(prim_path="{ENV_REGEX_NS}/DiagnosticCamera",
            update_period=0., width=960, height=720, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=20.955,
                clipping_range=(.01, 100.)))
    env = gym.make("Isaac-Franka-Cell-Lab-Play-v0", cfg=cfg)
    env.reset()
    if args.diagnostic_image:
        env.unwrapped.scene["diagnostic_camera"].set_world_poses_from_view(
            torch.tensor([[1.1, -1.5, .9]], device=env.unwrapped.device),
            torch.tensor([[.25, .1, .18]], device=env.unwrapped.device))
    controller = CellLabCubeController(env)
    furniture = {name: asset.data.root_pos_w.clone() for name, asset in env.unwrapped.scene.articulations.items() if name != "robot"}
    assert len(furniture) == 10, f"Expected ten articulated furniture items, got {len(furniture)}"
    previous = ""
    with torch.inference_mode():
        for step in range(3000):
            _, rewards, term, trunc, _ = env.step(controller(None))
            assert torch.isfinite(rewards).all() and not term.any() and not trunc.any()
            if previous != controller.stage:
                print("CELL_LAB_STAGE", step, controller.stage, env.unwrapped.scene["object"].data.root_pos_w.tolist(), flush=True)
                previous = controller.stage
            if step % 100 == 0:
                print("CELL_LAB_PROGRESS", step, controller.stage,
                    env.unwrapped.scene["robot"].data.body_pose_w[0, controller.hand].tolist(),
                    "target", controller.target.tolist() if controller.target is not None else None, flush=True)
            if args.diagnostic_image and (step == 300 or controller.stage == "failed" or (controller.stage == "done" and controller.ticks == 1)):
                from PIL import Image
                env.unwrapped.sim.render()
                env.unwrapped.scene["diagnostic_camera"].update(env.unwrapped.step_dt, force_recompute=True)
                Image.fromarray(env.unwrapped.scene["diagnostic_camera"].data.output["rgb"][0].cpu().numpy()).save(args.diagnostic_image)
            assert controller.stage != "failed", "Physical cube pickup failed"
            if controller.stage == "done" and controller.ticks >= 150:
                break
        assert controller.stage == "done", "Lift did not complete"
        assert float(env.unwrapped.scene["object"].data.root_pos_w[0, 2]) > .13
        for name, initial in furniture.items():
            assert torch.linalg.vector_norm(env.unwrapped.scene[name].data.root_pos_w - initial).item() < .002, name
        result = {"controller": "cell_lab_cube_ik", "physics_verified": True,
            "task": "cell_lab", "articulated_furniture": len(furniture),
            "cube_height_m": float(env.unwrapped.scene["object"].data.root_pos_w[0, 2]),
            "steps": step + 1, "limitation": "Simulator-state Lula IK reference, not neural; scenery props static."}
        print("CELL_LAB_PASS", json.dumps(result), flush=True)
        if args.validation_output:
            args.validation_output.write_text(json.dumps(result, indent=2))
    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    launcher.app.close()
