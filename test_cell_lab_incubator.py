"""GPU regression for the additional lab's two-door physical loading task."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--validation-output", type=Path)
parser.add_argument("--diagnostic-image", type=Path)
parser.add_argument("--contact-debug", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.contact_debug:
    import os
    os.environ["LAB_INCUBATOR_CONTACT_DEBUG"] = "1"
launcher = AppLauncher(args)
try:
    import torch
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api="blas")
    import gymnasium as gym
    import isaaclab_tasks
    import isaac_cell_lab_incubator_task
    from isaac_cell_lab_incubator_controller import CellLabIncubatorController
    from isaaclab_tasks.utils import parse_env_cfg
    from isaac_rack_five_layout import TUBE_NAMES
    cfg = parse_env_cfg("Isaac-Franka-Cell-Lab-Incubator-Play-v0", device=args.device, num_envs=1)
    cfg.seed = 42
    cfg.sim.render_interval = 1000
    if args.diagnostic_image:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg
        cfg.scene.diagnostic_camera = CameraCfg(prim_path="{ENV_REGEX_NS}/DiagnosticCamera",
            update_period=0., width=1280, height=720, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=20.955,
                clipping_range=(.01, 100.)))
    env = gym.make("Isaac-Franka-Cell-Lab-Incubator-Play-v0", cfg=cfg)
    env.reset()
    if args.contact_debug:
        from pxr import PhysicsSchemaTools
        import omni.physx
        seen_contacts = set()
        def contact_report(headers, data):
            for header in headers:
                paths = tuple(str(PhysicsSchemaTools.intToSdfPath(value)) for value in (header.collider0, header.collider1))
                if paths not in seen_contacts:
                    seen_contacts.add(paths)
                    print("LAB_INCUBATOR_CONTACT", paths, flush=True)
        contact_subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact_report)
    if args.diagnostic_image:
        env.unwrapped.scene["diagnostic_camera"].set_world_poses_from_view(
            torch.tensor([[-.6, -1.3, .85]], device=env.unwrapped.device),
            torch.tensor([[.32, .16, .21]], device=env.unwrapped.device))
    controller = CellLabIncubatorController(env)
    initial = {name: env.unwrapped.scene[name].data.root_pos_w.clone() for name in TUBE_NAMES}
    furniture = {name: asset.data.root_pos_w.clone() for name, asset in env.unwrapped.scene.articulations.items() if name != "robot"}
    peak, previous = 0., ""
    with torch.inference_mode():
        for step in range(9000):
            _, rewards, term, trunc, _ = env.step(controller(None))
            obj = env.unwrapped.scene["object"].data
            peak = max(peak, float(obj.root_pos_w[0, 2]))
            assert torch.isfinite(rewards).all() and not term.any() and not trunc.any()
            changed = previous != controller.stage
            if changed:
                previous = controller.stage
                print("LAB_INCUBATOR_STAGE", step, controller.stage, obj.root_pose_w[0].tolist(), flush=True)
            if step % 300 == 0:
                print("LAB_INCUBATOR_PROGRESS", step, controller.stage, "hand",
                    env.unwrapped.scene["robot"].data.body_pose_w[0, controller.hand].tolist(),
                    "doors", controller.door_values()[2].tolist(), flush=True)
            if args.diagnostic_image and (step in {300, 700, 2200} or controller.stage == "failed"
                    or (controller.stage == "done" and controller.ticks == 1)):
                from PIL import Image
                env.unwrapped.sim.render()
                env.unwrapped.scene["diagnostic_camera"].update(.02, force_recompute=True)
                Image.fromarray(env.unwrapped.scene["diagnostic_camera"].data.output["rgb"][0].cpu().numpy()).save(args.diagnostic_image)
            assert controller.stage != "failed", "Laboratory loading failed"
            if controller.stage == "done" and controller.ticks > 300:
                assert controller.deposited(), "Tube did not settle on the shelf"
                assert controller.doors_closed(), "Both doors must close"
                assert peak > float(initial["object"][0, 2]) + .12, "No physical pickup"
                for name in TUBE_NAMES[1:]:
                    assert torch.linalg.vector_norm(env.unwrapped.scene[name].data.root_pos_w - initial[name]).item() < .006, name
                for name, start in furniture.items():
                    assert torch.linalg.vector_norm(env.unwrapped.scene[name].data.root_pos_w - start).item() < .002, name
                # Inspection may reveal the tube without disturbing it or
                # opening the glass door, and Reset must restore the task.
                controller.inspect = True
                for _ in range(200):
                    env.step(controller(None))
                assert float(controller.door_values()[2][0]) < -1.8
                assert abs(float(controller.door_values()[2][1])) < .025
                assert controller.deposited()
                controller.inspect = False
                for _ in range(200):
                    env.step(controller(None))
                assert controller.doors_closed() and controller.deposited()
                result = {"controller": "cell_lab_incubator_ik", "physics_verified": True,
                    "task": "cell_lab_incubator", "both_doors_closed": True, "neighbours_preserved": True,
                    "final_pose": obj.root_pose_w[0].tolist(), "peak_height_m": peak, "stable_seconds": 6., "steps": step + 1,
                    "limitation": "Simulator-state contact grasp + Lula IK; automatic doors. Not neural or image-based."}
                env.reset()
                controller.reset()
                assert controller.stage == "open_door" and not controller.inspect
                assert controller.doors_closed()
                for name, position in initial.items():
                    assert torch.linalg.vector_norm(env.unwrapped.scene[name].data.root_pos_w - position).item() < .001
                print("LAB_INCUBATOR_RESET_PASS", flush=True)
                print("LAB_INCUBATOR_PASS", json.dumps(result), flush=True)
                if args.validation_output:
                    args.validation_output.write_text(json.dumps(result, indent=2))
                break
        else:
            raise RuntimeError("Laboratory loading timed out")
    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    launcher.app.close()
