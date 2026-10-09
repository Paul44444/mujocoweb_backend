"""GPU regression: standing tube, open rack hole, solid rim, finite rewards."""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg
    import isaac_rack_insert_task as task
    from pxr import UsdPhysics

    cfg = parse_env_cfg("Isaac-Franka-Upright-Tube-Rack-Insert-Play-v0", device="cuda:0", num_envs=1)
    cfg.seed = 42
    cfg.episode_length_s = 60
    env = gym.make("Isaac-Franka-Upright-Tube-Rack-Insert-Play-v0", cfg=cfg)
    base = env.unwrapped
    env.reset()
    meshes = [p for p in base.sim.stage.Traverse() if p.HasAPI(UsdPhysics.MeshCollisionAPI) and "/RackVisual/" in str(p.GetPath())]
    assert meshes and all(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == "none" for p in meshes)
    assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in meshes), "Rack must stay static"
    actions = torch.zeros((1, 8), device=base.device)
    actions[:, 7] = 1
    obj = base.scene["object"]
    def settle(position=None):
        env.reset()
        if position:
            pose = obj.data.default_root_state[:, :7].clone()
            pose[:, :3] = pose.new_tensor(position) + base.scene.env_origins
            pose[:, 3:] = pose.new_tensor((1., 0., 0., 0.))
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=base.device))
            base.sim.forward()
            base.scene.update(0.)
        for _ in range(150):
            _, reward, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(reward).all() and not terminated.any() and not truncated.any()
        p = (obj.data.root_pos_w - base.scene.env_origins)[0].cpu().tolist()
        print("SETTLED", position, p, flush=True)
        return p
    standing = settle()
    assert abs(standing[2] - 0.046) < 0.004, standing
    assert obj.data.root_quat_w[0, 0].abs() > 0.99, "Initial tube tipped over"
    inserted = settle((*task.HOLE_CENTER, 0.18))
    assert 0.07 < inserted[2] < 0.09, "Tube did not pass the upper hole and rest on the lower shelf"
    assert task.insertion_reward(base, "release").item() > 0.9, "Placed tube not recognized"
    # The solid bridge between adjacent holes must support/block the tube.
    blocked = settle((task.RACK_POSITION[0], task.HOLE_CENTER[1], 0.18))
    assert blocked[2] > 0.09, "Rack bridge does not collide"
    print("RACK_INSERT_PHYSICS_PASS", flush=True)
    env.close()
finally:
    app.close()
