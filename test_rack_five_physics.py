"""GPU regression: five separate dynamic tubes, seated stability and reset."""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
env = None
try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks
    import isaac_rack_five_task
    from isaaclab_tasks.utils import parse_env_cfg
    from isaac_rack_five_layout import HOLES_LOCAL, OCCUPIED_SLOTS, TUBE_NAMES
    from isaac_rack_insert_task import RACK_POSITION, SEATED_CENTER_Z
    from isaaclab.utils.math import quat_apply
    from pxr import UsdPhysics
    cfg = parse_env_cfg("Isaac-Franka-Five-Tube-Rack-Play-v0", device="cuda:0", num_envs=1)
    env = gym.make("Isaac-Franka-Five-Tube-Rack-Play-v0", cfg=cfg)
    base = env.unwrapped
    action = torch.zeros((1, 8), device=base.device)
    action[:, 7] = 1.
    def check():
        for index, (name, slot) in enumerate(zip(TUBE_NAMES, OCCUPIED_SLOTS)):
            tube = base.scene[name]
            p = (tube.data.root_pos_w - base.scene.env_origins)[0]
            xy = p.new_tensor(HOLES_LOCAL[slot]) + p.new_tensor(RACK_POSITION[:2])
            assert torch.linalg.vector_norm(p[:2] - xy) < .0018, (name, p)
            assert abs(float(p[2]) - SEATED_CENTER_Z) < .003, (name, p)
            up = quat_apply(tube.data.root_quat_w, p.new_tensor([[0., 0., 1.]]))
            assert float(up[0, 2]) > .99, (name, up)
            assert torch.linalg.vector_norm(tube.data.root_lin_vel_w) < .03
            # RigidObject is genuinely simulated, not welded/kinematic decor.
            path = "Object" if index == 0 else f"RackTube{index + 1}"
            prim = base.sim.stage.GetPrimAtPath("/World/envs/env_0/" + path)
            assert prim.HasAPI(UsdPhysics.RigidBodyAPI), name
            assert not UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get(), name
            print("SEATED", name, slot + 1, p.cpu().tolist(), flush=True)
    env.reset()
    for _ in range(300):
        _, reward, _, _, _ = env.step(action)
        assert torch.isfinite(reward).all()
    check()
    # Move every tube out of its hole, then prove reset restores ALL objects.
    for name in TUBE_NAMES:
        obj = base.scene[name]
        pose = obj.data.root_pose_w.clone()
        pose[:, 0] += .15
        obj.write_root_pose_to_sim(pose)
    env.reset()
    for _ in range(150):
        env.step(action)
    check()
    print("FIVE_TUBE_PHYSICS_PASS", flush=True)
finally:
    if env is not None:
        env.close()
    app.close(wait_for_replicator=False)
