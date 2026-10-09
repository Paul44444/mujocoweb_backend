"""Separate pretrained-grasp pickup scene; existing labware tasks stay intact."""
import gymnasium as gym
from isaaclab.assets import RigidObjectCfg
from isaaclab import sim as sim_utils
from isaaclab.utils import configclass
from isaac_labware_task import FrankaTestTubeLiftEnvCfg_PLAY, TestTubeLiftPPORunnerCfg


@configclass
class GraspGenTubePickEnvCfg(FrankaTestTubeLiftEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        # A horizontal tube on two physical supports leaves its center
        # accessible to the standard Franka fingers. No grasp attachment.
        self.scene.object.init_state.pos = (0.48, -0.16, 0.0475)
        self.scene.object.init_state.rot = (0.70710678, 0.0, 0.70710678, 0.0)
        self.events.reset_object_position.params["pose_range"] = {}
        for name, x in (("support_a", .45), ("support_b", .51)):
            setattr(self.scene, name, RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, -.16, .02)),
                spawn=sim_utils.CuboidCfg(size=(.008, .025, .04),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.15, .3, .4)))))


def vary_pickup_layout(env, position_range=0., yaw_range_degrees=0., generator=None):
    """Move tube and supports together after reset, never during a rollout."""
    import math
    import torch
    from isaaclab.utils import math as math_utils
    from isaacsim.core.prims import XFormPrim
    env = env.unwrapped
    position_range = max(0., min(.05, float(position_range)))
    yaw_range_degrees = max(0., min(30., float(yaw_range_degrees)))
    sample = torch.rand((3,), device=env.device, generator=generator) * 2 - 1
    offset = torch.zeros(3, device=env.device)
    offset[:2] = sample[:2] * position_range
    yaw = sample[2] * math.radians(yaw_range_degrees)
    rotation = torch.stack((torch.cos(yaw / 2), yaw * 0, yaw * 0, torch.sin(yaw / 2))).unsqueeze(0)
    center = env.scene["object"].data.default_root_state[:1, :3].clone()
    for name in ("object", "support_a", "support_b"):
        asset = env.scene[name]
        pose = asset.data.default_root_state[:1, :7].clone()
        pose[:, :3] = center + math_utils.quat_apply(rotation, pose[:, :3] - center) + offset + env.scene.env_origins[:1]
        pose[:, 3:7] = math_utils.quat_mul(rotation, pose[:, 3:7])
        if name.startswith("support_"):
            # PhysX tensor teleports do not reliably author the USD/Fabric
            # transforms for kinematic blocks. Mirror only these reset poses
            # into the scene graph; the dynamic tube stays physics-driven.
            views = getattr(env, "_pickup_support_visuals", {})
            if name not in views:
                views[name] = XFormPrim(asset.root_physx_view.prim_paths[0], reset_xform_properties=False)
                env._pickup_support_visuals = views
            views[name].set_world_poses(pose[:, :3], pose[:, 3:7], usd=True)
        asset.write_root_pose_to_sim(pose)
        asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    # Flush scene-graph changes to the renderer before the paused reset frame.
    env.sim.forward()
    env.scene.update(0.)
    return {"offset_m": offset.tolist(), "yaw_degrees": math.degrees(float(yaw))}


if "Isaac-Franka-GraspGenX-Tube-Pick-Play-v0" not in gym.registry:
    gym.register("Isaac-Franka-GraspGenX-Tube-Pick-Play-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": GraspGenTubePickEnvCfg, "rsl_rl_cfg_entry_point": TestTubeLiftPPORunnerCfg}, disable_env_checker=True)
