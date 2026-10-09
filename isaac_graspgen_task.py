"""Separate pretrained-grasp pickup scene; existing labware tasks stay intact."""
import gymnasium as gym
from isaaclab.assets import AssetBaseCfg
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
            setattr(self.scene, name, AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                init_state=AssetBaseCfg.InitialStateCfg(pos=(x, -.16, .02)),
                spawn=sim_utils.CuboidCfg(size=(.008, .025, .04),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.15, .3, .4)))))


if "Isaac-Franka-GraspGenX-Tube-Pick-Play-v0" not in gym.registry:
    gym.register("Isaac-Franka-GraspGenX-Tube-Pick-Play-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": GraspGenTubePickEnvCfg, "rsl_rl_cfg_entry_point": TestTubeLiftPPORunnerCfg}, disable_env_checker=True)
