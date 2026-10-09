"""Separate RGB-D marker experiment. No changes to the original rack task."""
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaac_rack_insert_task import FrankaRackInsertEnvCfg_PLAY, RackInsertPPORunnerCfg, RACK_POSITION

from isaac_rack_markers import RACK_MARKERS, TUBE_UPPER_OFFSET, TUBE_LOWER_OFFSET

def marker(path, position, color, radius):
    return AssetBaseCfg(prim_path="{ENV_REGEX_NS}/" + path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.SphereCfg(radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color, emissive_color=tuple(c * .15 for c in color), roughness=1.)))

@configclass
class RackVisionEnvCfg(FrankaRackInsertEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.debug_vis = False
        self.scene.tube_upper_marker = marker("Object/UpperMarker", (0., 0., TUBE_UPPER_OFFSET), (1., 0., 1.), .012)
        self.scene.tube_lower_marker = marker("Object/LowerMarker", (0., 0., TUBE_LOWER_OFFSET), (0., 1., 0.), .012)
        for index, position in enumerate(RACK_MARKERS):
            setattr(self.scene, f"rack_marker_{index}", marker(f"RackVisual/Marker{index}", position, (1., .055, 0.), .007))
        for name in ("rack_camera_a", "rack_camera_b"):
            setattr(self.scene, name, CameraCfg(prim_path="{ENV_REGEX_NS}/" + name, update_period=0.,
                width=960, height=720, data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=22., horizontal_aperture=20.955,
                    clipping_range=(.05, 3.)),
                offset=CameraCfg.OffsetCfg(pos=(.7, -.5, .6), rot=(1., 0., 0., 0.), convention="ros")))

gym.register(id="Isaac-Franka-Rack-Insert-Vision-Play-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": RackVisionEnvCfg, "rsl_rl_cfg_entry_point": RackInsertPPORunnerCfg})

def aim_cameras(env):
    import torch
    from pxr import UsdShade
    base = env.unwrapped
    # UsdFileCfg binds the rack material stronger than its descendants;
    # allow the added markers to keep their independently coded colors.
    rack = base.sim.stage.GetPrimAtPath("/World/envs/env_0/RackVisual")
    binding = UsdShade.MaterialBindingAPI(rack).GetDirectBindingRel()
    if binding:
        binding.SetMetadata("bindMaterialAs", "weakerThanDescendants")
    for name, eye in (("rack_camera_a", (.75, -.55, .65)), ("rack_camera_b", (.15, -.40, .72))):
        camera = base.scene[name]
        camera.set_world_poses_from_view(torch.tensor([eye], device=base.device),
            torch.tensor([[.49, -.02, .10]], device=base.device))
        # These cameras are fixed. Refresh calibration once after aiming,
        # rather than querying USD/Fabric transforms for every image.
        camera.reset()
