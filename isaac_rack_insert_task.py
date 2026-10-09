"""Additional upright-tube insertion task; never replaces the old labware tasks."""
import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaac_labware_task import FrankaLabwarePlacementEnvCfg, LabwarePlacementPPORunnerCfg, LABWARE_ASSET_DIR

RACK_POSITION = (0.55, 0.12, 0.0)
# Measured from the source mesh, not its bounding box. Hole radius 9.560 mm;
# the 7.5-mm tube collider has approximately 1.88 mm clearance to the polygon.
HOLE_CENTER = (0.5636571, 0.1063427)
# The source rack has a lower support shelf at about 31.7 mm. The tube
# rests there after passing the upper hole, rather than reaching the table.
SEATED_CENTER_Z = 0.0777
RACK_TOP_Z = 0.0696992


def insertion_reward(env, phase: str):
    obj = env.scene["object"].data
    position = obj.root_pos_w - env.scene.env_origins
    xy_error = torch.linalg.vector_norm(position[:, :2] - position.new_tensor(HOLE_CENTER), dim=-1)
    up = quat_apply(obj.root_quat_w, position.new_tensor((0., 0., 1.)).expand_as(position))[:, 2]
    upright = ((up - 0.85) / 0.15).clamp(0., 1.)
    if phase == "align":
        # Only moving the tube over the opening while clear of the top plate
        # earns alignment reward, not pushing it towards the rack on the table.
        return (1. - torch.tanh(xy_error / 0.04)) * upright * (position[:, 2] > RACK_TOP_Z + 0.045)
    seated = (xy_error < 0.0015) & ((position[:, 2] - SEATED_CENTER_Z).abs() < 0.008) & (up > 0.98)
    if phase == "insert":
        return (1. - torch.tanh(xy_error / 0.008)) * upright * (1. - torch.tanh((position[:, 2] - SEATED_CENTER_Z).abs() / 0.035))
    if phase == "release":
        robot = env.scene["robot"]
        finger_ids, _ = robot.find_joints("panda_finger_joint.*")
        open_gripper = robot.data.joint_pos[:, finger_ids].mean(dim=-1) > 0.025
        stable = torch.linalg.vector_norm(obj.root_lin_vel_w, dim=-1) < 0.03
        return (seated & open_gripper & stable).float()
    raise ValueError(f"Unknown insertion reward phase: {phase}")


@configclass
class FrankaRackInsertEnvCfg(FrankaLabwarePlacementEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0
        self.scene.object.init_state.pos = (0.43, -0.16, 0.047)
        self.scene.object.init_state.rot = (1., 0., 0., 0.)
        # One mesh is BOTH the visual and the collider: no displaced visual,
        # invisible box, convex hull, or separate over-sized rails.
        self.scene.rack_visual = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/RackVisual",
            init_state=AssetBaseCfg.InitialStateCfg(pos=RACK_POSITION),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(LABWARE_ASSET_DIR / "rack_insert_collision.usda"),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.34, 0.56), roughness=0.28),
            ),
        )
        self.scene.rack_left = self.scene.rack_right = None
        self.scene.rack_front = self.scene.rack_back = None
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (0., 0.), "yaw": (-0.15, 0.15)
        }
        for axis, value in zip("xyz", (*HOLE_CENTER, SEATED_CENTER_Z)):
            setattr(self.commands.object_pose.ranges, f"pos_{axis}", (value, value))
        self.rewards.reaching_object.weight = 2.0
        self.rewards.lifting_object.weight = 4.0
        self.rewards.lifting_object.params["minimal_height"] = 0.12
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None
        self.rewards.rack_alignment = RewardTermCfg(func=insertion_reward, weight=8.0, params={"phase": "align"})
        self.rewards.rack_insertion = RewardTermCfg(func=insertion_reward, weight=12.0, params={"phase": "insert"})
        self.rewards.rack_released = RewardTermCfg(func=insertion_reward, weight=30.0, params={"phase": "release"})


@configclass
class FrankaRackInsertEnvCfg_PLAY(FrankaRackInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class RackInsertPPORunnerCfg(LabwarePlacementPPORunnerCfg):
    experiment_name = "franka_upright_tube_rack_insert"


for task_id, cfg in (
    ("Isaac-Franka-Upright-Tube-Rack-Insert-v0", FrankaRackInsertEnvCfg),
    ("Isaac-Franka-Upright-Tube-Rack-Insert-Play-v0", FrankaRackInsertEnvCfg_PLAY),
):
    if task_id not in gym.registry:
        gym.register(id=task_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                     kwargs={"env_cfg_entry_point": cfg, "rsl_rl_cfg_entry_point": RackInsertPPORunnerCfg})
