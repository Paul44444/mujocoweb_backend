"""Franka test-tube placement task, kept separate from the cube tasks."""

from __future__ import annotations

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg


def rack_bar(name: str, position: tuple[float, float, float], size: tuple[float, float, float]) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.12, 0.32, 0.48), metallic=0.15, roughness=0.32),
            physics_material=RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.65),
        ),
    )


@configclass
class FrankaLabwarePlacementEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # A light, graspable 10 cm laboratory tube.  The detailed source USDs
        # remain available under assets/labware for the next visual refinement.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.48, -0.16, 0.052), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CapsuleCfg(
                radius=0.012,
                height=0.10,
                axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=2,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.018),
                physics_material=RigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.55),
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.25, 0.78, 0.92), opacity=0.72, roughness=0.18),
            ),
        )

        # Open collision rails form a real slot rather than a decorative mesh.
        # The target slot is centered at (0.55, 0.18).
        self.scene.rack_left = rack_bar("RackLeft", (0.505, 0.18, 0.055), (0.018, 0.15, 0.11))
        self.scene.rack_right = rack_bar("RackRight", (0.595, 0.18, 0.055), (0.018, 0.15, 0.11))
        self.scene.rack_front = rack_bar("RackFront", (0.55, 0.115, 0.055), (0.108, 0.018, 0.11))
        self.scene.rack_back = rack_bar("RackBack", (0.55, 0.245, 0.055), (0.108, 0.018, 0.11))

        self.commands.object_pose.ranges.pos_x = (0.55, 0.55)
        self.commands.object_pose.ranges.pos_y = (0.18, 0.18)
        self.commands.object_pose.ranges.pos_z = (0.055, 0.055)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)
        self.commands.object_pose.resampling_time_range = (1.0e6, 1.0e6)

        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.035, 0.035),
            "y": (-0.025, 0.025),
            "z": (0.0, 0.0),
            "yaw": (-0.20, 0.20),
        }
        self.rewards.lifting_object.params["minimal_height"] = 0.075
        self.rewards.object_goal_tracking.params["minimal_height"] = 0.045
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = 0.045
        self.rewards.object_goal_tracking.weight = 22.0
        self.rewards.object_goal_tracking_fine_grained.weight = 12.0


@configclass
class FrankaLabwarePlacementEnvCfg_PLAY(FrankaLabwarePlacementEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class LabwarePlacementPPORunnerCfg(LiftCubePPORunnerCfg):
    experiment_name = "franka_labware_placement"


def register_tasks() -> None:
    registrations = {
        "Isaac-Franka-Labware-Placement-v0": FrankaLabwarePlacementEnvCfg,
        "Isaac-Franka-Labware-Placement-Play-v0": FrankaLabwarePlacementEnvCfg_PLAY,
    }
    for task_id, config in registrations.items():
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": config,
                "rsl_rl_cfg_entry_point": LabwarePlacementPPORunnerCfg,
            },
            disable_env_checker=True,
        )


register_tasks()
