"""Franka test-tube placement task, kept separate from the cube tasks."""

from __future__ import annotations

import gymnasium as gym
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg


LABWARE_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "labware"


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
            visible=False,
        ),
    )


@configclass
class FrankaLabwarePlacementEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Keep a simple, stable collision body and attach the detailed mesh as
        # its child.  This avoids using the high-poly imported surface as a
        # PhysX collider while the rendered tube follows the rigid body exactly.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.48, -0.16, 0.052), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CylinderCfg(
                # Keep the collider several millimetres inside the rendered
                # glass. The Franka fingertips have their own contact envelope;
                # using the full visual diameter makes them appear to touch the
                # tube while a visible gap is still present.
                radius=0.0075,
                height=0.092,
                axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=2,
                    max_depenetration_velocity=0.20,
                    max_linear_velocity=1.5,
                    max_angular_velocity=8.0,
                    linear_damping=0.15,
                    angular_damping=0.20,
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.018),
                physics_material=RigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.55),
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.25, 0.78, 0.92), opacity=0.0),
            ),
        )
        self.scene.tube_visual = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Object/TubeVisual",
            # The wrapper origin is at the tube base; the cylinder origin is
            # at its center, hence the local -5 cm offset.
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(LABWARE_ASSET_DIR / "test_tube.usda"),
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=(0.08, 0.72, 0.95), emissive_color=(0.02, 0.12, 0.18), opacity=1.0, roughness=0.24
                ),
            ),
        )

        # Detailed rack visuals.  Invisible primitive rails below remain the
        # collision representation, which is considerably more stable for PPO.
        self.scene.rack_visual = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/RackVisual",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.18, 0.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(LABWARE_ASSET_DIR / "test_tube_rack.usda"),
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=(0.10, 0.34, 0.56), metallic=0.18, roughness=0.28
                ),
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
        self.rewards.lifting_object.params["minimal_height"] = 0.12
        self.rewards.object_goal_tracking.params["minimal_height"] = 0.10
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = 0.10
        self.rewards.object_goal_tracking.weight = 22.0
        self.rewards.object_goal_tracking_fine_grained.weight = 12.0


@configclass
class FrankaTestTubeLiftEnvCfg(FrankaLabwarePlacementEnvCfg):
    """First-stage curriculum: reach, grasp, and lift the tube only."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 6.0
        self.rewards.reaching_object.weight = 2.0
        self.rewards.lifting_object.params["minimal_height"] = 0.12
        self.rewards.lifting_object.weight = 25.0
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None


@configclass
class FrankaTestTubeLiftEnvCfg_PLAY(FrankaTestTubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaLabwarePlacementEnvCfg_PLAY(FrankaLabwarePlacementEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


_BASE_LIFT_RUNNER = LiftCubePPORunnerCfg()


@configclass
class TestTubeLiftPPORunnerCfg(LiftCubePPORunnerCfg):
    experiment_name = "franka_test_tube_lift"
    policy = _BASE_LIFT_RUNNER.policy.replace(init_noise_std=0.65, noise_std_type="log")
    algorithm = _BASE_LIFT_RUNNER.algorithm.replace(
        entropy_coef=0.003, learning_rate=5.0e-5, max_grad_norm=0.5
    )


@configclass
class LabwarePlacementPPORunnerCfg(LiftCubePPORunnerCfg):
    experiment_name = "franka_labware_placement"
    policy = _BASE_LIFT_RUNNER.policy.replace(init_noise_std=0.65, noise_std_type="log")
    algorithm = _BASE_LIFT_RUNNER.algorithm.replace(
        entropy_coef=0.003, learning_rate=5.0e-5, max_grad_norm=0.5
    )


def register_tasks() -> None:
    registrations = {
        "Isaac-Franka-Test-Tube-Lift-v0": (FrankaTestTubeLiftEnvCfg, TestTubeLiftPPORunnerCfg),
        "Isaac-Franka-Test-Tube-Lift-Play-v0": (FrankaTestTubeLiftEnvCfg_PLAY, TestTubeLiftPPORunnerCfg),
        "Isaac-Franka-Labware-Placement-v0": (FrankaLabwarePlacementEnvCfg, LabwarePlacementPPORunnerCfg),
        "Isaac-Franka-Labware-Placement-Play-v0": (FrankaLabwarePlacementEnvCfg_PLAY, LabwarePlacementPPORunnerCfg),
    }
    for task_id, (config, runner_config) in registrations.items():
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": config,
                "rsl_rl_cfg_entry_point": runner_config,
            },
            disable_env_checker=True,
        )


register_tasks()
