"""Separate vision-oriented Franka lift task for the web platform.

The actor never receives the exact cube position.  The first implementation
uses a noisy, intermittently unavailable pose-estimator proxy with confidence;
the fixed RGB-D sensor in the play task is the integration point for replacing
that proxy with FoundationPose without changing the policy interface.
"""

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg


def estimated_object_pose(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_noise_std: float = 0.008,
    dropout_probability: float = 0.025,
) -> torch.Tensor:
    """Return a camera-estimator-like XYZ measurement plus confidence.

    Ground truth is used only inside this temporary sensor model to generate a
    noisy measurement.  It is never exposed directly to the actor.  The output
    shape intentionally matches the future FoundationPose adapter: x, y, z and
    confidence, all expressed in the robot root frame.
    """
    object_asset = env.scene[object_cfg.name]
    robot_asset = env.scene[robot_cfg.name]
    position, _ = subtract_frame_transforms(
        robot_asset.data.root_pos_w,
        robot_asset.data.root_quat_w,
        object_asset.data.root_pos_w[:, :3],
    )
    noisy_position = position + torch.randn_like(position) * position_noise_std
    visible = torch.rand((position.shape[0], 1), device=position.device) >= dropout_probability
    confidence = torch.where(
        visible,
        torch.full_like(visible, 0.92, dtype=position.dtype),
        torch.zeros_like(visible, dtype=position.dtype),
    )
    measured_position = torch.where(visible, noisy_position, torch.zeros_like(noisy_position))
    return torch.cat((measured_position, confidence), dim=-1)


@configclass
class VisionFrankaCubeLiftEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.object_position = ObsTerm(
            func=estimated_object_pose,
            params={"position_noise_std": 0.008, "dropout_probability": 0.025},
        )


@configclass
class VisionFrankaCubeLiftEnvCfg_PLAY(VisionFrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.vision_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/VisionCamera",
            update_period=0.0,
            height=360,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=1.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 5.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.15, 0.65, 1.10),
                rot=(-0.1736, 0.2549, 0.7903, -0.5299),
                convention="ros",
            ),
        )


@configclass
class VisionLiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_lift_vision"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def register_tasks() -> None:
    registrations = {
        "Isaac-Lift-Cube-Franka-Vision-v0": VisionFrankaCubeLiftEnvCfg,
        "Isaac-Lift-Cube-Franka-Vision-Play-v0": VisionFrankaCubeLiftEnvCfg_PLAY,
    }
    for task_id, config in registrations.items():
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": config,
                "rsl_rl_cfg_entry_point": VisionLiftCubePPORunnerCfg,
            },
            disable_env_checker=True,
        )


register_tasks()
