"""Separate vision-oriented Franka lift task for the web platform.

The actor never receives the exact cube position in the web playback task.
Training uses a noisy three-coordinate perception proxy; playback replaces it
with XYZ triangulated from two fixed calibrated RGB cameras.  The three-value
interface remains compatible with existing state policies and a later
FoundationPose adapter.
"""

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, subtract_frame_transforms
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg


_camera_position_cache: dict[int, torch.Tensor] = {}


def estimated_object_position(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_noise_std: float = 0.008,
) -> torch.Tensor:
    """Training proxy for the future RGB-D estimator, preserving the 3-D state-policy ABI."""
    object_asset = env.scene[object_cfg.name]
    robot_asset = env.scene[robot_cfg.name]
    position, _ = subtract_frame_transforms(
        robot_asset.data.root_pos_w,
        robot_asset.data.root_quat_w,
        object_asset.data.root_pos_w[:, :3],
    )
    return position + torch.randn_like(position) * position_noise_std


def camera_object_position(
    env,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("vision_camera"),
    side_camera_cfg: SceneEntityCfg = SceneEntityCfg("vision_camera_side"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    saturation_threshold: float = 42.0,
) -> torch.Tensor:
    """Triangulate cube XYZ exclusively from two calibrated RGB images."""
    camera = env.scene[camera_cfg.name]
    side_camera = env.scene[side_camera_cfg.name]
    robot = env.scene[robot_cfg.name]
    cache_key = id(env)
    cached = _camera_position_cache.get(cache_key)

    def image_rays(sensor):
        rgb = sensor.data.output["rgb"][..., :3].to(torch.float32)
        origins, directions, visible = [], [], []
        for index in range(rgb.shape[0]):
            chroma = rgb[index].amax(dim=-1) - rgb[index].amin(dim=-1)
            mask = chroma > saturation_threshold
            if int(mask.sum().item()) < 6:
                origins.append(sensor.data.pos_w[index])
                directions.append(torch.zeros(3, dtype=rgb.dtype, device=rgb.device))
                visible.append(False)
                continue
            rows, columns = torch.where(mask)
            intrinsics = sensor.data.intrinsic_matrices[index]
            pixel_u = columns.to(rgb.dtype).median()
            pixel_v = rows.to(rgb.dtype).median()
            ray = torch.stack((
                (pixel_u - intrinsics[0, 2]) / intrinsics[0, 0],
                (pixel_v - intrinsics[1, 2]) / intrinsics[1, 1],
                torch.ones((), dtype=rgb.dtype, device=rgb.device),
            )).unsqueeze(0)
            world_ray = quat_apply(sensor.data.quat_w_ros[index:index + 1], ray)[0]
            origins.append(sensor.data.pos_w[index])
            directions.append(world_ray / world_ray.norm().clamp_min(1.0e-6))
            visible.append(True)
        return torch.stack(origins), torch.stack(directions), visible

    origins_a, rays_a, visible_a = image_rays(camera)
    origins_b, rays_b, visible_b = image_rays(side_camera)
    estimates = []
    for index in range(origins_a.shape[0]):
        if visible_a[index] and visible_b[index]:
            offset = origins_a[index] - origins_b[index]
            ray_dot = torch.dot(rays_a[index], rays_b[index])
            denominator = (1.0 - ray_dot.square()).clamp_min(1.0e-5)
            first_distance = (ray_dot * torch.dot(rays_b[index], offset) - torch.dot(rays_a[index], offset)) / denominator
            second_distance = (torch.dot(rays_b[index], offset) - ray_dot * torch.dot(rays_a[index], offset)) / denominator
            point_a = origins_a[index] + first_distance * rays_a[index]
            point_b = origins_b[index] + second_distance * rays_b[index]
            world_position = ((point_a + point_b) * 0.5).unsqueeze(0)
            position_b, _ = subtract_frame_transforms(
                robot.data.root_pos_w[index:index + 1],
                robot.data.root_quat_w[index:index + 1],
                world_position,
            )
            estimate = position_b[0]
        elif cached is not None and index < cached.shape[0]:
            estimate = cached[index]
        else:
            estimate = torch.tensor([0.5, 0.0, 0.055], dtype=origins_a.dtype, device=origins_a.device)
        estimates.append(estimate)
    result = torch.stack(estimates)
    _camera_position_cache[cache_key] = result.detach().clone()
    return result


@configclass
class VisionFrankaCubeLiftEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.object_position = ObsTerm(
            func=estimated_object_position,
            params={"position_noise_std": 0.008},
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
            data_types=["rgb"],
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
        self.scene.vision_camera_side = CameraCfg(
            prim_path="{ENV_REGEX_NS}/VisionCameraSide",
            update_period=0.0,
            height=360,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=1.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 5.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, -0.9, 0.85),
                rot=(0.0, 0.0, 0.0, 1.0),
                convention="ros",
            ),
        )
        # The RTX camera has no valid frame during the environment's initial reset.
        # The persistent web worker replaces these three proxy values with the
        # calibrated RGB estimate after every rendered simulation step.


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
