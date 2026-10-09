"""Execute pretrained GraspGenX poses with Isaac Lab IK and real contacts.

GraspGenX chooses the grasp. Approach/close/lift is an explicit motion state
machine, not an end-to-end neural arm policy. No object attachment is used.
"""
import json
from pathlib import Path

import torch
from isaaclab.utils import math as math_utils


class GraspGenTubeController:
    web_demo_control = True

    def __init__(self, env, path):
        self.env = env.unwrapped
        self.robot = self.env.scene["robot"]
        self.hand = self.robot.body_names.index("panda_hand")
        payload = json.loads(Path(path).read_text())
        if payload.get("model") != "NVIDIA GraspGenX" or payload.get("gripper") != "franka_panda":
            raise ValueError("Not a validated Franka GraspGenX prediction file")
        self.grasps = payload["grasps"]
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.robot_motion.motion_generation")
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, interface_config_loader
        self.lula = LulaKinematicsSolver(**interface_config_loader.load_supported_lula_kinematics_solver_config("Franka"))
        self.reset()

    def reset(self):
        self.stage = "plan"
        self.stage_steps = 0
        self.settle_steps = 0
        self.target_w = None

    def __call__(self, observations):
        robot, env = self.robot, self.env
        hand_pose = robot.data.body_pose_w[:1, self.hand]
        if self.stage == "plan":
            self.settle_steps += 1
            if self.settle_steps < 30:
                actions = torch.zeros(env.action_space.shape, device=env.device)
                actions[:, :7] = (robot.data.joint_pos[:1, :7] - robot.data.default_joint_pos[:1, :7]) / 0.5
                actions[:, 7] = 1.0
                return actions
            obj_pose = env.scene["object"].data.root_pose_w[:1]
            candidates = []
            for grasp in self.grasps:
                q = torch.tensor([[grasp["orientation"]["w"], *grasp["orientation"]["xyz"]]], device=env.device)
                p = torch.tensor([grasp["position"]], device=env.device)
                position, rotation = math_utils.combine_frame_transforms(obj_pose[:, :3], obj_pose[:, 3:7], p, q)
                # Official GraspGenX Franka URDF has world -> panda_hand
                # Rz(+pi/2). Predictions describe world, not panda_hand.
                rotation = math_utils.quat_mul(rotation, torch.tensor([[0.70710678, 0., 0., 0.70710678]], device=env.device))
                # Seating allowance for our Isaac finger-pad collision shape,
                # experimentally checked in the horizontal pickup setup.
                # Not a claim that model predictions alone solve arm motion.
                position += math_utils.matrix_from_quat(rotation)[:, :, 2] * 0.008
                # GraspGenX usually grasps upright tubes from the side. Filter
                # using the actual gripper mesh, rather than forcing top-down.
                corners = torch.tensor(grasp["gripper_bounds_corners"], device=env.device)
                corners_w = corners @ math_utils.matrix_from_quat(obj_pose[:, 3:7])[0].T + obj_pose[0, :3]
                lowest = float(corners_w[:, 2].min())
                if lowest > 0.004 and float(position[0, 2]) > 0.075:
                    self.lula.set_robot_base_pose(robot.data.root_pos_w[0].cpu().numpy(), robot.data.root_quat_w[0].cpu().numpy())
                    flipped = math_utils.quat_mul(rotation, torch.tensor([[0., 0., 0., 1.]], device=env.device))
                    for orientation in (rotation, flipped):
                        solution, feasible = self.lula.compute_inverse_kinematics("panda_hand", position[0].cpu().numpy(), orientation[0].cpu().numpy(), warm_start=robot.data.joint_pos[0, :7].cpu().numpy())
                        clear = feasible and all(float(self.lula.compute_forward_kinematics(link, solution)[0][2]) > minimum for link, minimum in (("panda_link4", 0.15), ("panda_link5", 0.12), ("panda_link6", 0.10)))
                        if clear:
                            candidates.append((grasp["confidence"], position, orientation))
            if not candidates:
                raise RuntimeError("GraspGenX found no table-clear tube grasp; refusing a heuristic fallback")
            _, self.grasp_pos_w, self.grasp_quat_w = max(candidates, key=lambda candidate: candidate[0])
            self.target_w = torch.cat((hand_pose[:, :3].clone(), self.grasp_quat_w), dim=-1)
            self.pregrasp_pos_w = self.grasp_pos_w - math_utils.matrix_from_quat(self.grasp_quat_w)[:, :, 2] * 0.06
            self.stage = "approach"
        goal = self.grasp_pos_w.clone()
        if self.stage == "approach":
            # Establish wrist orientation above the table before descending;
            # a long horizontal standoff can be too close to the robot base.
            goal = self.pregrasp_pos_w.clone()
            goal[:, 2] += 0.15
        elif self.stage == "descend":
            goal = self.pregrasp_pos_w.clone()
        elif self.stage in {"lift", "hold"}:
            goal[:, 2] += 0.20
        delta = goal - self.target_w[:, :3]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        self.target_w[:, :3] += delta * (0.0015 / distance.clamp_min(0.0015)).clamp_max(1.0)
        solution, feasible = self.lula.compute_inverse_kinematics("panda_hand", self.target_w[0, :3].cpu().numpy(), self.target_w[0, 3:7].cpu().numpy(), warm_start=robot.data.joint_pos[0, :7].cpu().numpy())
        joints = torch.tensor(solution.copy(), device=env.device, dtype=torch.float32).reshape(1, 7) if feasible else robot.data.joint_pos[:1, :7]
        actions = torch.zeros(env.action_space.shape, device=env.device)
        actions[:, :7] = (joints - robot.data.default_joint_pos[:1, :7]) / 0.5
        actions[:, 7] = -1.0 if self.stage in {"close", "lift", "hold"} else 1.0
        self.stage_steps += 1
        error = float(torch.linalg.vector_norm(hand_pose[:, :3] - goal))
        angle_error = float(math_utils.quat_error_magnitude(hand_pose[:, 3:7], self.grasp_quat_w))
        if self.stage in {"approach", "descend", "insert", "lift"} and error < 0.004 and angle_error < 0.04 and self.stage_steps > 20:
            self.stage = {"approach": "descend", "descend": "insert", "insert": "close", "lift": "hold"}[self.stage]
            self.stage_steps = 0
        elif self.stage == "close" and self.stage_steps >= 80:
            self.stage, self.stage_steps = "lift", 0
        elif self.stage in {"approach", "descend", "insert", "lift"} and self.stage_steps > 750:
            print(f"GraspGenX motion failed to converge at {self.stage}; use Reset to retry", flush=True)
            self.stage = "failed"
            self.grasp_pos_w = hand_pose[:, :3].clone()
            self.target_w = hand_pose.clone()
        return actions
