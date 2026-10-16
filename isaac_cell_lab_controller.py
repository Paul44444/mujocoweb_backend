"""Simple physical cube lift reference for the new lab; IK, not a learned actor."""
import torch
from isaaclab.utils import math as math_utils
from isaac_rack_controller import RackInsertionController


class CellLabCubeController(RackInsertionController):
    def reset(self):
        self.stage, self.ticks, self.target = "settle", 0, None

    def __call__(self, observations):
        robot, env = self.robot, self.env
        hand = robot.data.body_pose_w[:1, self.hand]
        obj = self.object_pose()
        self.ticks += 1
        actions = torch.zeros(env.action_space.shape, device=env.device)
        actions[:, :7] = (robot.data.joint_pos[:1, :7] - robot.data.default_joint_pos[:1, :7]) / .5
        actions[:, 7] = 1.
        if self.stage == "settle":
            if self.ticks < 40:
                return actions
            self.start_z = float(obj[0, 2])
            self.grasp = obj[:, :3].clone()
            self.grasp[:, 2] += .1034
            self.orientation = obj.new_tensor([[0., 1., 0., 0.]])
            self.target = hand[:, :3].clone()
            self.stage, self.ticks = "approach", 0
        goal = self.grasp.clone()
        if self.stage == "approach":
            goal[:, 2] += .15
        elif self.stage in {"lift", "done"}:
            goal[:, 2] += .18
        if self.stage == "failed":
            return actions
        delta = goal - self.target
        self.target += delta * (.0015 / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(.0015))
        self.lula.set_robot_base_pose(robot.data.root_pos_w[0].cpu().numpy(), robot.data.root_quat_w[0].cpu().numpy())
        solution, feasible = self.lula.compute_inverse_kinematics("panda_hand", self.target[0].cpu().numpy(),
            self.orientation[0].cpu().numpy(), warm_start=robot.data.joint_pos[0, :7].cpu().numpy())
        if feasible:
            actions[:, :7] = (torch.as_tensor(solution.copy(), device=env.device).reshape(1, 7) - robot.data.default_joint_pos[:1, :7]) / .5
        actions[:, 7] = -1. if self.stage in {"close", "lift", "done"} else 1.
        reached = float(torch.linalg.vector_norm(hand[:, :3] - goal)) < .005 and float(math_utils.quat_error_magnitude(hand[:, 3:7], self.orientation)) < .05
        if self.stage in {"approach", "grasp", "lift"} and reached and self.ticks > 20:
            self.stage = {"approach": "grasp", "grasp": "close", "lift": "done"}[self.stage]
            if self.stage == "done" and float(obj[0, 2]) < self.start_z + .12:
                self.stage = "failed"
            self.ticks = 0
        elif self.stage == "close" and self.ticks >= 100:
            self.stage, self.ticks = "lift", 0
        elif self.stage not in {"done", "failed"} and self.ticks > 1200:
            self.stage = "failed"
        return actions
