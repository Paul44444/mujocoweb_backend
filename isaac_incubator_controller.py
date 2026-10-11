"""Physical pickup, horizontal loading and automatic door; no object teleporting."""
import math
import torch
from isaaclab.utils import math as math_utils
from isaac_rack_controller import RackInsertionController
from isaac_incubator_task import DEPOSIT_CENTER, INCUBATOR_POSITION, FLOOR_Z


class IncubatorLoadingController(RackInsertionController):
    def reset(self):
        super().reset()
        self.stage = "open_door"
        self.grasp_offset = None

    def __call__(self, observations):
        env, robot = self.env, self.robot
        obj = self.object_pose()
        hand = robot.data.body_pose_w[:1, self.hand]
        door = env.scene["incubator"]
        self.ticks += 1
        actions = torch.zeros(env.action_space.shape, device=env.device)
        actions[:, :7] = (robot.data.joint_pos[:1, :7] - robot.data.default_joint_pos[:1, :7]) / .5
        actions[:, 7] = 1.
        closed = self.stage in {"close_door", "done"}
        door.set_joint_position_target(obj.new_full((1, 1), 0. if closed else -math.radians(110)))
        if self.stage == "open_door":
            if self.ticks > 60 and float(door.data.joint_pos[0, 0]) < -1.85:
                self.start_z = float(obj[0, 2])
                self.grasp = obj[:, :3].clone()
                self.grasp[:, 2] += .1374
                self.orientation = obj.new_tensor([[0., 1., 0., 0.]])
                self.target = hand[:, :3].clone()
                self.stage, self.ticks = "approach", 0
            elif self.ticks > 1400:
                print("Incubator door blocked; reset to retry", flush=True)
                self.stage = "failed"
            return actions
        if self.stage in {"failed", "done"}:
            return actions
        if self.stage == "close_door":
            if self.ticks > 100 and abs(float(door.data.joint_pos[0, 0])) < .025:
                self.stage, self.ticks = ("done" if self.deposited() else "failed"), 0
            elif self.ticks > 1000:
                self.stage = "failed"
            return actions
        goal = self.grasp.clone()
        if self.stage == "approach":
            goal[:, 2] += .12
        elif self.stage == "lift":
            # Extract vertically above the rack AND neighbouring tube tops
            # before starting the horizontal rotation.
            goal[:, 2] += .18
        elif self.stage == "rotate":
            goal = self.rotation_position.clone()
        elif self.stage in {"preinsert", "insert", "release", "withdraw", "park"}:
            center = obj.new_tensor([DEPOSIT_CENTER]) + env.scene.env_origins[:1]
            goal = center + self.grasp_offset
            if self.stage in {"preinsert", "withdraw"}:
                goal[:, 1] -= .21
            if self.stage == "park":
                goal[:, 1] -= .25
                goal[:, 0] += .16
                goal[:, 2] += .12
        speed = .0006 if self.stage in {"insert", "withdraw"} else .0012
        delta = goal - self.target
        self.target += delta * (speed / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(speed))
        self.lula.set_robot_base_pose(robot.data.root_pos_w[0].cpu().numpy(), robot.data.root_quat_w[0].cpu().numpy())
        solution, feasible = self.lula.compute_inverse_kinematics("panda_hand", self.target[0].cpu().numpy(),
            self.orientation[0].cpu().numpy(), warm_start=robot.data.joint_pos[0, :7].cpu().numpy())
        if feasible:
            actions[:, :7] = (torch.as_tensor(solution.copy(), device=env.device).reshape(1, 7) - robot.data.default_joint_pos[:1, :7]) / .5
        actions[:, 7] = -1. if self.stage in {"close", "lift", "rotate", "preinsert", "insert"} else 1.
        error = float(torch.linalg.vector_norm(hand[:, :3] - goal))
        angular = float(math_utils.quat_error_magnitude(hand[:, 3:7], self.orientation))
        advance = error < .004 and angular < .05 and self.ticks > 30
        if self.stage == "lift" and advance:
            if float(obj[0, 2]) < self.start_z + .12:
                self.stage = "failed"
            else:
                self.rotation_position = hand[:, :3].clone()
                # Rotate tube axis from +Z to -Y, with hand on the front side.
                desired = obj.new_tensor([[math.sqrt(.5), math.sqrt(.5), 0., 0.]])
                relative = math_utils.quat_mul(math_utils.quat_inv(obj[:, 3:7]), hand[:, 3:7])
                self.orientation = math_utils.quat_mul(desired, relative)
                self.stage = "rotate"
            self.ticks = 0
        elif self.stage == "rotate" and advance:
            self.grasp_offset = (hand[:, :3] - obj[:, :3]).clone()
            self.stage, self.ticks = "preinsert", 0
        elif self.stage in {"approach", "grasp", "preinsert", "insert", "withdraw", "park"} and advance:
            self.stage = {"approach": "grasp", "grasp": "close", "preinsert": "insert",
                "insert": "release", "withdraw": "park", "park": "close_door"}[self.stage]
            self.ticks = 0
        elif self.stage in {"close", "release"} and self.ticks > 120:
            self.stage = "lift" if self.stage == "close" else "withdraw"
            self.ticks = 0
        elif self.ticks > 1400:
            print(f"Incubator loading stalled at {self.stage}: error {error:.4f}", flush=True)
            self.stage = "failed"
        return actions

    def deposited(self):
        obj = self.env.scene["object"].data
        p = obj.root_pos_w[0] - self.env.scene.env_origins[0]
        return (abs(float(p[0]) - INCUBATOR_POSITION[0]) < .12
            and INCUBATOR_POSITION[1] - .10 < float(p[1]) < INCUBATOR_POSITION[1] + .12
            and FLOOR_Z < float(p[2]) < FLOOR_Z + .04
            and float(torch.linalg.vector_norm(obj.root_lin_vel_w[0])) < .03)
