"""Contact-only upright pickup/insertion reference controller (not a neural policy)."""
import torch
from isaaclab.utils import math as math_utils
from isaac_rack_insert_task import HOLE_CENTER, SEATED_CENTER_Z


class RackInsertionController:
    web_demo_control = True
    web_reference_control = True

    def __init__(self, env):
        self.env = env.unwrapped
        self.robot = self.env.scene["robot"]
        self.hand = self.robot.body_names.index("panda_hand")
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.robot_motion.motion_generation")
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, interface_config_loader
        self.lula = LulaKinematicsSolver(**interface_config_loader.load_supported_lula_kinematics_solver_config("Franka"))
        self.reset()

    def reset(self):
        self.stage = "settle"
        self.ticks = 0
        self.target = None

    def object_pose(self):
        return self.env.scene["object"].data.root_pose_w[:1]

    def destination(self):
        return self.robot.data.root_pos_w.new_tensor([[*HOLE_CENTER, SEATED_CENTER_Z]]) + self.env.scene.env_origins[:1]

    def __call__(self, observations):
        env, robot = self.env, self.robot
        hand = robot.data.body_pose_w[:1, self.hand]
        obj = self.object_pose()
        destination = self.destination()
        self.ticks += 1
        actions = torch.zeros(env.action_space.shape, device=env.device)
        actions[:, :7] = (robot.data.joint_pos[:1, :7] - robot.data.default_joint_pos[:1, :7]) / 0.5
        actions[:, 7] = 1.
        if self.stage == "settle":
            if self.ticks < 40:
                return actions
            self.start_z = float(obj[0, 2])
            self.grasp = obj[:, :3].clone()
            # Fingers meet the exposed upper half of the upright tube.
            self.grasp[:, 2] += 0.034 + 0.1034
            self.orientation = obj.new_tensor([[0., 1., 0., 0.]])
            self.target = hand[:, :3].clone()
            self.stage, self.ticks = "approach", 0
        goal = self.grasp.clone()
        if self.stage == "approach":
            goal[:, 2] += 0.12
        elif self.stage == "lift":
            goal[:, 2] += 0.16
        elif self.stage in {"transfer", "lower", "release", "retreat", "done"}:
            goal[:, :2] = destination[:, :2]
            goal[:, 2] = destination[:, 2] + 0.034 + 0.1034
            if self.stage == "transfer":
                goal[:, 2] += 0.14
            elif self.stage in {"retreat", "done"}:
                goal[:, 2] += 0.14
            elif self.stage == "lower":
                # Follow the actual held tube, correcting grasp slippage and
                # small offsets instead of trusting an ideal attachment.
                offset = hand[:, :3] - obj[:, :3]
                goal[:, :2] = destination[:, :2] + offset[:, :2]
                goal[:, 2] = destination[:, 2] + offset[:, 2] + 0.001
        if self.stage == "failed":
            return actions
        delta = goal - self.target
        speed = 0.00035 if self.stage == "lower" else 0.0015
        self.target += delta * (speed / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(speed))
        self.lula.set_robot_base_pose(robot.data.root_pos_w[0].cpu().numpy(), robot.data.root_quat_w[0].cpu().numpy())
        solution, feasible = self.lula.compute_inverse_kinematics("panda_hand", self.target[0].cpu().numpy(), self.orientation[0].cpu().numpy(), warm_start=robot.data.joint_pos[0, :7].cpu().numpy())
        if feasible:
            actions[:, :7] = (torch.as_tensor(solution.copy(), device=env.device).reshape(1, 7) - robot.data.default_joint_pos[:1, :7]) / 0.5
        actions[:, 7] = -1. if self.stage in {"close", "lift", "transfer", "lower"} else 1.
        error = float(torch.linalg.vector_norm(hand[:, :3] - goal))
        angular = float(math_utils.quat_error_magnitude(hand[:, 3:7], self.orientation))
        advance = error < (0.0015 if self.stage == "lower" else 0.004) and angular < 0.04 and self.ticks > 20
        if self.stage in {"approach", "grasp", "lift", "transfer", "lower", "retreat"} and advance:
            if self.stage == "lift" and float(obj[0, 2]) < self.start_z + 0.10:
                self.stage = "failed"
                print("Rack insertion: pickup failed, refusing to continue", flush=True)
            else:
                if self.stage == "lift":
                    # A real friction grasp can acquire a small tilt. Preserve
                    # the measured hand-to-object rotation, but command an
                    # upright object before descending through the tight hole.
                    self.orientation = math_utils.quat_mul(math_utils.quat_inv(obj[:, 3:7]), hand[:, 3:7]).clone()
                self.stage = {"approach": "grasp", "grasp": "close", "lift": "transfer", "transfer": "lower", "lower": "release", "retreat": "done"}[self.stage]
            self.ticks = 0
        elif self.stage in {"close", "release"} and self.ticks >= 100:
            self.stage = "lift" if self.stage == "close" else "retreat"
            self.ticks = 0
        elif self.stage not in {"done", "failed"} and self.ticks > 1200:
            print(f"Rack insertion stalled at {self.stage}; reset to retry", flush=True)
            self.stage = "failed"
        return actions
