"""Reuse validated physical tube handling; adapt to the lab's two-door asset."""
import math
import torch
from isaac_incubator_controller import IncubatorLoadingController
from isaac_cell_lab_incubator_task import DEPOSIT_CENTER, INCUBATOR_POSITION, SHELF_TOP, INCUBATOR_YAW
from isaaclab.utils import math as math_utils


class CellLabIncubatorController(IncubatorLoadingController):
    deposit_center = DEPOSIT_CENTER
    incubator_position = INCUBATOR_POSITION
    floor_z = SHELF_TOP
    incubator_yaw = INCUBATOR_YAW
    front_direction = (-math.sqrt(.5), -math.sqrt(.5), 0.)
    side_direction = (math.sqrt(.5), -math.sqrt(.5), 0.)
    # Rz(-45deg) * Rx(90deg): tube axis follows the incubator's front normal.
    tube_orientation = (math.cos(INCUBATOR_YAW/2)*math.sqrt(.5), math.cos(INCUBATOR_YAW/2)*math.sqrt(.5),
                        math.sin(INCUBATOR_YAW/2)*math.sqrt(.5), math.sin(INCUBATOR_YAW/2)*math.sqrt(.5))
    entry_clearance = .14
    park_position = (.50, -.30, .46)
    orientation_step = .006

    def reset(self):
        super().reset()
        self.inspect = False

    def __call__(self, observations):
        if self.stage != "clearance":
            previous = self.stage
            actions = super().__call__(observations)
            if previous == "open_door" and self.stage == "approach":
                self.stage, self.ticks = "clearance", 0
            return actions
        # Approach around the FRONT of the open doors, not straight through
        # their sweep from the parked arm. Keep all real collision geometry.
        self.ticks += 1
        self.update_doors(False)
        robot, env = self.robot, self.env
        hand = robot.data.body_pose_w[:1, self.hand]
        goal = hand.new_tensor([[.50, -.30, .46]]) + env.scene.env_origins[:1]
        delta = goal - self.target
        self.target += delta * (.0012 / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(.0012))
        self.lula.set_robot_base_pose(robot.data.root_pos_w[0].cpu().numpy(), robot.data.root_quat_w[0].cpu().numpy())
        solution, feasible = self.lula.compute_inverse_kinematics("panda_hand", self.target[0].cpu().numpy(),
            self.orientation[0].cpu().numpy(), warm_start=robot.data.joint_pos[0, :7].cpu().numpy())
        actions = torch.zeros(env.action_space.shape, device=env.device)
        joint_pos = torch.as_tensor(solution.copy(), device=env.device).reshape(1, 7) if feasible else robot.data.joint_pos[:1, :7]
        actions[:, :7] = (joint_pos - robot.data.default_joint_pos[:1, :7]) / .5
        actions[:, 7] = 1.
        if float(torch.linalg.vector_norm(hand[:, :3] - goal)) < .005 and float(math_utils.quat_error_magnitude(hand[:, 3:7], self.orientation)) < .05:
            self.stage, self.ticks = "approach", 0
        elif self.ticks > 1400:
            self.stage = "failed"
        return actions

    def door_values(self):
        door = self.env.scene["task_incubator"]
        outer = door.joint_names.index("Gelenk_Aussentuer")
        inner = door.joint_names.index("Gelenk_Glastuer")
        return door, [outer, inner], door.data.joint_pos[0, [outer, inner]]

    def update_doors(self, closed):
        door, ids, values = self.door_values()
        outer, inner = (float(v) for v in values)
        # Outer first when opening, inner first when closing: never force the
        # two hinged panels through each other. Inspect opens only the opaque
        # outer door after completion; the glass door remains shut.
        if closed:
            targets = (-math.radians(110.) if abs(inner) > .025 or (self.stage == "done" and self.inspect) else 0., 0.)
        else:
            targets = (-math.radians(110.), -math.radians(100.) if self.stage != "open_door" or outer < -1.8 else 0.)
        door.set_joint_position_target(values.new_tensor([targets]), joint_ids=ids)

    def doors_open(self):
        _, _, values = self.door_values()
        return float(values[0]) < -1.85 and float(values[1]) < -1.65

    def doors_closed(self):
        # The generated outer panel rests against the real inner-door frame
        # about two degrees before its nominal zero, not through the frame.
        values = self.door_values()[2].abs()
        return float(values[0]) < .04 and float(values[1]) < .025
