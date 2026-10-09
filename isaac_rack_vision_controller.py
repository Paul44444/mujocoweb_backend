"""Image-only rack control; robot joint/body feedback remains proprioception."""
import torch
from isaac_rack_controller import RackInsertionController
from isaac_rack_perception import MarkerPerception

class VisionRackController(RackInsertionController):
    def __init__(self, env):
        self.perception = MarkerPerception(env)
        super().__init__(env)

    def reset(self):
        super().reset()
        self.perception.regions = {}
        self.perception.calibrations = {}
        self.perception.rack_destination = None

    def object_pose(self):
        return self.measured_pose

    def destination(self):
        return self.measured_destination

    def __call__(self, observations):
        try:
            self.measured_pose, self.measured_destination = self.perception.estimate()
        except (ValueError, KeyError) as error:
            self.perception.last_error = str(error)
            # Never fall back to simulator truth or open a loaded gripper.
            actions = torch.zeros(self.env.action_space.shape, device=self.env.device)
            actions[:, :7] = (self.robot.data.joint_pos[:1, :7] - self.robot.data.default_joint_pos[:1, :7]) / .5
            actions[:, 7] = -1. if self.stage in {"close", "lift", "transfer", "lower"} else 1.
            return actions
        return super().__call__(observations)
