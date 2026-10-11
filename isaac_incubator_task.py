"""Separate contact-based incubator loading scene; preserves existing tasks."""
import gymnasium as gym
from isaaclab.utils import configclass
from isaac_rack_five_task import FrankaFiveTubeRackEnvCfg_PLAY
from isaac_rack_insert_task import RackInsertPPORunnerCfg

INCUBATOR_POSITION = (.48, .43, 0.)
# Release above the lower shelf, visible through the door's window (which
# starts at 237.2 mm). Keep the wide wrist outside the shelf opening.
DEPOSIT_CENTER = (.48, .375, .332)
FLOOR_Z = .2856


@configclass
class FrankaIncubatorEnvCfg_PLAY(FrankaFiveTubeRackEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.incubator.init_state.pos = INCUBATOR_POSITION
        # One free-standing tube, not five tubes constrained inside rack holes.
        self.scene.object.init_state.pos = (.43, -.16, .047)
        for index in range(2, 6):
            setattr(self.scene, f"rack_tube_{index}", None)
            setattr(self.scene, f"rack_tube_{index}_visual", None)
        self.scene.rack_visual = None
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        for name in ("panda_shoulder", "panda_forearm"):
            self.scene.robot.actuators[name].stiffness = 400.
            self.scene.robot.actuators[name].damping = 80.
        self.episode_length_s = 180.


gym.register(id="Isaac-Franka-Tube-Incubator-Play-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": FrankaIncubatorEnvCfg_PLAY,
                                    "rsl_rl_cfg_entry_point": RackInsertPPORunnerCfg})
