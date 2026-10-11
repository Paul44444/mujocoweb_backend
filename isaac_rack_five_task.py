"""Separate five-tube scene; old single-tube tasks and policies are unchanged."""
from copy import deepcopy
import gymnasium as gym
from isaaclab.utils import configclass
from isaac_rack_insert_task import FrankaRackInsertEnvCfg_PLAY, RackInsertPPORunnerCfg, RACK_POSITION, SEATED_CENTER_Z
from isaac_rack_five_layout import HOLES_LOCAL, OCCUPIED_SLOTS, TUBE_NAMES


@configclass
class FrankaFiveTubeRackEnvCfg_PLAY(FrankaRackInsertEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.debug_vis = False
        self.episode_length_s = 120.
        tube_template = deepcopy(self.scene.object)
        visual_template = deepcopy(self.scene.tube_visual)
        colors = ((.02, .72, 1.), (.10, .85, .55), (.55, .35, 1.), (1., .65, .15), (.95, .25, .55))
        for index, (name, slot, color) in enumerate(zip(TUBE_NAMES, OCCUPIED_SLOTS, colors)):
            tube = deepcopy(tube_template)
            path = "Object" if index == 0 else f"RackTube{index + 1}"
            tube.prim_path = "{ENV_REGEX_NS}/" + path
            x, y = HOLES_LOCAL[slot]
            # Start seated on the lower shelf, not overlapping the upper rim.
            tube.init_state.pos = (RACK_POSITION[0] + x, RACK_POSITION[1] + y, SEATED_CENTER_Z)
            tube.init_state.rot = (1., 0., 0., 0.)
            tube.spawn.visual_material.diffuse_color = color
            tube.spawn.visual_material.emissive_color = (0., 0., 0.)
            setattr(self.scene, name, tube)
            visual = deepcopy(visual_template)
            visual.prim_path = "{ENV_REGEX_NS}/" + path + "/TubeVisual"
            visual.spawn.visual_material.diffuse_color = color
            visual.spawn.visual_material.emissive_color = (0., 0., 0.)
            setattr(self.scene, "tube_visual" if index == 0 else name + "_visual", visual)
        self.events.reset_object_position.params["pose_range"] = {axis: (0., 0.) for axis in ("x", "y", "z", "yaw")}
        # This is a scene preview, not an insertion-learning objective.
        for name in ("reaching_object", "lifting_object", "rack_alignment", "rack_insertion", "rack_released", "action_rate", "joint_vel"):
            term = getattr(self.rewards, name, None)
            if term is not None:
                term.weight = 0.


gym.register(id="Isaac-Franka-Five-Tube-Rack-Play-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": FrankaFiveTubeRackEnvCfg_PLAY,
                                    "rsl_rl_cfg_entry_point": RackInsertPPORunnerCfg})
