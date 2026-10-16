"""Additional laboratory task: five-tube rack -> owner's two-door incubator."""
from copy import deepcopy
import math
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaac_cell_lab_task import FrankaCellLabEnvCfg_PLAY, spawn_fixed_furniture, MOUNT
from isaac_rack_five_task import FrankaFiveTubeRackEnvCfg_PLAY
from isaac_rack_five_layout import HOLES_LOCAL, OCCUPIED_SLOTS, TUBE_NAMES
from isaac_rack_insert_task import SEATED_CENTER_Z, RackInsertPPORunnerCfg

INCUBATOR_POSITION = (.49, .43, .14)
INCUBATOR_SCALE = .5
INCUBATOR_YAW = -math.pi / 4
# Middle shelf: source center .48 m, thickness .003 m, scaled by .5.
SHELF_TOP = INCUBATOR_POSITION[2] + .4815 * INCUBATOR_SCALE
DEPOSIT_CENTER = (INCUBATOR_POSITION[0] - .07 * math.sqrt(.5),
                  INCUBATOR_POSITION[1] - .07 * math.sqrt(.5), SHELF_TOP + .045)
RACK_POSITION = (.40, -.08, 0.)


def spawn_task_incubator(prim_path, cfg, translation=None, orientation=None):
    import os
    from pxr import Usd, UsdPhysics, PhysxSchema
    prim = spawn_fixed_furniture(prim_path, cfg, translation, orientation)
    for child in Usd.PrimRange(prim):
        if os.environ.get("LAB_INCUBATOR_CONTACT_DEBUG") == "1" and child.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(child).CreateThresholdAttr(0.)
        if child.HasAPI(UsdPhysics.CollisionAPI):
            collision = PhysxSchema.PhysxCollisionAPI.Apply(child)
            collision.CreateContactOffsetAttr(.0005)
            collision.CreateRestOffsetAttr(0.)
    return prim


@configclass
class FrankaCellLabIncubatorEnvCfg_PLAY(FrankaCellLabEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 180.
        # Keep both full-size wall incubators, and add a benchtop working copy.
        self.scene.task_incubator = deepcopy(self.scene.cell_co2_inkubator_unten)
        incubator = self.scene.task_incubator
        incubator.prim_path = "{ENV_REGEX_NS}/TaskIncubator"
        incubator.init_state.pos = INCUBATOR_POSITION
        incubator.init_state.rot = (math.cos(INCUBATOR_YAW / 2), 0., 0., math.sin(INCUBATOR_YAW / 2))
        incubator.spawn.scale = (INCUBATOR_SCALE,) * 3
        incubator.spawn.func = spawn_task_incubator
        for actuator in incubator.actuators.values():
            actuator.stiffness = 30.
            actuator.damping = 8.
            actuator.effort_limit_sim = 40.
            actuator.velocity_limit_sim = 1.
        # A real open riser keeps the door sweep above the rack and the back
        # of the incubator clear of the bench's electrical installation rail.
        for name, position, size in [
            ("top", (.49, .43, .134), (.42, .42, .012)),
            *[(f"leg_{i}", (x, y, .064), (.025, .025, .128))
              for i, (x, y) in enumerate(( (.35,.30), (.63,.30), (.35,.49), (.63,.49) ))],
        ]:
            setattr(self.scene, "incubator_riser_" + name, AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/IncubatorRiser_" + name,
                init_state=AssetBaseCfg.InitialStateCfg(pos=position),
                spawn=sim_utils.CuboidCfg(size=size,
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.62,.66,.68), metallic=.5, roughness=.3))))
        # An extra table holds the microscope, not an invisible obstacle on the
        # robot's pickup/insertion route. All original source assets stay intact.
        self.scene.cell_microscope_table = deepcopy(self.scene.cell_labortisch_mitte)
        self.scene.cell_microscope_table.prim_path = "{ENV_REGEX_NS}/MicroscopeTable"
        self.scene.cell_microscope_table.init_state.pos = tuple(p - m for p, m in zip((-1.65, -.3, 0.), MOUNT))
        self.scene.cell_mikroskop_01.init_state.pos = tuple(p - m for p, m in zip((-1.4, -.25, .9), MOUNT))
        source = FrankaFiveTubeRackEnvCfg_PLAY()
        self.scene.rack_visual = deepcopy(source.scene.rack_visual)
        self.scene.rack_visual.init_state.pos = RACK_POSITION
        for index, (name, slot) in enumerate(zip(TUBE_NAMES, OCCUPIED_SLOTS)):
            tube = deepcopy(getattr(source.scene, name))
            x, y = HOLES_LOCAL[slot]
            tube.init_state.pos = (RACK_POSITION[0] + x, RACK_POSITION[1] + y, SEATED_CENTER_Z)
            setattr(self.scene, name, tube)
            visual_name = "tube_visual" if index == 0 else name + "_visual"
            setattr(self.scene, visual_name, deepcopy(getattr(source.scene, visual_name)))
        self.scene.robot.init_state.joint_pos["panda_joint1"] = -1.3
        # This reference is not a cube-learning objective.
        for term in vars(self.rewards).values():
            if hasattr(term, "weight"):
                term.weight = 0.


gym.register(id="Isaac-Franka-Cell-Lab-Incubator-Play-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": FrankaCellLabIncubatorEnvCfg_PLAY,
                                    "rsl_rl_cfg_entry_point": RackInsertPPORunnerCfg})
