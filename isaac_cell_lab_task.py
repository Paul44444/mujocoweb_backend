"""Additional cell-culture laboratory; owner's generated assets stay untouched.

The lab is translated so the table mounting point becomes (0,0,0), retaining
the original Cube Lift observation/action coordinates and table-height rewards.
Scenery props are static in this first version; the task cube is dynamic.
"""
import importlib.util
import os
from pathlib import Path
import sys
from copy import deepcopy

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg_PLAY
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg

MODELS = Path(os.environ.get("CELL_LAB_MODELS", "/home/paul/Desktop/3Dmodels"))
SOURCE_CONFIG = MODELS / "laborszene/laborszene_cfg.py"
MOUNT = (.1, -.52, .9)


def spawn_static_prop(prim_path, cfg, translation=None, orientation=None):
    from pxr import Usd, UsdPhysics, PhysxSchema
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation)
    for child in Usd.PrimRange(prim):
        for schema in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, PhysxSchema.PhysxRigidBodyAPI):
            if child.HasAPI(schema):
                child.RemoveAPI(schema)
    from isaac_cell_lab_render import split_material_meshes
    split_material_meshes(prim)
    return prim


def spawn_fixed_furniture(prim_path, cfg, translation=None, orientation=None):
    """IsaacLab's fix_root_link helper assumes root prim == rigid body.

    These Blender exports have an articulation *group* above the bodies, so
    identify the parentless body and author its world mount explicitly.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation)
    bodies = {p.GetPath(): p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    children = set()
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdPhysics.Joint):
            children.update(UsdPhysics.Joint(p).GetBody1Rel().GetTargets())
    roots = [p for path, p in bodies.items() if path not in children]
    if len(roots) != 1:
        raise ValueError(f"Expected one furniture base at {prim_path}, got {len(roots)}")
    body = roots[0]
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(body)
    joint = UsdPhysics.FixedJoint.Define(prim.GetStage(), prim.GetPath().AppendChild("WorldMount"))
    joint.CreateBody1Rel().SetTargets([body.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(transform.ExtractTranslation()))
    quat = transform.ExtractRotationQuat()
    joint.CreateLocalRot0Attr(Gf.Quatf(quat.GetReal(), Gf.Vec3f(quat.GetImaginary())))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.))
    from isaac_cell_lab_render import split_material_meshes
    split_material_meshes(prim)
    return prim


def load_layout():
    if not SOURCE_CONFIG.is_file():
        raise FileNotFoundError(f"Cell laboratory assets missing: {SOURCE_CONFIG}; set CELL_LAB_MODELS")
    spec = importlib.util.spec_from_file_location("mujocoweb_cell_lab_layout", SOURCE_CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LaborszeneCfg(num_envs=1, env_spacing=8.)


@configclass
class FrankaCellLabEnvCfg_PLAY(FrankaCubeLiftEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.
        self.scene.table = None
        self.scene.plane = None
        self.scene.light = None
        self.commands.object_pose.debug_vis = False
        # IsaacLab normally disables translucent surfaces: retain the lab's
        # glass/plastic walls instead of rendering only caps and labels.
        self.sim.render.enable_translucency = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_global_illumination = True
        self.sim.render.antialiasing_mode = "DLAA"
        self.episode_length_s = 120.
        layout = load_layout()
        for name, original in vars(layout).items():
            if not isinstance(original, (ArticulationCfg, RigidObjectCfg, AssetBaseCfg)):
                continue
            asset = deepcopy(original)
            if name.startswith("licht_"):
                setattr(self.scene, "cell_" + name, asset)
                continue
            if name == "mikroskop_01":
                # Same bench, but clear of the front robot/cube workspace.
                asset.init_state.pos = (.58, -.22, .9)
            asset.init_state.pos = tuple(p - m for p, m in zip(asset.init_state.pos, MOUNT))
            if isinstance(asset, RigidObjectCfg):
                # Retain actual collision geometry, but don't let background
                # bottles/stacks fall over before manipulation tasks exist.
                asset.spawn.rigid_props = None
                asset.spawn.func = spawn_static_prop
                asset = AssetBaseCfg(prim_path=asset.prim_path, spawn=asset.spawn,
                    init_state=AssetBaseCfg.InitialStateCfg(pos=asset.init_state.pos, rot=asset.init_state.rot))
            if isinstance(asset, ArticulationCfg):
                asset.spawn.articulation_props.fix_root_link = None
                asset.spawn.func = spawn_fixed_furniture
                for actuator in asset.actuators.values():
                    actuator.stiffness = 200.
                    actuator.damping = 20.
            setattr(self.scene, "cell_" + name, asset)
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        for name in ("panda_shoulder", "panda_forearm"):
            self.scene.robot.actuators[name].stiffness = 400.
            self.scene.robot.actuators[name].damping = 80.
        # Clear strip along the front of the table, away from the microscope.
        self.scene.object.init_state.pos = (.43, -.10, .055)
        self.events.reset_object_position.params["pose_range"] = {axis: (0., 0.) for axis in ("x", "y", "z", "yaw")}


gym.register(id="Isaac-Franka-Cell-Lab-Play-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": FrankaCellLabEnvCfg_PLAY,
                                    "rsl_rl_cfg_entry_point": LiftCubePPORunnerCfg})
