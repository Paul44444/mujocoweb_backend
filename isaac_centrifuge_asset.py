"""Static owner's centrifuge mesh, with exact triangle-mesh surface collision."""
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaac_labware_task import LABWARE_ASSET_DIR

CENTRIFUGE_ASSET = LABWARE_ASSET_DIR / "centrifuge_scifi.usdc"
CENTRIFUGE_POSITION = (.63, -.43, 0.)


def spawn_centrifuge(prim_path, cfg, translation=None, orientation=None):
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation)
    # No rigid-body ancestor: concave triangle meshes are valid static shapes.
    # Do not replace this irregular model with a single convex bounding box.
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr("none")
            collision = PhysxSchema.PhysxCollisionAPI.Apply(child)
            collision.CreateContactOffsetAttr(.0005)
            collision.CreateRestOffsetAttr(0.)
    return prim


def centrifuge_cfg():
    if not CENTRIFUGE_ASSET.is_file():
        print("Optional local centrifuge asset missing; build_centrifuge_asset.py converts the owner's download", flush=True)
        return None
    return AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Centrifuge",
        init_state=AssetBaseCfg.InitialStateCfg(pos=CENTRIFUGE_POSITION),
        spawn=sim_utils.UsdFileCfg(usd_path=str(CENTRIFUGE_ASSET), func=spawn_centrifuge))
