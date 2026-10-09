"""Bake the rack's visual transforms into a static, non-convex USD collider.

Run with Isaac Lab's Python. No existing asset is changed. A static triangle
mesh preserves the twelve real holes; convex-hull cooking would seal them.
"""
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
try:
    from pathlib import Path
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema

    directory = Path(__file__).resolve().parent / "assets" / "labware"
    source = Usd.Stage.Open(str(directory / "test_tube_rack.usda"))
    destination = Usd.Stage.CreateNew(str(directory / "rack_insert_collision.usda"))
    UsdGeom.SetStageMetersPerUnit(destination, 1)
    UsdGeom.SetStageUpAxis(destination, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(destination, "/Rack")
    destination.SetDefaultPrim(root.GetPrim())
    mesh_count = 0
    for prim in source.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        original = UsdGeom.Mesh(prim)
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        mesh = UsdGeom.Mesh.Define(destination, f"/Rack/Surface{mesh_count}")
        mesh.CreatePointsAttr([Gf.Vec3f(matrix.Transform(p)) for p in original.GetPointsAttr().Get()])
        mesh.CreateFaceVertexCountsAttr(original.GetFaceVertexCountsAttr().Get())
        mesh.CreateFaceVertexIndicesAttr(original.GetFaceVertexIndicesAttr().Get())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        collision.CreateContactOffsetAttr(0.0002)
        collision.CreateRestOffsetAttr(0.0)
        mesh_count += 1
    assert mesh_count > 0, "Rack source contains no meshes"
    destination.GetRootLayer().Save()
    print(f"Built rack insertion collider: {mesh_count} triangle meshes", flush=True)
finally:
    app.close()
