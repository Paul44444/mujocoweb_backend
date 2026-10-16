"""Separate Blender material subsets for reliable RTX mixed-opacity rendering.

Composed-stage overrides only: original USDs/Blender files are never changed.
Rigid shapes and their invisible collider siblings retain their original pose.
"""
from pxr import Usd, UsdGeom, UsdShade, Gf


def split_material_meshes(root):
    stage = root.GetStage()
    for prim in list(Usd.PrimRange(root)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        subsets = [UsdGeom.Subset(p) for p in prim.GetChildren() if p.IsA(UsdGeom.Subset)]
        subsets = [s for s in subsets if s.GetFamilyNameAttr().Get() == "materialBind"]
        if len(subsets) < 2:
            continue
        counts = list(mesh.GetFaceVertexCountsAttr().Get())
        indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        group = UsdGeom.Xform.Define(stage, prim.GetPath().GetParentPath().AppendChild(prim.GetName() + "_RenderParts"))
        group.AddTransformOp().Set(UsdGeom.Xformable(prim).GetLocalTransformation())
        for serial, subset in enumerate(subsets):
            faces = list(subset.GetIndicesAttr().Get())
            corners = [i for face in faces for i in range(offsets[face], offsets[face + 1])]
            part = UsdGeom.Mesh.Define(stage, group.GetPath().AppendChild(f"Material_{serial}"))
            part.CreatePointsAttr(mesh.GetPointsAttr().Get())
            part.CreateFaceVertexCountsAttr([counts[f] for f in faces])
            part.CreateFaceVertexIndicesAttr([indices[i] for i in corners])
            part.CreateSubdivisionSchemeAttr(mesh.GetSubdivisionSchemeAttr().Get())
            part.CreateOrientationAttr(mesh.GetOrientationAttr().Get())
            part.CreateDoubleSidedAttr(True)
            def select(values, interpolation):
                if interpolation == "faceVarying":
                    return [values[i] for i in corners]
                if interpolation == "uniform":
                    return [values[i] for i in faces]
                return values
            normals = mesh.GetNormalsAttr().Get()
            if normals is not None and len(normals):
                interpolation = mesh.GetNormalsInterpolation()
                part.CreateNormalsAttr(select(normals, interpolation))
                part.SetNormalsInterpolation(interpolation)
            for variable in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
                values = variable.GetAttr().Get()
                if values is None:
                    continue
                interpolation = variable.GetInterpolation()
                target = UsdGeom.PrimvarsAPI(part).CreatePrimvar(variable.GetPrimvarName(),
                    variable.GetTypeName(), interpolation, variable.GetElementSize())
                if variable.IsIndexed():
                    target.Set(values)
                    target.SetIndices(select(variable.GetIndices(), interpolation))
                else:
                    target.Set(select(values, interpolation))
            material, _ = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()
            if material:
                UsdShade.MaterialBindingAPI.Apply(part.GetPrim()).Bind(material)
        mesh.CreateVisibilityAttr("invisible")
