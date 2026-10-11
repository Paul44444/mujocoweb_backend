"""Run with Blender --background --python this_file -- source.zip output.usdc.

Owner-supplied Cults OBJ -> scaled, decimated, textured USD. Original archive
is never modified. Keep generated third-party files local, outside Git.
"""
import argparse
from pathlib import Path
import sys
import tempfile
import zipfile
import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--preview", type=Path)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
with tempfile.TemporaryDirectory(prefix="mujocoweb-centrifuge-") as temporary:
    folder = Path(temporary)
    with zipfile.ZipFile(args.source) as archive:
        # Only flat OBJ, MTL and PNG files; never extract arbitrary paths.
        for name in archive.namelist():
            if Path(name).name != name or Path(name).suffix.lower() not in {".obj", ".mtl", ".png"}:
                raise ValueError(f"Unexpected archive member: {name}")
            (folder / name).write_bytes(archive.read(name))
    bpy.ops.wm.obj_import(filepath=str(next(folder.glob("*.obj"))), forward_axis="NEGATIVE_Z", up_axis="Y")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    assert len(meshes) == 1, "Inspect multipart sources manually before converting"
    obj = meshes[0]
    obj.name = "Centrifuge"
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    low = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    high = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    factor = .32 / (high.z - low.z)
    for vertex in obj.data.vertices:
        vertex.co = (vertex.co - Vector(((low.x + high.x) / 2, (low.y + high.y) / 2, low.z))) * factor
    obj.location = (0., 0., 0.)
    before = len(obj.data.polygons)
    modifier = obj.modifiers.new("Realtime mesh", "DECIMATE")
    modifier.ratio = min(1., 45000. / before)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    # OBJ importer supplies base colour. Explicitly wire missing PBR channels.
    material = obj.data.materials[0]
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    for suffix, socket in (("metallic", "Metallic"), ("roughness", "Roughness")):
        node = nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(str(folder / f"texture_pbr_20250901_{suffix}.png"), check_existing=True)
        node.image.colorspace_settings.name = "Non-Color"
        links.new(node.outputs["Color"], shader.inputs[socket])
    # Imported normals may have an unsuitable bump node; export a normal map.
    normal = nodes.new("ShaderNodeTexImage")
    normal.image = bpy.data.images.load(str(folder / "texture_pbr_20250901_normal.png"), check_existing=True)
    normal.image.colorspace_settings.name = "Non-Color"
    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(normal.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    textures = args.output.parent / "centrifuge_scifi_textures"
    textures.mkdir(exist_ok=True)
    for image in bpy.data.images:
        if image.source == "FILE" and image.filepath:
            image.scale(1024, 1024)
            image.filepath_raw = str(textures / Path(image.filepath).name)
            image.file_format = "PNG"
            image.save()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.wm.usd_export(filepath=str(args.output), selected_objects_only=True,
        export_materials=True, export_textures_mode="NEW", relative_paths=True)
    print("CENTRIFUGE_ASSET", before, len(obj.data.polygons), "height_m", .32, flush=True)
    if args.preview:
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 16
        scene.render.resolution_x = 640
        scene.render.resolution_y = 640
        scene.render.resolution_percentage = 100
        scene.world.color = (.35, .35, .35)
        bpy.ops.object.camera_add(location=(.55, -.65, .50))
        camera = bpy.context.object
        camera.rotation_euler = (Vector((0., 0., .16)) - camera.location).to_track_quat("-Z", "Y").to_euler()
        camera.data.lens = 55
        scene.camera = camera
        bpy.ops.object.light_add(type="AREA", location=(.3, -.4, .8))
        bpy.context.object.data.energy = 80
        bpy.context.object.data.shape = "DISK"
        bpy.context.object.data.size = .5
        scene.render.filepath = str(args.preview)
        bpy.ops.render.render(write_still=True)
