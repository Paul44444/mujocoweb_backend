"""Convert the downloaded Polyfork GLB to a small articulated Isaac USD.

Pure Python converter for this non-indexed GLB: Y-up -> Z-up, scale 0.4.
Visual meshes retain vertex colours and the original door pivot. The body has
compound box colliders around its cavity, NEVER a solid bounding-box collider.
Source cultures are omitted to leave the shelves available for manipulation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

SCALE = .4


def convert(source, destination):
    raw = source.read_bytes()
    assert raw[:4] == b"glTF" and struct.unpack_from("<I", raw, 4)[0] == 2
    size = struct.unpack_from("<I", raw, 12)[0]
    data = json.loads(raw[20:20 + size])
    binary = raw[28 + size:]
    def accessor(index):
        item = data["accessors"][index]
        view = data["bufferViews"][item["bufferView"]]
        assert item["componentType"] == 5126 and item["type"] == "VEC3" and view.get("byteStride", 12) == 12
        return np.frombuffer(binary, dtype="<f4", count=item["count"] * 3,
            offset=view.get("byteOffset", 0) + item.get("byteOffset", 0)).reshape(-1, 3)
    def z_up(points, scale=SCALE):
        return np.column_stack((points[:, 0], -points[:, 2], points[:, 1])) * scale
    def vector(p):
        return "(" + ", ".join(f"{float(v):.8g}" for v in p) + ")"
    lines = ['#usda 1.0', '(', '    defaultPrim = "Incubator"', '    metersPerUnit = 1', '    upAxis = "Z"',
        '    customLayerData = {', '        string source = "https://polyfork.dev/asset/co2-incubator-198c16"',
        f'        string sourceSha256 = "{hashlib.sha256(raw).hexdigest()}"', '    }', ')',
        'def Xform "Incubator" (prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"])', '{',
        '    bool physxArticulation:enabledSelfCollisions = false',
        '    def Scope "Looks" {', '        def Material "Paint" {',
        '            token outputs:surface.connect = </Incubator/Looks/Paint/Surface.outputs:surface>',
        '            def Shader "Color" {', '                uniform token info:id = "UsdPrimvarReader_float3"',
        '                string inputs:varname = "displayColor"', '                float3 outputs:result', '            }',
        '            def Shader "Surface" {', '                uniform token info:id = "UsdPreviewSurface"',
        '                color3f inputs:diffuseColor.connect = </Incubator/Looks/Paint/Color.outputs:result>',
        '                float inputs:roughness = 0.45', '                token outputs:surface', '            }', '        }',
        '        def Material "Glass" {', '            token outputs:surface.connect = </Incubator/Looks/Glass/Surface.outputs:surface>',
        '            def Shader "Surface" {', '                uniform token info:id = "UsdPreviewSurface"',
        '                color3f inputs:diffuseColor = (0.56, 0.79, 0.85)', '                float inputs:opacity = 0.25',
        '                float inputs:roughness = 0.15', '                token outputs:surface', '            }', '        }', '    }']
    def rigid(name, mass, translation):
        lines.extend([f'    def Xform "{name}" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]) {{',
            '        bool physics:rigidBodyEnabled = true', f'        float physics:mass = {mass}',
            '        bool physxRigidBody:disableGravity = true', '        int physxRigidBody:solverPositionIterationCount = 16',
            f'        double3 xformOp:translate = {vector(translation)}', '        uniform token[] xformOpOrder = ["xformOp:translate"]'])
    def mesh(node_name):
        node = next(n for n in data["nodes"] if n.get("name") == node_name)
        primitive = data["meshes"][node["mesh"]]["primitives"][0]
        assert primitive.get("mode", 4) == 4 and "indices" not in primitive
        points = z_up(accessor(primitive["attributes"]["POSITION"]))
        colors = accessor(primitive["attributes"]["COLOR_0"])
        normals = z_up(accessor(primitive["attributes"]["NORMAL"]), 1.)
        name = node_name.replace("-", "_")
        material = "Glass" if node_name == "door-glass" else "Paint"
        lines.extend([f'        def Mesh "{name}" (prepend apiSchemas = ["MaterialBindingAPI"]) {{',
            '            uniform token subdivisionScheme = "none"', '            uniform bool doubleSided = true',
            '            int[] faceVertexCounts = [' + ', '.join(['3'] * (len(points) // 3)) + ']',
            '            int[] faceVertexIndices = [' + ', '.join(map(str, range(len(points)))) + ']',
            '            point3f[] points = [' + ', '.join(map(vector, points)) + ']',
            '            normal3f[] normals = [' + ', '.join(map(vector, normals)) + '] (interpolation = "vertex")',
            '            color3f[] primvars:displayColor = [' + ', '.join(map(vector, colors)) + '] (interpolation = "vertex")',
            f'            rel material:binding = </Incubator/Looks/{material}>', '        }'])
    def box(name, low, high):
        low, high = np.array(low), np.array(high)
        lines.extend([f'        def Cube "{name}" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]) {{',
            '            double size = 1', '            token visibility = "invisible"',
            '            bool physics:collisionEnabled = true', '            float physxCollision:contactOffset = 0.0003',
            '            float physxCollision:restOffset = 0',
            f'            double3 xformOp:translate = {vector((low + high) / 2)}',
            f'            double3 xformOp:scale = {vector(high - low)}',
            '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]', '        }'])
    def source_box(name, low, high):
        corners = z_up(np.array([low, high]))
        box(name, corners.min(0), corners.max(0))
    rigid("Body", 10, (0, 0, 0))
    mesh("body")
    source_box("LeftWall", (-.45, .05, -.45), (-.415, 1.275, .345))
    source_box("RightWall", (.415, .05, -.45), (.45, 1.275, .345))
    source_box("BackWall", (-.415, .05, -.45), (.415, 1.275, -.415))
    source_box("Top", (-.415, 1.24, -.415), (.415, 1.275, .345))
    source_box("LowerCabinet", (-.415, .05, -.415), (.415, .418, .30))
    for index, height in enumerate((.703, .953)):
        source_box(f"Shelf{index + 1}", (-.355, height, -.37), (.355, height + .011, .27))
    lines.append('    }')
    door = next(n for n in data["nodes"] if n.get("name") == "door")
    pivot = z_up(np.array([door["matrix"][12:15]]))[0]
    rigid("Door", .15, pivot)
    mesh("door-panel")
    mesh("door-glass")
    # Glass is physically solid; the thin door collider moves with its frame.
    source_box("DoorSurface", (.007, .418, -.034), (.857, 1.263, -.014))
    source_box("Handle", (.77, .72, -.014), (.83, 1.0, .078))
    lines.extend(['    }', '    def PhysicsFixedJoint "BaseFixed" {',
        '        rel physics:body1 = </Incubator/Body>', '    }',
        '    def PhysicsRevoluteJoint "door_hinge" (prepend apiSchemas = ["PhysicsDriveAPI:angular"]) {',
        '        rel physics:body0 = </Incubator/Body>', '        rel physics:body1 = </Incubator/Door>',
        '        uniform token physics:axis = "Z"', f'        point3f physics:localPos0 = {vector(pivot)}',
        '        point3f physics:localPos1 = (0, 0, 0)', '        float physics:lowerLimit = -110',
        '        float physics:upperLimit = 0', '        float drive:angular:physics:stiffness = 8',
        '        float drive:angular:physics:damping = 1.5', '        float drive:angular:physics:maxForce = 2',
        '        float drive:angular:physics:targetPosition = 0', '        uniform token drive:angular:physics:type = "force"',
        '    }', '}'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(lines) + '\n')
    print(f"Built {destination}: hinge {pivot.tolist()}, 0.4-scale empty chamber with shelves")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "assets/labware/co2_incubator_articulated.usda")
    args = parser.parse_args()
    convert(args.source, args.output)
