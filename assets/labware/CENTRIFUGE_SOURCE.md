# Owner-supplied centrifuge

Source: 3DBicio, “FREE - SCI-FI LAB CENTRIFUGE 4”, Cults3D:
https://cults3d.com/en/3d-model/game/free-sci-fi-lab-centrifuge-4-medical-mixer-prop-stl-obj-4k-pbr-comme

Owner downloaded ZIP `0705345858d6174052d418b773c64ee8.zip` containing one
OBJ (1,500,000 triangles), MTL and four PNG textures. No separate lid/rotor,
animation or articulation data are supplied. Do not describe this as a working
centrifuge or a robot-loadable rotor: it is currently static scene equipment.

`build_centrifuge_asset.py` scales it to 32 cm high, decimates to approximately
45,000 faces, reduces textures to 1024 px and exports USD. Generated third-party
files are deliberately not redistributed in this public repository. Run the
builder with Blender on the simulation host and your own downloaded archive.
The listing describes digital commercial use as allowed; the archive contains
no separate license document. This asset is not used for learning data.

Runtime `isaac_centrifuge_asset.py` adds static, non-convex triangle collisions.
There is no lid joint or spinning rotor. Future functional parts require a
separate geometry/physics design, not just an animation flag.
