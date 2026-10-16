# Additional cell-culture laboratory

Task key: `cell_lab`; Gym ID: `Isaac-Franka-Cell-Lab-Play-v0`.
The website lists it as **Cell culture laboratory · Cube Lift**.
Existing environments, checkpoints, demos and camera presets are unchanged.

## Asset source

Owner's generated models and textures: `/home/paul/Desktop/3Dmodels`.
Set `CELL_LAB_MODELS` on both API and viewer to use another installation.
`laborszene/laborszene_cfg.py` is imported, not edited or rebuilt. Its relative
USD/texture references continue to resolve in the original model tree. The
large asset library is not duplicated into this public Git repository.

The complete layout is translated by `(-0.1,+0.52,-0.9)` so the original robot
mount `(0.1,-0.52,0.9)` becomes `(0,0,0)`. Cube Lift's existing relative robot
observations and table-height reward thresholds therefore retain their meaning.

## Initial task and limits

The Franka physically grips and lifts a cube on the front workbench strip.
The microscope is moved slightly rear/right on the same bench to clear the
gripper's workspace. Mixed-opacity Blender meshes are split by material in
the composed Isaac stage (`isaac_cell_lab_render.py`), preserving geometry,
UVs and original source files while avoiding missing RTX surfaces.
The default reference uses simulator positions and Lula IK, **not neural
inference**, image-based perception or a pretrained laboratory policy. No
object attachment/teleportation is used for grasping. Reset repeats the task.
The GUI has **Lab overview** and **Workbench** camera buttons; orbit, pan,
zoom and pause remain available.

Ten furniture articulations have fixed bases and damped position drives.
Background bottles, racks and other loose props keep their detailed collision
geometry but are static for this first version. They are not yet independent
manipulation targets. This version is playback/teleoperation only; use the
existing Cube Lift or tube tasks for PPO/BC training.

## Validation

Run with IsaacLab's Python:

```bash
/home/paul/miniconda3/envs/env_isaaclab1/bin/python test_cell_lab.py \
  --headless --device cuda:0 --validation-output assets/cell_lab_reference.json
```

Check for `CELL_LAB_PASS`, not merely process exit code (Isaac's shutdown can
mask Python failure exit codes). Test checks real cube pickup/held height,
finite rewards, no unintended resets, and fixed furniture base positions.
The API exposes the reference only with a successful validation artifact.
