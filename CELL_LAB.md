# Additional cell-culture laboratory

## Tube → Incubator variant

Additional key `cell_lab_incubator`, Gym ID
`Isaac-Franka-Cell-Lab-Incubator-Play-v0`; website entry
**Cell culture laboratory · Tube → Incubator**. The cube variant stays intact.
This version adds a 0.5-scale working copy of the generated two-door incubator
at `(.49,.43,.14)`, yaw -45 degrees facing the robot, on a collidable open
riser above the rack/door sweep. The
microscope moves to another copy of the generated lab table on the left.
The two original wall incubators and the remaining laboratory stay present.

The proven five-tube rack and physically grippable tube assets are reused from
the older task. The cyan tube is lifted, rotated, placed on the middle shelf
and released; the other four tubes remain seated. Outer door opens first,
inner glass door second; closure happens in reverse order. **Inspect incubator**
opens only the outer door after completion to see the tube behind the glass.
This is simulator-state IK, not a new learned policy or camera estimator.
The middle shelf avoids the tightly folded near-base arm posture required by
the bottom shelf. The generated outer door settles about 1.94 degrees from
nominal zero; completion allows 0.04 radians outside / 0.025 radians inside
instead of forcing the physical panels through their contact stop.

Validated on 2026-10-16: real pickup, deposition at Z=0.38825 m, six seconds
of stability, four neighbours preserved, inspection open/close and full reset.
The committed reference JSON is produced from this successful regression.

Run `test_cell_lab_incubator.py --headless --device cuda:0 --validation-output
assets/cell_lab_incubator_reference.json` with IsaacLab Python. Validation
checks physical pickup, stable deposition, closed doors, fixed furniture and
undisturbed neighbouring tubes. Optional `--enable_cameras --diagnostic-image
/tmp/lab-incubator.png` captures diagnostic frames. Source USDs are not edited.
`--contact-debug` additionally logs exact PhysX collider pairs for diagnosing
blocked doors or gripper contacts; it is off in the normal web worker.

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
