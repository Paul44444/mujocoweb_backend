"""Generate genuine pretrained GraspGenX candidates for our tube collider.

Run with the isolated GraspGenX interpreter, not the Isaac Lab environment.
This is grasp inference, not a trained arm-motion or reinforcement policy.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import trimesh
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    # Exact current physical collider, centered at the rigid-body origin.
    mesh_path = args.output_directory / "tube_collider.obj"
    trimesh.creation.cylinder(radius=0.0075, height=0.092, sections=64).export(mesh_path)
    output = args.output_directory / "predicted_grasps.yaml"
    if not args.reuse_predictions:
        subprocess.run([
        sys.executable, str(args.repository / "scripts/demo_object_mesh.py"),
        "--mesh_file", str(mesh_path), "--gripper_name", "franka_panda",
        "--num_grasps", "256", "--return_topk", "--topk_num_grasps", "100",
        "--grasp_threshold", "-1", "--no-visualization", "--output_file", str(output),
        ], cwd=args.repository, check=True)
    with output.open() as stream:
        predictions = yaml.safe_load(stream)["grasps"]
    if not predictions:
        raise RuntimeError("GraspGenX returned no grasp candidates")
    gripper_mesh = trimesh.load(args.repository / "ext/gripper_descriptions/gripper_descriptions/assets/x_grippers/franka_panda/coll_mesh.obj")
    for grasp in predictions.values():
        rotation = trimesh.transformations.quaternion_matrix([grasp["orientation"]["w"], *grasp["orientation"]["xyz"]])[:3, :3]
        points = gripper_mesh.vertices @ rotation.T + grasp["position"]
        grasp["gripper_lowest_z_m"] = float(points[:, 2].min())
        low, high = points.min(axis=0), points.max(axis=0)
        grasp["gripper_bounds_corners"] = [[float(x), float(y), float(z)] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
    result = {"model": "NVIDIA GraspGenX", "gripper": "franka_panda",
              "source": "https://github.com/NVlabs/GraspGenX",
              "checkpoint_source": "https://huggingface.co/adithyamurali/GraspGenXModel",
              "geometry": {"radius_m": 0.0075, "height_m": 0.092},
              "grasps": sorted(predictions.values(), key=lambda grasp: grasp["confidence"], reverse=True)}
    (args.output_directory / "tube_grasps.json").write_text(json.dumps(result, indent=2))
    print(f"Saved {len(predictions)} pretrained grasp candidates to {args.output_directory}")


if __name__ == "__main__":
    main()
