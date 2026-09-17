# info (paul): the examine_env.py

""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import faulthandler
import os
import pickle
import queue
import sys
import time
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager

# These must be set before importing MuJoCo, RoboHive, or PyOpenGL.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

_muj1_import_start = time.perf_counter()

print("muj1.py: import started", flush=True)

faulthandler.enable()


def _memory_usage() -> str:
    """Return Linux process memory counters without adding a dependency."""
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            values = {}
            for line in status_file:
                key, separator, value = line.partition(":")
                if separator and key in {"VmRSS", "VmSize"}:
                    values[key] = value.strip()
        return ", ".join(
            f"{key}={values.get(key, 'unknown')}"
            for key in ("VmRSS", "VmSize")
        )
    except OSError:
        return "memory counters unavailable"


@contextmanager
def _diagnose_stall(label: str, timeout_seconds: int = 30):
    """
    Emit Python thread stacks if a native/import operation stops responding.

    Render can otherwise show only the log line immediately before the stall.
    """
    started = time.perf_counter()
    print(f"CHECKPOINT START: {label} ({_memory_usage()})", flush=True)
    faulthandler.dump_traceback_later(
        timeout_seconds,
        repeat=True,
        file=sys.stderr,
    )
    try:
        yield
    except BaseException as exc:
        print(
            f"CHECKPOINT FAILED: {label}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    else:
        print(
            f"CHECKPOINT DONE: {label} in "
            f"{time.perf_counter() - started:.2f}s "
            f"({_memory_usage()})",
            flush=True,
        )
    finally:
        faulthandler.cancel_dump_traceback_later()


print("muj1.py: importing NumPy...", flush=True)
import numpy as np
print(
    f"muj1.py: NumPy imported after "
    f"{time.perf_counter() - _muj1_import_start:.2f}s",
    flush=True,
)


print("muj1.py: importing OpenCV...", flush=True)
import cv2
print(
    f"muj1.py: OpenCV imported after "
    f"{time.perf_counter() - _muj1_import_start:.2f}s",
    flush=True,
)

print("muj1.py: importing torch...", flush=True)
with _diagnose_stall("import torch"):
    import torch
print(
    f"muj1.py: torch imported after "
    f"{time.perf_counter() - _muj1_import_start:.2f}s",
    flush=True,
)

print("muj1.py: importing MuJoCo...", flush=True)
with _diagnose_stall("import mujoco"):
    import mujoco
print(
    f"muj1.py: MuJoCo imported after "
    f"{time.perf_counter() - _muj1_import_start:.2f}s",
    flush=True,
)


print("muj1.py: importing RoboHive gym...", flush=True)
from robohive.utils import gym
print(
    f"muj1.py: RoboHive gym imported after "
    f"{time.perf_counter() - _muj1_import_start:.2f}s",
    flush=True,
)


print("muj1.py: importing RoboHive environments...", flush=True)

# This import registers the RoboHive hand environments,
# including relocate-v1.
import robohive
import robohive.envs.hands

print(
    f"muj1.py: RoboHive environments imported after "
    f"{time.perf_counter() - _muj1_import_start:.2f}s",
    flush=True,
)


# This import is only used by the old plotting code below.
# Keep it so the rest of your file continues to work unchanged.
from robohive.utils.paths_utils import plot as plotnsave_paths


MODULE_DIRECTORY = os.path.dirname(
    os.path.abspath(__file__)
)

TASKS = {
    "relocate": {
        "environment": "relocate-v1",
        "policy": os.path.join(
            MODULE_DIRECTORY, "paultrain1", "iterations", "best_policy.pickle"
        ),
        "interactive_target": True,
    },
    "hammer": {
        "environment": "hammer-v1",
        "environment_kwargs": {
            "obs_keys": [
                "hand_jnt", "obj_vel", "palm_pos", "obj_pos",
                "obj_rot", "target_pos", "nail_impact",
            ],
        },
        "policy": os.path.join(
            MODULE_DIRECTORY, "hand_dapg", "dapg", "policies", "hammer-v0.pickle"
        ),
        "interactive_target": False,
    },
    "door": {
        "environment": "door-v1",
        "environment_kwargs": {
            "obs_keys": [
                "hand_jnt", "latch_pos", "door_pos", "palm_pos",
                "handle_pos", "reach_err", "door_open",
            ],
        },
        "policy": os.path.join(
            MODULE_DIRECTORY, "hand_dapg", "dapg", "policies", "door-v0.pickle"
        ),
        "interactive_target": False,
    },
    "pen": {
        "environment": "pen-v1",
        "policy": os.path.join(
            MODULE_DIRECTORY, "hand_dapg", "dapg", "policies", "pen-v0.pickle"
        ),
        "interactive_target": False,
    },
}


def _generated_relocate_model(object_spec):
    source_path = os.path.join(
        MODULE_DIRECTORY, "robohive", "robohive", "envs", "hands",
        "assets", "DAPG_relocate.xml",
    )
    tree = ET.parse(source_path)
    object_body = tree.find(".//body[@name='Object']")
    if object_body is None:
        raise ValueError("Relocation object body was not found")
    for child in list(object_body):
        if child.tag in {"geom", "inertial"}:
            object_body.remove(child)

    lowest_point = 0.0
    for index, part in enumerate(object_spec["parts"]):
        shape = part["shape"]
        size = part["size"]
        if shape == "sphere":
            mj_size = [size[0]]
            extent_z = size[0]
        elif shape in {"capsule", "cylinder"}:
            mj_size = [size[0], size[1]]
            extent_z = size[0] + size[1]
        else:
            mj_size = size
            extent_z = size[2]
        position = part["position"]
        lowest_point = min(lowest_point, position[2] - extent_z)
        ET.SubElement(object_body, "geom", {
            "name": f"generated_part_{index}",
            "type": shape,
            "size": " ".join(f"{value:.6g}" for value in mj_size),
            "pos": " ".join(f"{value:.6g}" for value in position),
            "euler": " ".join(f"{value:.6g}" for value in part["euler"]),
            "rgba": " ".join(f"{value:.6g}" for value in part["rgba"]),
            "mass": f"{part['mass']:.6g}",
            "condim": "4",
        })
    body_position = [float(v) for v in object_body.get("pos", "0 0 0.035").split()]
    body_position[2] = max(0.01, -lowest_point + 0.003)
    object_body.set("pos", " ".join(f"{value:.6g}" for value in body_position))

    model_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", prefix="generated_relocate_",
        dir=os.path.dirname(source_path), delete=False, encoding="utf-8",
    )
    try:
        tree.write(model_file, encoding="unicode", xml_declaration=True)
        return model_file.name
    finally:
        model_file.close()


def _editor_scene_model(scene_assets):
    source_path = os.path.join(
        MODULE_DIRECTORY, "robohive", "robohive", "envs", "hands",
        "assets", "DAPG_relocate.xml",
    )
    tree = ET.parse(source_path)
    worldbody = tree.find("worldbody")
    if worldbody is None:
        raise ValueError("Scene worldbody was not found")
    for index, item in enumerate(scene_assets):
        position = item["position"]
        rotation = item["rotation"]
        scale = item["scale"]
        asset = item["asset"]
        body = ET.SubElement(worldbody, "body", {
            "name": f"editor_asset_{index}",
            "pos": " ".join(str(value) for value in position),
            "euler": " ".join(str(value) for value in rotation),
        })
        common = {"contype": "0", "conaffinity": "0", "rgba": "0.98 0.42 0.1 1"}
        if asset == "hammer":
            ET.SubElement(body, "geom", {**common, "type": "capsule", "fromto": "0 0 -0.065 0 0 0.065", "size": "0.012"})
            ET.SubElement(body, "geom", {**common, "type": "box", "pos": "0 0 0.07", "size": "0.05 0.016 0.018"})
        else:
            geom_type = {"box": "box", "sphere": "sphere", "cylinder": "cylinder"}[asset]
            size = [scale[0]] if asset == "sphere" else [scale[0], scale[2]] if asset == "cylinder" else scale
            ET.SubElement(body, "geom", {**common, "type": geom_type, "size": " ".join(str(value) for value in size)})
    model_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", prefix="editor_scene_",
        dir=os.path.dirname(source_path), delete=False, encoding="utf-8",
    )
    try:
        tree.write(model_file, encoding="unicode", xml_declaration=True)
        return model_file.name
    finally:
        model_file.close()

# Development settings
render = "none"
num_episodes = 1
ENABLE_WEB_TEST = True


def _integer_setting(
    name,
    default,
    minimum,
    maximum,
):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        print(
            f"Ignoring invalid {name}={raw_value!r}; using {default}",
            flush=True,
        )
        return default
    if not minimum <= value <= maximum:
        print(
            f"Ignoring out-of-range {name}={value}; using {default}",
            flush=True,
        )
        return default
    return value


SIMULATION_WIDTH = _integer_setting(
    "SIMULATION_WIDTH",
    480,
    160,
    1920,
)
SIMULATION_HEIGHT = _integer_setting(
    "SIMULATION_HEIGHT",
    360,
    120,
    1080,
)


print(
    f"muj1.py: all module imports completed after "
    f"{time.perf_counter() - _muj1_import_start:.2f}s",
    flush=True,
)

DESC = '''
Helper script to examine an environment and associated policy for behaviors; \n
- either onscreen, or offscreen, or just rollout without rendering.\n
- save resulting paths as pickle or as 2D plots \n
- rollout either learned policies or scripted policies (e.g. see rand_policy class below) \n
USAGE:\n
    $ python examine_env.py --env_name door-v1 \n
    $ python examine_env.py --env_name door-v1 --policy_path robohive.utils.examine_env.rand_policy \n
    $ python examine_env.py --env_name door-v1 --policy_path my_policy.pickle --mode evaluation --episodes 10 \n
'''

class Dummy(object):
    def __init__(self, *args, **kwargs):
        pass

class PolicyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # The published DAPG checkpoints use the original ``mjrl`` package
        # name. This repository carries the compatible implementation as
        # ``mjrlpaul`` so both old and newly trained policies can be loaded.
        if module == "mjrl" or module.startswith("mjrl."):
            module = "mjrlpaul" + module[len("mjrl"):]
        print(
            f"PolicyUnpickler: requesting {module}.{name}",
            flush=True,
        )

        class_load_start = time.perf_counter()

        try:
            loaded_class = super().find_class(module, name)
        except Exception as exc:
            print(
                f"PolicyUnpickler: FAILED to load "
                f"{module}.{name}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise

        print(
            f"PolicyUnpickler: loaded {module}.{name} in "
            f"{time.perf_counter() - class_load_start:.2f}s",
            flush=True,
        )

        return loaded_class
# Random policy
class rand_policy():
    def __init__(self, env, seed):
        self.env = env
        self.env.action_space.seed(seed) # requires explicit seeding

    def get_action(self, obs):
        # return self.env.np_random.uniform(high=self.env.action_space.high, low=self.env.action_space.low)
        return self.env.action_space.sample(), {'mode': 'random samples', 'evaluation':self.env.action_space.sample()}

def load_class_from_str(module_name, class_name):
    try:
        m = __import__(module_name, globals(), locals(), class_name)
        return getattr(m, class_name)
    except (ImportError, AttributeError):
        return None

def test_frame_callback(frame, metadata):
    print(frame.shape)

def test_frame_callback(frame, metadata):
    print(
        f"Frame received: shape={frame.shape}, "
        f"dtype={frame.dtype}, "
        f"episode={metadata['episode']}, "
        f"step={metadata['step']}"
    )

    # Save only the first frame of each episode.
    if metadata["step"] == 0:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        filename = f"test_frame_episode_{metadata['episode']}.jpg"

        success = cv2.imwrite(filename, frame_bgr)

        if success:
            print(f"Saved {filename}")
        else:
            print(f"Could not save {filename}")

# MAIN =========================================================
"""
@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required= False, default=env_name)
@click.option('-p', '--policy_path', type=str, help='absolute path of the policy file', default=policy_path)
@click.option('-m', '--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-n', '--num_episodes', type=int, help='number of episodes to visualize', default=10)
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='visualize onscreen or offscreen', default='onscreen')
@click.option('-c', '--camera_name', type=str, default=None, help=('Camera name for rendering'))
@click.option('-o', '--output_dir', type=str, default='./', help=('Directory to save the outputs'))
@click.option('-on', '--output_name', type=str, default=None, help=('The name to save the outputs as'))
@click.option('-sp', '--save_paths', type=bool, default=False, help=('Save the rollout paths'))
@click.option('-pp', '--plot_paths', type=bool, default=False, help=('2D-plot of individual paths'))
@click.option('-rv', '--render_visuals', type=bool, default=False, help=('render the visual keys of the env, if present'))
@click.option('-ea', '--env_args', type=str, default=None, help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
"""

def run_simulation(
    frame_callback=None,
    target_queue=None,
    control_queue=None,
    pause_event=None,
    task_id="relocate",
    object_spec=None,
    editor_mode=False,
    editor_camera=None,
    scene_assets=None,
    stop_event=None,
):
    simulation_start = time.perf_counter()

    print("run_simulation: started", flush=True)
    print(
        "run_simulation: video configuration:",
        {
            "width": SIMULATION_WIDTH,
            "height": SIMULATION_HEIGHT,
            "jpeg_quality": os.environ.get("JPEG_QUALITY", "70"),
        },
        flush=True,
    )

    if task_id not in TASKS:
        raise ValueError(f"Unknown simulation task: {task_id!r}")

    task = TASKS[task_id]
    env_name_local = task["environment"]
    policy_path_local = task["policy"]
    seed = 123
    mode = "evaluation"
    camera_name_local = None

    print("run_simulation: importing MuJoCo...", flush=True)
    import mujoco
    
    print(
        f"run_simulation: MuJoCo imported after "
        f"{time.perf_counter() - simulation_start:.2f}s",
        flush=True,
    )
    print("ABA ---------------- CAC", flush=True)
    
    np.random.seed(seed)

    print(
        "RoboHive loaded from:",
        robohive.__file__,
        flush=True,
    )
    print(
        "Requested environment:",
        repr(env_name_local),
        flush=True,
    )

    try:
        registered_ids = sorted(gym.envs.registry.keys())
    except AttributeError:
        registered_ids = sorted(
            spec.id for spec in gym.envs.registry.values()
        )

    print(
        "Registered DAPG environments:",
        [
            environment_id
            for environment_id in registered_ids
            if any(name in environment_id.lower() for name in TASKS)
        ],
        flush=True,
    )

    print(
        f"run_simulation: creating environment "
        f"{env_name_local!r}...",
        flush=True,
    )

    environment_start = time.perf_counter()

    environment_kwargs = dict(task.get("environment_kwargs", {}))
    generated_model_path = None
    if editor_mode and scene_assets:
        generated_model_path = _editor_scene_model(scene_assets)
        environment_kwargs["model_path"] = generated_model_path
    elif object_spec is not None:
        generated_model_path = _generated_relocate_model(object_spec)
        environment_kwargs["model_path"] = generated_model_path
    try:
        envw = gym.make(env_name_local, **environment_kwargs)
    finally:
        if generated_model_path:
            try:
                os.unlink(generated_model_path)
            except OSError:
                pass
    env = envw.unwrapped
    env.seed(seed)

    print(
        f"run_simulation: environment created in "
        f"{time.perf_counter() - environment_start:.2f}s",
        flush=True,
    )
    print("\n=== MuJoCo bodies ===")
    
    for body_id in range(env.sim.model.nbody):
        body_name = env.sim.model.id2name(
            body_id,
            "body",
        )

        print(
            f"body_id={body_id}, "
            f"name={body_name!r}, "
            f"position={env.sim.data.body_xpos[body_id]}"
        )


    print("\n=== MuJoCo joints ===")

    for joint_id in range(env.sim.model.njnt):
        joint_name = env.sim.model.id2name(
            joint_id,
            "joint",
        )

        joint_type = env.sim.model.jnt_type[joint_id]
        qpos_address = env.sim.model.jnt_qposadr[joint_id]

        print(
            f"joint_id={joint_id}, "
            f"name={joint_name!r}, "
            f"type={joint_type}, "
            f"qpos_address={qpos_address}"
        )

    print("=== End MuJoCo model information ===\n")

    camera_defaults = {
        "azimuth": 90.0,
        "elevation": -35.0,
        "distance": 2.35,
        "lookat": np.array([0.0, -0.12, 0.18]),
    }
    interactive_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(interactive_camera)

    def reset_camera():
        interactive_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        interactive_camera.azimuth = camera_defaults["azimuth"]
        interactive_camera.elevation = camera_defaults["elevation"]
        interactive_camera.distance = camera_defaults["distance"]
        interactive_camera.lookat[:] = camera_defaults["lookat"]

    reset_camera()
    if editor_mode and editor_camera:
        interactive_camera.azimuth = editor_camera["azimuth"]
        interactive_camera.elevation = editor_camera["elevation"]
        interactive_camera.distance = editor_camera["distance"]
        interactive_camera.lookat[:] = editor_camera["lookat"]
    camera_name_local = interactive_camera
    print("Interactive orbit camera initialized", camera_defaults, flush=True)

    if editor_mode:
        # Match the initialized DAPG scene shown at the beginning of a normal
        # rollout before taking the static editor frame.
        env.reset()
        env.sim.forward()
        while stop_event is None or not stop_event.is_set():
            if control_queue is not None:
                while True:
                    try:
                        command = control_queue.get_nowait()
                    except queue.Empty:
                        break
                    if command.get("type") == "camera_orbit":
                        interactive_camera.azimuth -= command["delta_x"] * 0.25
                        interactive_camera.elevation = float(np.clip(interactive_camera.elevation - command["delta_y"] * 0.2, -85.0, -5.0))
                    elif command.get("type") == "camera_zoom":
                        interactive_camera.distance = float(np.clip(interactive_camera.distance * (1.12 ** command["delta"]), 0.45, 5.0))
                    elif command.get("type") == "camera_reset":
                        reset_camera()
            frame = env.sim.renderer.render_offscreen(width=SIMULATION_WIDTH, height=SIMULATION_HEIGHT, camera_id=interactive_camera, device_id=0)
            if frame_callback is not None:
                frame_callback(frame, {
                    "editor_preview": True,
                    "episode": 0,
                    "step": 0,
                    "simulation_time": 0.0,
                    "reward": 0.0,
                    "camera": {
                        "azimuth": float(interactive_camera.azimuth),
                        "elevation": float(interactive_camera.elevation),
                        "distance": float(interactive_camera.distance),
                        "lookat": [float(value) for value in interactive_camera.lookat],
                    },
                })
            time.sleep(1 / 20)
        return None

    # Load the trained policy.
    # Load the trained policy.
    print(
        "run_simulation: loading policy from:",
        policy_path_local,
        flush=True,
    )

    if not os.path.isfile(policy_path_local):
        raise FileNotFoundError(
            "Policy file was not found: "
            f"{policy_path_local!r}. "
            "This path must exist inside the Render Docker container."
        )

    policy_load_start = time.perf_counter()

    # Resolve the policy class before entering pickle.  The old final log line
    # came from find_class(), immediately before this import was attempted.
    with _diagnose_stall("import mjrlpaul"):
        import mjrlpaul
    print(
        "run_simulation: mjrlpaul imported from:",
        mjrlpaul.__file__,
        flush=True,
    )

    with _diagnose_stall(
        "import mjrlpaul.policies.gaussian_mlp.MLP"
    ):
        from mjrlpaul.policies.gaussian_mlp import MLP
    print(
        "run_simulation: policy class imported:",
        f"{MLP.__module__}.{MLP.__name__}",
        flush=True,
    )

    print("run_simulation: opening policy pickle", flush=True)
    with open(policy_path_local, "rb") as policy_file:
        print(
            "run_simulation: policy pickle opened; reconstructing policy",
            flush=True,
        )
        with _diagnose_stall("unpickle policy"):
            pi = PolicyUnpickler(policy_file).load()

    print(
        f"run_simulation: policy reconstructed in "
        f"{time.perf_counter() - policy_load_start:.2f}s",
        flush=True,
    )

    import inspect

    print("Environment class:", type(env))
    print(
        "examine_policy_new file:",
        inspect.getfile(env.examine_policy_new),
    )
    print(
        "examine_policy_new signature:",
        inspect.signature(env.examine_policy_new),
    )

    latest_target = None

    def read_latest_browser_target():
        """
        Read all pending clicks and retain only the newest one.
        """

        nonlocal latest_target

        if target_queue is None:
            return latest_target

        while True:
            try:
                latest_target = target_queue.get_nowait()
            except queue.Empty:
                break
            
        return latest_target

    def image_click_to_table_position(
        u,
        v,
        table_z=0.0,
        image_width=640,
        image_height=480,
    ):
        """
        Convert normalized image coordinates into a world-space point
        by intersecting a camera ray with the horizontal table plane.
        """

        azimuth = np.deg2rad(interactive_camera.azimuth)
        elevation = np.deg2rad(interactive_camera.elevation)
        camera_position = interactive_camera.lookat + interactive_camera.distance * np.array([
            np.cos(elevation) * np.cos(azimuth),
            -np.cos(elevation) * np.sin(azimuth),
            -np.sin(elevation),
        ])
        forward = interactive_camera.lookat - camera_position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        aspect_ratio = image_width / image_height

        vertical_fov = np.deg2rad(45.0)

        half_height = np.tan(vertical_fov / 2.0)
        half_width = aspect_ratio * half_height

        image_x = (2.0 * u - 1.0) * half_width
        image_y = (1.0 - 2.0 * v) * half_height

        ray_camera = np.array(
            [
                image_x,
                image_y,
                1.0,
            ],
            dtype=np.float64,
        )

        ray_camera /= np.linalg.norm(ray_camera)

        ray_world = (
            right * ray_camera[0]
            + up * ray_camera[1]
            + forward * ray_camera[2]
        )
        ray_world /= np.linalg.norm(ray_world)

        if abs(ray_world[2]) < 1e-8:
            raise ValueError(
                "Camera ray is parallel to the table plane"
            )

        distance = (
            table_z - camera_position[2]
        ) / ray_world[2]

        if distance <= 0:
            raise ValueError(
                "Clicked ray does not intersect the table "
                "in front of the camera"
            )

        intersection = (
            camera_position
            + distance * ray_world
        )

        print(
            "Click projection:",
            f"u={u:.4f},",
            f"v={v:.4f},",
            f"camera={camera_position},",
            f"intersection={intersection}",
        )

        return intersection
    
    def browser_episode_reset_callback(
        callback_env,
        episode_index,
    ):
        """
        Called immediately after the environment randomizes and resets
        the object for a new episode.
        """

        if not task["interactive_target"]:
            return

        target = read_latest_browser_target()

        if target is None:
            print(
                f"Episode {episode_index}: "
                "no browser target available; "
                "keeping randomized object position"
            )
            return

        u = target["u"]
        v = target["v"]

        table_position = image_click_to_table_position(
            u=u,
            v=v,
            table_z=0.0,
            image_width=SIMULATION_WIDTH,
            image_height=SIMULATION_HEIGHT,
        )

        target_x = float(table_position[0])
        target_y = float(table_position[1])

        object_z = 0.035

        callback_env.sim.data.qpos[30] = target_x
        callback_env.sim.data.qpos[31] = target_y
        callback_env.sim.data.qpos[32] = object_z
        
        callback_env.sim.forward()

        print(
            f"Episode {episode_index}: "
            "applied browser target:",
            f"u={u:.4f},",
            f"v={v:.4f},",
            f"x={target_x:.4f},",
            f"y={target_y:.4f},",
            f"z={object_z:.4f}",
        )
    
    def apply_camera_commands():
        changed = False
        if control_queue is None:
            return changed
        while True:
            try:
                command = control_queue.get_nowait()
            except queue.Empty:
                break
            if command["type"] == "camera_orbit":
                interactive_camera.azimuth -= command["delta_x"] * 0.25
                interactive_camera.elevation = float(np.clip(
                    interactive_camera.elevation - command["delta_y"] * 0.2,
                    -85.0,
                    -5.0,
                ))
                changed = True
            elif command["type"] == "camera_zoom":
                interactive_camera.distance = float(np.clip(
                    interactive_camera.distance * (1.12 ** command["delta"]),
                    0.45,
                    5.0,
                ))
                changed = True
            elif command["type"] == "camera_reset":
                reset_camera()
                changed = True
        return changed

    def interactive_frame_callback(frame, metadata):
        """
        Forward rendered frames to server.py.

        Browser clicks are consumed here as well, so a click made during
        the current episode is retained for the next episode reset.
        """

        if metadata.get("episode") == 0 and metadata.get("step") == 0:
            # The first frame proves policy evaluation and OSMesa rendering
            # both returned, so the pre-frame stall watchdog is no longer
            # needed while the remaining rollout continues.
            faulthandler.cancel_dump_traceback_later()
            print(
                "run_simulation: first frame callback called",
                flush=True,
            )

        read_latest_browser_target()
        apply_camera_commands()

        if frame_callback is not None:
            frame_callback(frame, metadata)

        while pause_event is not None and pause_event.is_set():
            if apply_camera_commands() and frame_callback is not None:
                paused_frame = env.sim.renderer.render_offscreen(
                    width=SIMULATION_WIDTH,
                    height=SIMULATION_HEIGHT,
                    camera_id=interactive_camera,
                    device_id=0,
                )
                frame_callback(paused_frame, {**metadata, "paused": True})
            time.sleep(0.03)

    print("run_simulation: starting examine_policy_new()", flush=True)
    with _diagnose_stall(
        "examine_policy_new before first frame",
        timeout_seconds=30,
    ):
        paths = env.examine_policy_new(
            policy=pi,
            horizon=envw.spec.max_episode_steps,
            num_episodes=10,
            frame_size=(SIMULATION_WIDTH, SIMULATION_HEIGHT),
            mode=mode,
            output_dir="./",
            filename="web_test",
            camera_name=camera_name_local,
            render="none",
            frame_callback=interactive_frame_callback,
            episode_reset_callback=(
                browser_episode_reset_callback
                if task["interactive_target"]
                else None
            ),
        )
    print("run_simulation: examine_policy_new() returned", flush=True)

    # evaluate paths
    success_percentage = env.evaluate_success(paths)
    print(f'Average success over rollouts: {success_percentage}%')
    return paths

    # save paths
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    if save_paths:
        file_name = output_dir + '/' + output_name + '{}_trace.h5'.format(time_stamp)
        paths.save(trace_name=file_name, verify_length=True, f_res=np.float64)

    # plot paths
    if plot_paths:
        file_name = output_dir + '/' + output_name + '{}'.format(time_stamp)
        plotnsave_paths(paths, env=env, fileName_prefix=file_name)

    # render visuals keys
    if env.visual_keys and render_visuals:
        paths.close()
        render_keys = ['env_infos/visual_dict/'+ key for key in env.visual_keys]
        paths.render(output_dir=output_dir, output_format="mp4", groups=["Trial0",], datasets=render_keys, input_fps=1/env.dt)

if __name__ == '__main__':
    #main()
    print("ABA ---------------- CAC")
    run_simulation(
        frame_callback=test_frame_callback
    )
