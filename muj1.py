# info (paul): the examine_env.py

""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from robohive.utils import gym
from robohive.utils.paths_utils import plot as plotnsave_paths
import click
import numpy as np
import pickle
import time
import os
import cv2
import queue

#import mjrlpaul

# Run this first to find the right module path
import robohive
import robohive.envs.hands
import pkgutil

for importer, modname, ispkg in pkgutil.walk_packages(
    path=robohive.__path__,
    prefix='robohive.',
    onerror=lambda x: None
):
    if 'polic' in modname.lower() or 'gaussian' in modname.lower() or 'npg' in modname.lower():
        print(modname)

env_name = "relocate-v1"#"hammer-v1"#"FrankaReachRandom-v0"
#policy_path = "/home/paul/PycharmProjects/dapg/hand_dapg/dapg/policies/relocate-v0.pickle"
policy_path = "/home/paul/PycharmProjects/dapg/paultrain1/iterations/best_policy.pickle"#best_policy.pickle"
#policy_path = "/home/paul/PycharmProjects/dapg/hand_dapg/dapg/policies/policy_paul70rl.pickle"

# Development settings
render = "none"
num_episodes = 1
ENABLE_WEB_TEST = True

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
        print(f"pickle wants: {module}.{name}")   # <-- add this
        
        #redirect mjrl → robohive equivalent
        if False and module.startswith("mjrl"):
            module = module.replace("mjrl", "mjrlpaul")#"robohive")
        #try:
        #    return super().find_class(module, name)
        #except Exception:
        #    return Dummy  # fallback for unknown classes
        
        return super().find_class(module, name)

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
):
    env_name_local = env_name
    policy_path_local = policy_path
    seed = 123
    mode = "evaluation"
    camera_name_local = None

    import mujoco
    import numpy as np
    np.random.seed(seed)




    print("RoboHive loaded from:", robohive.__file__, flush=True)
    print("Requested environment:", repr(env_name_local), flush=True)

    try:
        registered_ids = sorted(gym.envs.registry.keys())
    except AttributeError:
        registered_ids = sorted(
            spec.id for spec in gym.envs.registry.values()
        )

    print(
        "Registered relocate environments:",
        [
            environment_id
            for environment_id in registered_ids
            if "relocate" in environment_id.lower()
        ],
        flush=True,
    )



    
    envw = gym.make(env_name_local)
    env = envw.unwrapped
    env.seed(seed)

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

    camera_name_local = "fixed"

    camera_id = env.sim.model.name2id(
        camera_name_local,
        "camera",
    )
    
    env.sim.model.cam_pos[camera_id] = np.array([
        0.0,
        -2.0,
        2.0,
    ])

    env.sim.model.cam_fovy[camera_id] = 32.0

    print("\n=== Active camera configuration ===")
    print("Camera name:", camera_name_local)
    print("Camera ID:", camera_id)
    print("Camera position:", env.sim.model.cam_pos[camera_id])
    print("Camera quaternion:", env.sim.model.cam_quat[camera_id])
    print("Camera FOV:", env.sim.model.cam_fovy[camera_id])
    print("=== End camera configuration ===\n")

    env.sim.model.cam_fovy[camera_id] = 30.0
    print(
        "Final camera configuration:",
        {
            "position": env.sim.model.cam_pos[camera_id].copy(),
            "quaternion": env.sim.model.cam_quat[camera_id].copy(),
            "fovy": float(env.sim.model.cam_fovy[camera_id]),
        },
    )

    ##AA

    print("Available cameras:")
    
    for camera_id in range(env.sim.model.ncam):
        try:
            camera_name = env.sim.model.id2name(
                camera_id,
                "camera",
            )
        except Exception:
            camera_name = None
    
        print(
            camera_id,
            camera_name,
            "position:",
            env.sim.model.cam_pos[camera_id],
            "fovy:",
            env.sim.model.cam_fovy[camera_id],
        )

    ##AA

    try:
        camera_id = env.sim.model.name2id(
            camera_name_local,
            "camera",
        )
    except Exception as exc:
        raise ValueError(
            f"Camera '{camera_name_local}' was not found"
        ) from exc

    print("Camera ID:", camera_id)
    print(
        "Original camera position:",
        env.sim.model.cam_pos[camera_id].copy(),
    )
    print(
        "Original camera field of view:",
        env.sim.model.cam_fovy[camera_id],
    )

    # Smaller value = stronger zoom.
    env.sim.model.cam_fovy[camera_id] = 30.0

    # Load the trained policy.
    policy_l = open(policy_path_local, "rb")
    pi = PolicyUnpickler(policy_l).load()

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

        callback_sim = env.sim
        callback_sim.forward()

        camera_position = (
            callback_sim.data.cam_xpos[camera_id]
            .copy()
        )

        camera_rotation = (
            callback_sim.data.cam_xmat[camera_id]
            .reshape(3, 3)
            .copy()
        )

        aspect_ratio = image_width / image_height

        vertical_fov = np.deg2rad(
            callback_sim.model.cam_fovy[camera_id]
        )

        half_height = np.tan(vertical_fov / 2.0)
        half_width = aspect_ratio * half_height

        image_x = (2.0 * u - 1.0) * half_width
        image_y = (1.0 - 2.0 * v) * half_height

        ray_camera = np.array(
            [
                image_x,
                image_y,
                -1.0,
            ],
            dtype=np.float64,
        )

        ray_camera /= np.linalg.norm(ray_camera)

        ray_world = camera_rotation @ ray_camera
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
            image_width=640,
            image_height=480,
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
    
    def interactive_frame_callback(frame, metadata):
        """
        Forward rendered frames to server.py.

        Browser clicks are consumed here as well, so a click made during
        the current episode is retained for the next episode reset.
        """

        read_latest_browser_target()

        if frame_callback is not None:
            frame_callback(frame, metadata)

    paths = env.examine_policy_new(
        policy=pi,
        horizon=envw.spec.max_episode_steps,
        num_episodes=10,
        frame_size=(640, 480),
        mode=mode,
        output_dir="./",
        filename="web_test",
        camera_name=camera_name_local,
        render="none",
        frame_callback=interactive_frame_callback,
        episode_reset_callback=browser_episode_reset_callback,
    )

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
    run_simulation(
        frame_callback=test_frame_callback
    )
