from gym.envs.registration import register

# ----------------------------------------
# mjrl environments
# ----------------------------------------

register(
    id='mjrl_point_mass-v0',
    entry_point='mjrl.envs:PointMassEnv',
    max_episode_steps=25,
)

register(
    id='mjrl_swimmer-v0',
    entry_point='mjrl.envs:SwimmerEnv',
    max_episode_steps=500,
)

register(
    id='mjrl_reacher_7dof-v0',
    entry_point='mjrl.envs:Reacher7DOFEnv',
    max_episode_steps=50,
)

register(
    id='mjrl_peg_insertion-v0',
    entry_point='mjrl.envs:PegEnv',
    max_episode_steps=50,
)

if False:#01052026
    from mjrlpaul.envs.mujoco_env import MujocoEnv
    # ^^^^^ so that user gets the correct error
    # message if mujoco is not installed correctly
    from mjrlpaul.envs.point_mass import PointMassEnv
    from mjrlpaul.envs.swimmer import SwimmerEnv
    from mjrlpaul.envs.reacher_sawyer import Reacher7DOFEnv
    from mjrlpaul.envs.peg_insertion_sawyer import PegEnv
