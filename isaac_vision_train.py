"""Launch the official Isaac Lab RSL-RL trainer with web vision tasks registered."""

from pathlib import Path
import json
import os
import sys

script = Path("/home/paul/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py")
sys.path.insert(0, str(script.parent))
source = script.read_text(encoding="utf-8")
source = source.replace("import os\n", "import os\nimport json\n", 1)
marker = "import isaaclab_tasks  # noqa: F401"
if marker not in source:
    raise RuntimeError("Isaac Lab train.py registration marker changed")
source = source.replace(
    marker,
    marker + "\nimport isaac_vision_task  # register web vision environments"
    + "\nimport isaac_labware_task  # register web labware environments",
    1,
)
training_marker = "    # dump the configuration into log-directory"
if training_marker not in source:
    raise RuntimeError("Isaac Lab train.py runner marker changed")
source = source.replace(
    training_marker,
    """    demonstration_paths = json.loads(os.environ.get(\"ISAAC_DEMO_PATHS\", \"[]\"))
    if demonstration_paths:
        from isaac_demo_bc import pretrain_runner_from_demonstrations
        os.makedirs(log_dir, exist_ok=True)
        pretrain_runner_from_demonstrations(
            runner,
            demonstration_paths,
            int(os.environ.get(\"ISAAC_BC_EPOCHS\", \"200\")),
        )
        runner.web_demo_control = True
        torch.save({
            \"model_state_dict\": runner.alg.policy.state_dict(),
            \"optimizer_state_dict\": runner.alg.optimizer.state_dict(),
            \"iter\": 0,
            \"infos\": {\"phase\": \"behavior_cloning\", \"demo_control\": True, \"demo_duration\": demo_episode_seconds},
        }, os.path.join(log_dir, \"model_bc.pt\"))
        original_save = runner.save
        def save_with_demo_control(path, infos=None):
            original_save(path, infos={**(infos or {}), \"demo_control\": True, \"demo_duration\": demo_episode_seconds, \"demo_guided_ppo\": True})
        runner.save = save_with_demo_control
        from isaac_demo_ppo import install_demo_guided_ppo
        install_demo_guided_ppo(runner, demonstration_paths, agent_cfg.max_iterations)

""" + training_marker,
    1,
)
environment_marker = "    # create isaac environment"
if environment_marker not in source:
    raise RuntimeError("Isaac Lab train.py environment marker changed")
source = source.replace(
    environment_marker,
    """    demonstration_paths = json.loads(os.environ.get(\"ISAAC_DEMO_PATHS\", \"[]\"))
    if demonstration_paths:
        from isaac_demo_bc import longest_demonstration_seconds
        demo_episode_seconds = longest_demonstration_seconds(
            demonstration_paths,
            float(env_cfg.sim.dt) * int(env_cfg.decimation),
        )
        env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), demo_episode_seconds)
        # Match the controller used to collect web demonstrations.
        env_cfg.scene.robot.actuators[\"panda_shoulder\"].stiffness = 400.0
        env_cfg.scene.robot.actuators[\"panda_shoulder\"].damping = 80.0
        env_cfg.scene.robot.actuators[\"panda_forearm\"].stiffness = 400.0
        env_cfg.scene.robot.actuators[\"panda_forearm\"].damping = 80.0
        env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
        agent_cfg.clip_actions = None
        env_cfg.observations.policy.enable_corruption = False
        from isaac_demo_bc import demonstration_phase
        env_cfg.observations.policy.actions.func = demonstration_phase
        env_cfg.observations.policy.actions.params = {\"duration\": demo_episode_seconds}
        env_cfg.observations.policy.actions.scale = None
        from isaac_demo_ppo import reset_from_demonstrations
        env_cfg.events.reset_object_position.func = reset_from_demonstrations
        env_cfg.events.reset_object_position.params = {\"demonstration_paths\": demonstration_paths}
        print(
            f\"Demo-aware PPO episode length: {env_cfg.episode_length_s:.2f}s \"
            f\"(longest demonstration: {demo_episode_seconds:.2f}s)\",
            flush=True,
        )

""" + environment_marker,
    1,
)
namespace = {"__name__": "__main__", "__file__": str(script)}
source = source.replace(
    "init_at_random_ep_len=True",
    'init_at_random_ep_len=not bool(json.loads(os.environ.get("ISAAC_DEMO_PATHS", "[]")))',
)
exec(compile(source, str(script), "exec"), namespace)
