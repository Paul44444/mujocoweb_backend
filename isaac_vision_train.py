"""Launch the official Isaac Lab RSL-RL trainer with web vision tasks registered."""

from pathlib import Path
import json
import os
import sys

script = Path("/home/paul/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py")
sys.path.insert(0, str(script.parent))
source = script.read_text(encoding="utf-8")
source = source.replace("import os\n", "import os\nimport json\nfrom pathlib import Path\n", 1)
marker = "import isaaclab_tasks  # noqa: F401"
if marker not in source:
    raise RuntimeError("Isaac Lab train.py registration marker changed")
source = source.replace(
    marker,
    marker + "\nimport isaac_vision_task  # register web vision environments"
    + "\nimport isaac_labware_task  # register web labware environments"
    + "\nimport isaac_rack_insert_task  # register additional upright insertion task",
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
        from isaac_demo_ppo import prepare_resumed_demo_dataset, checkpoint_demo_state
        resume_state = None
        if resume_demo_infos.get(\"demo_control\"):
            prepare_resumed_demo_dataset(runner, demonstration_paths, demo_episode_seconds, resume_demo_infos.get(\"demo_dataset\"))
            resume_state = resume_demo_infos.get(\"demo_ppo_state\")
            if resume_state is None and resume_demo_infos.get(\"demo_guided_ppo\"):
                legacy_config = json.loads(Path(os.environ[\"ISAAC_RESUME_CHECKPOINT\"]).parent.parent.joinpath(\"config.json\").read_text())
                horizon = int(legacy_config.get(\"iterations\", agent_cfg.max_iterations))
                resume_state = {\"iteration\": int(resume_payload.get(\"iter\", 0)) + 1, \"curriculum_iterations\": horizon, \"warmup_iterations\": min(20, max(1, horizon // 10))}
                print(\"Restoring legacy demo curriculum from checkpoint iteration and original run configuration\", flush=True)
            if resume_demo_infos.get(\"phase\") != \"behavior_cloning\":
                runner.current_learning_iteration = int(resume_payload.get(\"iter\", 0)) + 1
            print(\"Continuing demo policy without repeating behavior cloning\", flush=True)
        else:
            pretrain_runner_from_demonstrations(
                runner, demonstration_paths,
                int(os.environ.get(\"ISAAC_BC_EPOCHS\", \"200\")),
            )
        runner.web_demo_control = True
        from isaac_demo_ppo import install_demo_guided_ppo
        install_demo_guided_ppo(runner, demonstration_paths, agent_cfg.max_iterations, resume_state)
        if not resume_demo_infos.get(\"demo_control\"):
            torch.save({
            \"model_state_dict\": runner.alg.policy.state_dict(),
            \"optimizer_state_dict\": runner.alg.optimizer.state_dict(),
            \"iter\": 0,
            \"infos\": {**checkpoint_demo_state(runner, demonstration_paths, demo_episode_seconds), \"phase\": \"behavior_cloning\"},
            }, os.path.join(log_dir, \"model_bc.pt\"))
        original_save = runner.save
        def save_with_demo_control(path, infos=None):
            original_save(path, infos={**(infos or {}), **checkpoint_demo_state(runner, demonstration_paths, demo_episode_seconds)})
        runner.save = save_with_demo_control

""" + training_marker,
    1,
)
environment_marker = "    # create isaac environment"
if environment_marker not in source:
    raise RuntimeError("Isaac Lab train.py environment marker changed")
source = source.replace(
    environment_marker,
    """    demonstration_paths = json.loads(os.environ.get(\"ISAAC_DEMO_PATHS\", \"[]\"))
    resume_payload = {}
    resume_demo_infos = {}
    if os.environ.get(\"ISAAC_RESUME_CHECKPOINT\"):
        resume_payload = torch.load(os.environ[\"ISAAC_RESUME_CHECKPOINT\"], map_location=\"cpu\", weights_only=False)
        resume_demo_infos = resume_payload.get(\"infos\") or {}
        if resume_demo_infos.get(\"demo_paths\"):
            demonstration_paths = resume_demo_infos[\"demo_paths\"]
            os.environ[\"ISAAC_DEMO_PATHS\"] = json.dumps(demonstration_paths)
        for path in demonstration_paths:
            if not Path(path).is_file():
                raise RuntimeError(f\"Required checkpoint demonstration is missing: {path}\")
    if demonstration_paths:
        from isaac_demo_bc import longest_demonstration_seconds
        demo_episode_seconds = longest_demonstration_seconds(
            demonstration_paths,
            float(env_cfg.sim.dt) * int(env_cfg.decimation),
        )
        demo_episode_seconds = float(resume_demo_infos.get(\"demo_duration\", demo_episode_seconds))
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
