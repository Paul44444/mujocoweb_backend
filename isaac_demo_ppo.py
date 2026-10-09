"""Demo-preserving PPO updates and gradually randomized demonstration resets."""
import json
from pathlib import Path

import numpy as np
import torch


def curriculum_settings(iteration, total):
    progress = max(0.0, min(1.0, iteration / max(1, total)))
    # First 10%: exact demonstration. Then increase XY variation to +/-5 cm.
    variation = 0.05 * max(0.0, min(1.0, (progress - 0.10) / 0.65))
    return variation, 50.0 * (1.0 - progress) + 5.0 * progress


def reset_from_demonstrations(env, env_ids, demonstration_paths):
    """Reset every selected environment to one saved demo, offset by its origin."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not len(env_ids):
        return
    if not hasattr(env, "web_demo_initial_states"):
        env.web_demo_initial_states = []
        for path in demonstration_paths:
            with np.load(path) as data:
                env.web_demo_initial_states.append({key: np.asarray(data[key]).copy()
                    for key in data.files if key.startswith("initial_")})
    choices = torch.randint(len(env.web_demo_initial_states), (len(env_ids),), device=env.device)
    robot, obj = env.scene["robot"], env.scene["object"]
    for index, saved in enumerate(env.web_demo_initial_states):
        ids = env_ids[choices == index]
        if not len(ids):
            continue
        def value(key):
            return torch.as_tensor(saved[key], dtype=torch.float32, device=env.device).reshape(1, -1).repeat(len(ids), 1)
        robot_pose = value("initial_robot_root_pose")
        robot_pose[:, :3] += env.scene.env_origins[ids]
        robot.write_root_pose_to_sim(robot_pose, env_ids=ids)
        robot.write_root_velocity_to_sim(value("initial_robot_root_velocity"), env_ids=ids)
        joints = value("initial_robot_joint_pos")
        robot.write_joint_state_to_sim(joints, value("initial_robot_joint_vel"), env_ids=ids)
        robot.set_joint_position_target(joints, env_ids=ids)
        pose = value("initial_object_root_pose")
        pose[:, :3] += env.scene.env_origins[ids]
        variation = getattr(env, "web_demo_variation", 0.0)
        pose[:, :2] += (torch.rand((len(ids), 2), device=env.device) * 2 - 1) * variation
        obj.write_root_pose_to_sim(pose, env_ids=ids)
        obj.write_root_velocity_to_sim(value("initial_object_root_velocity"), env_ids=ids)
        if "initial_task_target_pose" in saved:
            command = env.command_manager.get_term("object_pose")
            command.command[ids] = value("initial_task_target_pose")
            command.time_left[ids] = 86400.0
    env.episode_length_buf[env_ids] = 0


def install_demo_guided_ppo(runner, demonstration_paths, total_iterations):
    """Add BC gradients to each PPO minibatch using the same optimizer step."""
    env = runner.env.unwrapped
    alg = runner.alg
    observations, actions = runner.web_demo_dataset
    observations, actions = observations.to(runner.device), actions.to(runner.device)
    state = {"iteration": 0, "losses": []}
    warmup = min(20, max(1, total_iterations // 10))
    actor_parameters = list(alg.policy.actor.parameters())
    alg.learning_rate = 1.0e-5
    alg.schedule = "fixed"
    alg.entropy_coef = 0.0
    alg.clip_param = 0.05
    alg.desired_kl = 0.002
    for group in alg.optimizer.param_groups:
        group["lr"] = alg.learning_rate

    def preserve_demo(optimizer, args, kwargs):
        if state["iteration"] < warmup:
            # Train the initially untrained critic without changing the BC actor.
            for parameter in actor_parameters:
                parameter.grad = None
        else:
            ids = torch.randint(len(observations), (min(256, len(observations)),), device=runner.device)
            prediction = alg.policy.actor(observations[ids])
            loss = 10 * (prediction[:, :7] - actions[ids, :7]).square().mean()
            loss = loss + (prediction[:, 7:] - actions[ids, 7:]).square().mean()
            _, weight = curriculum_settings(state["iteration"], total_iterations)
            (weight * loss).backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, 0.5)
            state["losses"].append(float(loss.detach()))
        # Keep exploration small; entropy must not inflate it after successful BC.
        for name in ("log_std", "std"):
            parameter = getattr(alg.policy, name, None)
            if parameter is not None:
                parameter.grad = None

    runner.web_demo_optimizer_hook = alg.optimizer.register_step_pre_hook(preserve_demo)
    original_update = alg.update
    def update():
        result = original_update()
        state["iteration"] += 1
        variation, weight = curriculum_settings(state["iteration"], total_iterations)
        env.web_demo_variation = variation
        payload = {"iteration": state["iteration"], "variation_cm": variation * 100,
                   "bc_weight": weight, "critic_warmup": state["iteration"] < warmup,
                   "bc_loss": float(np.mean(state["losses"])) if state["losses"] else None}
        state["losses"].clear()
        print("Demo-guided PPO: " + json.dumps(payload), flush=True)
        if runner.log_dir:
            (Path(runner.log_dir) / "demo_ppo_status.json").write_text(json.dumps(payload))
        return result
    alg.update = update
    # BC validation has reset physics. The runner must obtain fresh observations.
    env.reset()
    print(f"Demo-guided PPO enabled: critic warmup {warmup} iterations; BC retained; XY curriculum 0..5 cm", flush=True)
