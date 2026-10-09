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


def install_demo_guided_ppo(runner, demonstration_paths, total_iterations, resume_state=None):
    """Add BC gradients to each PPO minibatch using the same optimizer step."""
    env = runner.env.unwrapped
    alg = runner.alg
    observations, actions = runner.web_demo_dataset
    observations, actions = observations.to(runner.device), actions.to(runner.device)
    state = dict(resume_state or {})
    state.setdefault("iteration", 0)
    state.setdefault("curriculum_iterations", total_iterations)
    state.setdefault("warmup_iterations", min(20, max(1, total_iterations // 10)))
    state["losses"] = []
    horizon = state["curriculum_iterations"]
    warmup = state["warmup_iterations"]
    runner.web_demo_ppo_state = state
    env.web_demo_variation = curriculum_settings(state["iteration"], horizon)[0]
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
            _, weight = curriculum_settings(state["iteration"], horizon)
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
        variation, weight = curriculum_settings(state["iteration"], horizon)
        state["variation_cm"] = variation * 100
        state["bc_weight"] = weight
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
    print(f"Demo-guided PPO enabled: cumulative iteration {state['iteration']}; remaining critic warmup {max(0, warmup - state['iteration'])}; XY variation +/-{env.web_demo_variation * 100:.2f} cm", flush=True)


def prepare_resumed_demo_dataset(runner, paths, duration, saved_dataset=None):
    """Recover the exact BC correction dataset, or rebuild it for legacy files."""
    if saved_dataset is not None:
        runner.web_demo_dataset = (saved_dataset["observations"], saved_dataset["actions"])
        return
    observations, actions = [], []
    for path in paths:
        with np.load(path) as data:
            obs = np.asarray(data["observations"], dtype=np.float32).copy()
            obs[:, -8:] = 0.0
            obs[:, -8] = np.arange(len(obs)) * runner.env.unwrapped.step_dt / duration
            observations.append(obs)
            actions.append(np.asarray(data["actions"], dtype=np.float32).copy())
    runner.web_demo_dataset = (torch.from_numpy(np.concatenate(observations)), torch.from_numpy(np.concatenate(actions)))


def checkpoint_demo_state(runner, paths, duration):
    state = {key: value for key, value in runner.web_demo_ppo_state.items() if key != "losses"}
    variation, weight = curriculum_settings(state["iteration"], state["curriculum_iterations"])
    state.update(variation_cm=variation * 100, bc_weight=weight)
    obs, actions = runner.web_demo_dataset
    return {"demo_control": True, "demo_duration": duration, "demo_guided_ppo": True,
            "demo_paths": list(paths), "demo_ppo_state": state,
            "demo_dataset": {"observations": obs.detach().cpu(), "actions": actions.detach().cpu()}}
