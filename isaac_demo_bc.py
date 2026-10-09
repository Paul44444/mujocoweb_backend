"""Behavior-cloning warm start for Isaac Lab's RSL-RL actor."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def demonstration_phase(env, duration: float):
    result = torch.zeros((env.num_envs, 8), device=env.device)
    result[:, 0] = (env.episode_length_buf.float() * env.step_dt / duration).clamp(0, 1)
    return result


def longest_demonstration_seconds(demonstration_paths: list[str], fallback_step_dt: float) -> float:
    """Return the longest recorded demo duration, preferring its saved metadata."""
    longest = 0.0
    for raw_path in demonstration_paths:
        path = Path(raw_path)
        try:
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            duration = float(metadata.get("duration", 0.0))
        except (OSError, ValueError, TypeError):
            with np.load(path) as payload:
                duration = len(payload["actions"]) * fallback_step_dt
        longest = max(longest, duration)
    return longest


def pretrain_runner_from_demonstrations(
    runner,
    demonstration_paths: list[str],
    epochs: int,
    learning_rate: float = 3.0e-4,
    batch_size: int = 64,
) -> None:
    """Fit the existing RSL-RL actor to web-recorded observation/action pairs."""
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    duration = longest_demonstration_seconds(demonstration_paths, runner.env.unwrapped.step_dt)
    for raw_path in demonstration_paths:
        path = Path(raw_path)
        with np.load(path) as payload:
            demo_observations = np.asarray(payload["observations"], dtype=np.float32)
            demo_observations = demo_observations.copy()
            demo_observations[:, -8:] = 0.0
            demo_observations[:, -8] = np.arange(len(demo_observations)) * runner.env.unwrapped.step_dt / duration
            demo_actions = np.asarray(payload["actions"], dtype=np.float32)
            demo_actions = demo_actions.copy()
        if demo_observations.ndim != 2 or demo_actions.ndim != 2:
            raise ValueError(f"Demonstration {path.name} does not contain matrix observations/actions")
        if len(demo_observations) != len(demo_actions) or not len(demo_actions):
            raise ValueError(f"Demonstration {path.name} has mismatched or empty samples")
        observations.append(demo_observations)
        actions.append(demo_actions)

    observation_tensor = torch.from_numpy(np.concatenate(observations)).float()
    action_tensor = torch.from_numpy(np.concatenate(actions)).float()
    device = torch.device(runner.device)
    actor = runner.alg.policy.actor
    actor.train()

    with torch.no_grad():
        probe = actor(observation_tensor[:1].to(device))
    if probe.shape[-1] != action_tensor.shape[-1]:
        raise ValueError(
            f"Demonstration action size {action_tensor.shape[-1]} does not match policy size {probe.shape[-1]}"
        )

    loader = DataLoader(
        TensorDataset(observation_tensor, action_tensor),
        batch_size=min(batch_size, len(observation_tensor)),
        shuffle=True,
    )
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    best_state = None
    best_error = float("inf")
    print(
        f"Behavior cloning: {len(demonstration_paths)} demonstration(s), "
        f"{len(observation_tensor)} samples, {epochs} epochs",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_observations, batch_actions in loader:
            batch_observations = batch_observations.to(device)
            batch_actions = batch_actions.to(device)
            batch_observations = batch_observations.clone()
            batch_actions = batch_actions.clone()
            position_noise = torch.randn_like(batch_observations[:, :7]) * 0.003
            batch_observations[:, :7] += position_noise
            prediction = actor(batch_observations)
            # Report/optimize squared joint-action error and the binary
            # gripper separately; averaging a tiny gripper error into eight
            # channels can conceal a sign error that prevents grasping.
            loss = 10.0 * torch.nn.functional.mse_loss(prediction[:, :7], batch_actions[:, :7])
            loss = loss + torch.nn.functional.mse_loss(prediction[:, 7:], batch_actions[:, 7:])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_observations)
        mean_loss = total_loss / len(observation_tensor)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 20) == 0:
            print(f"BC epoch {epoch}/{epochs} loss: {mean_loss:.8f}", flush=True)
        # Train on states the policy actually visits, with the demonstration's
        # action at the same phase as the supervised target. This closes the
        # teacher-forcing gap without any task reward/PPO updates.
        if epoch < epochs and epoch % max(1, epochs // 5) == 0:
            rollout_observations, rollout_actions = validate_demonstration_rollout(
                runner, demonstration_paths[0], collect=True
            )
            evaluation = runner.web_bc_validation
            if evaluation["lifted"] and evaluation["ee_rmse_m"] < best_error:
                best_error = evaluation["ee_rmse_m"]
                best_state = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
            observation_tensor = torch.cat((observation_tensor, rollout_observations))
            action_tensor = torch.cat((action_tensor, rollout_actions))
            loader = DataLoader(TensorDataset(observation_tensor, action_tensor), batch_size=batch_size, shuffle=True)
            print(f"BC on-policy correction: {len(rollout_actions)} additional visited states", flush=True)
    print("Behavior cloning completed; continuing with PPO fine-tuning", flush=True)
    with torch.no_grad():
        prediction = actor(observation_tensor.to(device))
        error = prediction - action_tensor.to(device)
        print(
            f"BC validation: joint target RMSE={float(error[:, :7].square().mean().sqrt()) * 0.5:.6f} rad, "
            f"max={float(error[:, :7].abs().max()) * 0.5:.6f} rad, "
            f"gripper sign accuracy={float(((prediction[:, 7] >= 0) == (action_tensor[:, 7].to(device) >= 0)).float().mean()):.4f}",
            flush=True,
        )
    validate_demonstration_rollout(runner, demonstration_paths[0], replay=True)
    validate_demonstration_rollout(runner, demonstration_paths[0])
    if best_state is not None and (
        not runner.web_bc_validation["lifted"] or runner.web_bc_validation["ee_rmse_m"] > best_error
    ):
        actor.load_state_dict(best_state)
        print("BC selection: restored the best successfully lifting intermediate policy", flush=True)
        validate_demonstration_rollout(runner, demonstration_paths[0])
    with torch.no_grad():
        if hasattr(runner.alg.policy, "log_std"):
            runner.alg.policy.log_std.fill_(float(np.log(0.03)))
        elif hasattr(runner.alg.policy, "std"):
            runner.alg.policy.std.fill_(0.03)
    runner.web_demo_dataset = (observation_tensor, action_tensor)


def validate_demonstration_rollout(runner, demonstration_path: str, replay: bool = False, collect: bool = False):
    """Evaluate the learned actor in physics from the recorded start (env 0)."""
    env = runner.env.unwrapped
    device = env.device
    with np.load(demonstration_path) as payload:
        saved = {key: np.asarray(payload[key]).copy() for key in payload.files}
    if "initial_robot_joint_pos" not in saved:
        print("BC rollout validation skipped: demo has no saved initial state", flush=True)
        return
    env.reset()
    # Parallel training places env_0 on a grid; the one-environment web demo
    # used an origin at zero. Translate all saved world coordinates together.
    origin = env.scene.env_origins[0].detach().cpu().numpy()
    for key in ("initial_robot_root_pose", "initial_object_root_pose"):
        saved[key][:3] += origin
    saved["ee_positions"] += origin
    ids = torch.tensor([0], device=device, dtype=torch.long)
    robot, obj = env.scene["robot"], env.scene["object"]
    def value(key):
        return torch.as_tensor(saved[key], device=device, dtype=torch.float32).reshape(1, -1)
    robot.write_root_pose_to_sim(value("initial_robot_root_pose"), env_ids=ids)
    robot.write_root_velocity_to_sim(value("initial_robot_root_velocity"), env_ids=ids)
    robot.write_joint_state_to_sim(value("initial_robot_joint_pos"), value("initial_robot_joint_vel"), env_ids=ids)
    obj.write_root_pose_to_sim(value("initial_object_root_pose"), env_ids=ids)
    obj.write_root_velocity_to_sim(value("initial_object_root_velocity"), env_ids=ids)
    env.action_manager.action[0] = torch.as_tensor(saved["observations"][0, -8:], device=device)
    env.action_manager.prev_action[0] = env.action_manager.action[0]
    if "initial_task_target_pose" in saved:
        term = env.command_manager.get_term("object_pose")
        term.command[0] = value("initial_task_target_pose")[0]
        term.time_left[0] = 86400.0
    env.sim.forward()
    env.scene.update(0.0)
    observations = env.observation_manager.compute()
    actor = runner.alg.policy.actor
    hand_id = robot.body_names.index("panda_hand")
    errors = []
    visited_observations = []
    target_actions = []
    peak_height = float(obj.data.root_pos_w[0, 2])
    with torch.no_grad():
        for step in range(len(saved["actions"])):
            obs = observations["policy"] if isinstance(observations, dict) else observations
            if collect:
                visited_observations.append(obs[0].detach().cpu().clone())
                target_actions.append(torch.as_tensor(saved["actions"][step], dtype=torch.float32))
            errors.append(float(torch.linalg.vector_norm(robot.data.body_pos_w[0, hand_id] - torch.as_tensor(saved["ee_positions"][step], device=device))))
            actions = actor(obs)
            if replay:
                actions[0] = torch.as_tensor(saved["actions"][step], device=device)
            observations, _, terminated, truncated, _ = env.step(actions)
            peak_height = max(peak_height, float(obj.data.root_pos_w[0, 2]))
            if bool(terminated[0]) or bool(truncated[0]):
                break
    result = {"steps": len(errors), "demo_steps": len(saved["actions"]), "ee_rmse_m": float(np.sqrt(np.mean(np.square(errors)))), "ee_max_error_m": max(errors), "peak_object_height_m": peak_height, "lifted": peak_height > 0.12}
    if not replay:
        runner.web_bc_validation = result
    print(("Demo replay validation: " if replay else "BC rollout validation: ") + json.dumps(result), flush=True)
    if runner.log_dir:
        (Path(runner.log_dir) / ("demo_replay_validation.json" if replay else "bc_rollout_validation.json")).write_text(json.dumps(result, indent=2))
    env.reset()
    if collect:
        return torch.stack(visited_observations), torch.stack(target_actions)
