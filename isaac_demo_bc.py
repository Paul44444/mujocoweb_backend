"""Behavior-cloning warm start for Isaac Lab's RSL-RL actor."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


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
    batch_size: int = 256,
) -> None:
    """Fit the existing RSL-RL actor to web-recorded observation/action pairs."""
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for raw_path in demonstration_paths:
        path = Path(raw_path)
        with np.load(path) as payload:
            demo_observations = np.asarray(payload["observations"], dtype=np.float32)
            demo_actions = np.asarray(payload["actions"], dtype=np.float32)
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
            prediction = actor(batch_observations)
            loss = torch.nn.functional.smooth_l1_loss(prediction, batch_actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_observations)
        mean_loss = total_loss / len(observation_tensor)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 20) == 0:
            print(f"BC epoch {epoch}/{epochs} loss: {mean_loss:.8f}", flush=True)
    print("Behavior cloning completed; continuing with PPO fine-tuning", flush=True)
