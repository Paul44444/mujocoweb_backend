"""Isolated, bounded DAPG fine-tuning worker for the web training UI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import signal
import time

import numpy as np

from mjrlpaul.algos.dapg import DAPG
from mjrlpaul.baselines.mlp_baseline import MLPBaseline
from robohive.utils import gym


ROOT = Path(__file__).resolve().parent
REFERENCE_POLICY = ROOT / "paultrain1" / "iterations" / "best_policy.pickle"
stopping = False


class PolicyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("mjrl") and not module.startswith("mjrlpaul"):
            module = module.replace("mjrl", "mjrlpaul", 1)
        return super().find_class(module, name)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def stop_worker(_signum, _frame) -> None:
    global stopping
    stopping = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--trajectories", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    run_directory = Path(args.run_directory).resolve()
    checkpoints = run_directory / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    status_path = run_directory / "status.json"
    metrics_path = run_directory / "metrics.json"
    signal.signal(signal.SIGTERM, stop_worker)
    signal.signal(signal.SIGINT, stop_worker)

    config = {
        "engine": "mujoco",
        "environment": "relocate-v1",
        "algorithm": "DAPG fine-tuning",
        "reference_policy": str(REFERENCE_POLICY),
        "iterations": args.iterations,
        "trajectories": args.trajectories,
        "horizon": args.horizon,
        "seed": args.seed,
    }
    atomic_json(run_directory / "config.json", config)
    metrics = []
    started_at = time.time()
    atomic_json(status_path, {"status": "starting", "started_at": started_at, "iteration": 0})

    try:
        if not REFERENCE_POLICY.is_file():
            raise FileNotFoundError(f"Reference policy not found: {REFERENCE_POLICY}")
        np.random.seed(args.seed)
        wrapped_env = gym.make("relocate-v1")
        env = wrapped_env.unwrapped
        env.seed(args.seed)
        with REFERENCE_POLICY.open("rb") as policy_file:
            policy = PolicyUnpickler(policy_file).load()
        baseline = MLPBaseline(
            env,
            env.spec,
            reg_coef=1e-3,
            # The legacy optimizer drops the final minibatch and requires at
            # least two full batches. Web smoke runs intentionally use short
            # trajectories, so keep this baseline batch small.
            batch_size=2,
            epochs=2,
            learn_rate=1e-3,
        )
        agent = DAPG(
            env,
            policy,
            baseline,
            demo_paths=None,
            normalized_step_size=0.05,
            lam_0=0.0,
            lam_1=0.0,
            seed=args.seed,
            save_logs=True,
        )
        atomic_json(status_path, {"status": "training", "started_at": started_at, "iteration": 0})

        for iteration in range(args.iterations):
            if stopping:
                break
            iteration_started = time.time()
            stats = agent.train_step(
                N=args.trajectories,
                env=env,
                sample_mode="trajectories",
                horizon=args.horizon,
                gamma=0.995,
                gae_lambda=0.97,
                num_cpu=1,
            )
            log = agent.logger.get_current_log()
            metric = {
                "iteration": iteration + 1,
                "reward_mean": float(stats[0]),
                "reward_std": float(stats[1]),
                "reward_min": float(stats[2]),
                "reward_max": float(stats[3]),
                "kl_distance": float(log.get("kl_dist", 0.0)),
                "surrogate_improvement": float(log.get("surr_improvement", 0.0)),
                "vf_error_before": float(log.get("VF_error_before", 0.0)),
                "vf_error_after": float(log.get("VF_error_after", 0.0)),
                "success_rate": float(log.get("success_rate", log.get("success_percentage", 0.0))),
                "samples": int(log.get("num_samples", 0)),
                "seconds": round(time.time() - iteration_started, 3),
            }
            metrics.append(metric)
            atomic_json(metrics_path, {"metrics": metrics})
            checkpoint_path = checkpoints / f"policy_{iteration + 1:04d}.pickle"
            with checkpoint_path.open("wb") as checkpoint_file:
                pickle.dump(agent.policy, checkpoint_file)
            with (checkpoints / f"baseline_{iteration + 1:04d}.pickle").open("wb") as baseline_file:
                pickle.dump(agent.baseline, baseline_file)
            atomic_json(status_path, {
                "status": "training",
                "started_at": started_at,
                "iteration": iteration + 1,
                "latest_checkpoint": checkpoint_path.name,
            })

        final_status = "stopped" if stopping else "completed"
        atomic_json(status_path, {
            "status": final_status,
            "started_at": started_at,
            "finished_at": time.time(),
            "iteration": len(metrics),
            "latest_checkpoint": metrics and f"policy_{len(metrics):04d}.pickle" or None,
        })
    except Exception as exc:
        atomic_json(status_path, {
            "status": "failed",
            "started_at": started_at,
            "finished_at": time.time(),
            "iteration": len(metrics),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


if __name__ == "__main__":
    main()
