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
        torch.save({
            \"model_state_dict\": runner.alg.policy.state_dict(),
            \"optimizer_state_dict\": runner.alg.optimizer.state_dict(),
            \"iter\": 0,
            \"infos\": {\"phase\": \"behavior_cloning\"},
        }, os.path.join(log_dir, \"model_0.pt\"))

""" + training_marker,
    1,
)
namespace = {"__name__": "__main__", "__file__": str(script)}
exec(compile(source, str(script), "exec"), namespace)
