"""Launch the official Isaac Lab RSL-RL trainer with web vision tasks registered."""

from pathlib import Path
import sys

script = Path("/home/paul/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py")
sys.path.insert(0, str(script.parent))
source = script.read_text(encoding="utf-8")
marker = "import isaaclab_tasks  # noqa: F401"
if marker not in source:
    raise RuntimeError("Isaac Lab train.py registration marker changed")
source = source.replace(
    marker,
    marker + "\nimport isaac_vision_task  # register web vision environments"
    + "\nimport isaac_labware_task  # register web labware environments",
    1,
)
namespace = {"__name__": "__main__", "__file__": str(script)}
exec(compile(source, str(script), "exec"), namespace)
