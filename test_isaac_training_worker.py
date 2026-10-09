"""Lightweight supervisor regressions; run with the Python 3.8 API interpreter."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import isaac_training_worker as worker


class TrainingWorkerTests(unittest.TestCase):
    def test_checkpoint_sort_is_python38_compatible(self):
        source = Mock()
        source.glob.return_value = []
        source.__truediv__ = Mock(return_value=Mock(is_file=Mock(return_value=False)))
        destination = Mock()
        destination.glob.return_value = [Path(name) for name in (
            "model_100.pt", "model_bc.pt", "model_0.pt", "model_50.pt"
        )]
        self.assertEqual(worker.sync_checkpoints(source, destination), [
            "model_bc.pt", "model_0.pt", "model_50.pt", "model_100.pt"
        ])

    def test_supervisor_records_all_1000_iterations_after_bc(self):
        output = ["BC epoch 500/500 loss: 0.001\n"]
        for index in range(1000):
            output.extend([
                "Learning iteration %d/1000\n" % index,
                "Mean reward: 20.0\n",
                "Total timesteps: %d\n" % ((index + 1) * 12288),
            ])
        child = Mock(pid=12345, stdout=iter(output))
        child.wait.return_value = 0
        published = {}
        def publish(path, payload):
            published[path.name] = payload
        arguments = ["worker", "--run-directory", "/tmp/worker-regression",
                     "--iterations", "1000", "--num-envs", "512", "--seed", "123",
                     "--isaac-task", "labware_lift"]
        with patch("sys.argv", arguments), patch.object(Path, "mkdir"), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", return_value="{}"), \
             patch.object(worker, "atomic_json", side_effect=publish), \
             patch.object(worker, "find_output_directory", return_value=Path("/tmp/output")), \
             patch.object(worker, "sync_checkpoints", return_value=["model_bc.pt", "model_999.pt"]), \
             patch.object(worker.subprocess, "Popen", return_value=child), \
             patch.object(worker.signal, "signal"), contextlib.redirect_stdout(io.StringIO()):
            worker.main()
            command = worker.subprocess.Popen.call_args[0][0]
            self.assertEqual(command[command.index("--max_iterations") + 1], "1000")
        self.assertEqual(published["status.json"]["status"], "completed")
        self.assertEqual(published["status.json"]["iteration"], 1000)
        self.assertEqual(len(published["metrics.json"]["metrics"]), 1000)
        self.assertEqual(published["metrics.json"]["bc_metrics"][0]["epoch"], 500)


if __name__ == "__main__":
    unittest.main()
