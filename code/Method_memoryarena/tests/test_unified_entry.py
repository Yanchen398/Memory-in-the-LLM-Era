import tempfile
import unittest
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from Method_memoryarena.run import (
    infer_task,
    main,
    prepare_config,
    resolve_method,
    runner_command,
)


class UnifiedEntryTests(unittest.TestCase):
    def test_infer_task_supports_physics_config(self):
        self.assertEqual(infer_task({"task_name": "phys"}), "physics")

    def test_method_alias_is_canonicalized(self):
        self.assertEqual(resolve_method("mem0-g"), "mem0g")

    def test_prepare_config_synchronizes_memory_url_and_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            config = prepare_config(
                {"task_name": "travel", "memory": {}, "output": {}},
                method="zep",
                task="travel",
                memory_url="http://127.0.0.1:8000/",
                output_root=output_root,
            )

            self.assertEqual(
                config["memory"],
                {
                    "memory_system_name": "zep",
                    "server_url": "http://127.0.0.1:8000",
                    "memory_url": "http://127.0.0.1:8000",
                    "base_url": "http://127.0.0.1:8000",
                },
            )
            self.assertEqual(
                config["output"]["output_dir"],
                str(output_root.resolve() / "travel" / "zep"),
            )
            self.assertEqual(
                config["output"]["log_dir"],
                str(output_root.resolve() / "travel" / "zep" / "logs"),
            )

    def test_runner_command_uses_absolute_existing_runner(self):
        command = runner_command("search", Path("config.json"))
        self.assertTrue(Path(command[1]).is_absolute())
        self.assertEqual(Path(command[1]).name, "run_search.py")
        self.assertEqual(command[-2], "--config")

    def test_main_dry_run_dispatches_without_starting_runner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            config_path = temp_root / "travel.json"
            with config_path.open("w", encoding="utf-8") as handle:
                json.dump({"task_name": "travel", "memory": {}, "output": {}}, handle)
            output = StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "run",
                        "--config",
                        str(config_path),
                        "--method",
                        "mem0-g",
                        "--output-root",
                        str(temp_root / "results"),
                        "--dry-run",
                    ]
                )

            self.assertEqual(status, 0)
            self.assertIn("task=travel method=mem0g", output.getvalue())
            self.assertIn("run_travel.py", output.getvalue())


if __name__ == "__main__":
    unittest.main()
