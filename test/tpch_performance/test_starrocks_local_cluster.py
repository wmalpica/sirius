"""Verify multi-CN isolation without requiring FE binaries, CUDA, or network services."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

HELPER = (
    Path(__file__).resolve().parents[2]
    / "experimental/starrocks/integration/local_cluster.py"
)
SPEC = importlib.util.spec_from_file_location("starrocks_local_cluster", HELPER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
LocalCluster = MODULE.LocalCluster


class LocalClusterTests(unittest.TestCase):
    def test_each_node_has_its_device_config_ports_and_fragment_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            output.mkdir()
            config = root / "input.yaml"
            config.write_text(
                "sirius:\n  memory:\n    host: {capacity_bytes: 40GiB}\n  telemetry:\n    enable_quent: true\n    output_directory: quent\n"
            )
            workspace = root / "prototype"
            package = workspace / "starrocks/output/fe"
            (package / "bin").mkdir(parents=True)
            (package / "conf").mkdir()
            (package / "conf/fe.conf").write_text("run_mode=shared_data\n")
            cluster = LocalCluster(output, config, ["MIG-first", "MIG-second"])
            cluster.workspace = workspace
            process = MagicMock()
            process.poll.return_value = None
            probe = MagicMock()
            probe.__enter__.return_value.connect_ex.return_value = 1
            with patch.object(
                MODULE.socket, "socket", return_value=probe
            ), patch.object(
                MODULE.subprocess, "Popen", return_value=process
            ) as launch, patch.object(
                MODULE.os, "killpg"
            ):
                with cluster:
                    self.assertEqual(launch.call_count, 3)
                    cn_calls = launch.call_args_list[1:]
                    self.assertEqual(
                        [
                            call.kwargs["env"]["CUDA_VISIBLE_DEVICES"]
                            for call in cn_calls
                        ],
                        ["MIG-first", "MIG-second"],
                    )
                    for index in (0, 1):
                        settings = yaml.safe_load(
                            (output / f"cn{index}/sirius.yaml").read_text()
                        )
                        self.assertEqual(
                            settings["sirius"]["memory"]["host"]["capacity_bytes"],
                            "40GiB",
                        )
                        self.assertEqual(
                            settings["sirius"]["telemetry"]["output_directory"],
                            str(output / f"quent/cn{index}"),
                        )
                        (output / f"cn{index}/fragments/same-name.txt").write_text(
                            "fragment"
                        )
                    self.assertEqual(
                        cluster.fragment_files(),
                        {"cn0/fragments/same-name.txt", "cn1/fragments/same-name.txt"},
                    )
                    nodes = json.loads((output / "cluster.json").read_text())
                    self.assertTrue(
                        set(nodes[0]["ports"].values()).isdisjoint(
                            nodes[1]["ports"].values()
                        )
                    )

    def test_duplicate_device_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            LocalCluster("unused", "unused", ["0", "0"])


if __name__ == "__main__":
    unittest.main()
