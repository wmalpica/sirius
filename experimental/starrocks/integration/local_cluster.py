"""A private FE with one Sirius CN per explicitly selected CUDA/MIG device."""

import copy
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess

import yaml


class LocalCluster:
    def __init__(self, output, config, devices=("0",)):
        self.output = Path(output).resolve()
        self.config = Path(config).resolve()
        self.devices = list(devices)
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError("Select at least one distinct CUDA/MIG device")
        self.workspace = Path(__file__).resolve().parents[1]
        self.processes = []
        self.handles = []
        self.node_roots = [
            self.output if len(self.devices) == 1 else self.output / f"cn{i}"
            for i in range(len(self.devices))
        ]
        self.env = dict(
            os.environ, CUDA_VISIBLE_DEVICES=",".join(self.devices), NO_COLOR="1"
        )
        for key in ("SIRIUS_CN_TRANSLATE_ONLY", "SIRIUS_DISABLE"):
            self.env.pop(key, None)

    @staticmethod
    def ports(index):
        offset = index * 100
        return {
            "heartbeat": 9050 + offset,
            "thrift": 9060 + offset,
            "http": 8040 + offset,
            "brpc": 8060 + offset,
            "starlet": 9070 + offset,
        }

    def fragment_files(self):
        return {
            str(path.relative_to(self.output))
            for root in self.node_roots
            for path in (root / "fragments").iterdir()
        }

    def __enter__(self):
        ports = {8030, 9010, 9020, 9030}
        for index in range(len(self.devices)):
            ports.update(self.ports(index).values())
        for port in ports:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(
                        f"Port {port} is occupied; refusing to reuse a cluster"
                    )
        package = self.workspace / "starrocks/output/fe"
        home = self.output / "fe"
        home.mkdir()
        for name in ("bin", "conf"):
            shutil.copytree(package / name, home / name)
        for name in ("lib", "webroot", "hive-udf", "spark-dpp"):
            (home / name).symlink_to(package / name, target_is_directory=True)
        for name in ("log", "meta", "temp_dir", "plugins"):
            (home / name).mkdir()
        with (home / "conf/fe.conf").open("a") as handle:
            handle.write(
                f"\nmeta_dir = {home}/meta\nsys_log_dir = {home}/log\naudit_log_dir = {home}/log\n"
            )
        manifest = str(self.workspace / "pixi.toml")
        commands = [
            (
                "fe",
                [
                    "pixi",
                    "run",
                    "--manifest-path",
                    manifest,
                    "-e",
                    "fe",
                    "bash",
                    str(home / "bin/start_fe.sh"),
                    "--logconsole",
                ],
                self.env,
            )
        ]
        base_config = yaml.safe_load(self.config.read_text())
        records = []
        for index, (device, root) in enumerate(zip(self.devices, self.node_roots)):
            root.mkdir(exist_ok=True)
            for name in ("fragments", "engine", "tmp"):
                (root / name).mkdir(exist_ok=name == "tmp")
            config = self.config
            if len(self.devices) > 1:
                settings = copy.deepcopy(base_config)
                telemetry = settings.setdefault("sirius", {}).setdefault(
                    "telemetry", {}
                )
                directory = Path(telemetry.get("output_directory", "telemetry_data"))
                if not directory.is_absolute():
                    directory = self.output / directory
                telemetry["output_directory"] = str(directory / f"cn{index}")
                config = root / "sirius.yaml"
                config.write_text(yaml.safe_dump(settings, sort_keys=False))
            env = dict(
                self.env,
                CUDA_VISIBLE_DEVICES=device,
                SIRIUS_CN_DUMP_FRAGMENTS=str(root / "fragments"),
                SIRIUS_LOG_DIR=str(root / "engine"),
                TMPDIR=str(root / "tmp"),
            )
            ports = self.ports(index)
            argv = [str(config)]
            for key, value in ports.items():
                argv.extend([f"--{key}-port", str(value)])
            name = "cn" if len(self.devices) == 1 else f"cn{index}"
            command = [
                "pixi",
                "run",
                "--manifest-path",
                manifest,
                "bash",
                "-c",
                'source scripts/cn-env.sh && exec target/release/sirius-starrocks-cn --sirius-config "$@"',
                "cn",
                *argv,
            ]
            commands.append((name, command, env))
            records.append(
                dict(
                    name=name,
                    cuda_visible_devices=device,
                    config=str(config),
                    ports=ports,
                    cpu_affinity=sorted(os.sched_getaffinity(0)),
                )
            )
        (self.output / "cluster.json").write_text(json.dumps(records, indent=2) + "\n")
        try:
            for name, command, env in commands:
                handle = (self.output / f"{name}.log").open("w")
                self.handles.append(handle)
                self.processes.append(
                    subprocess.Popen(
                        command,
                        cwd=self.workspace,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def check(self):
        if any(process.poll() is not None for process in self.processes):
            raise RuntimeError("A cluster process exited; see FE/CN logs")

    def __exit__(self, *_):
        for process in reversed(self.processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for handle in self.handles:
            handle.close()
