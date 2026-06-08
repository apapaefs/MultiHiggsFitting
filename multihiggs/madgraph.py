from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

from .config import ProjectConfig
from .grid import ScanPoint


def mg_runtime_env(mg5_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    lib_paths = [
        mg5_path / "HEPTools" / "lib",
        mg5_path / "HEPTools" / "collier",
        mg5_path / "HEPTools" / "collier" / "COLLIER-1.2.9",
    ]
    existing = [
        str(path)
        for path in lib_paths
        if (path / "libcollier.so").exists() or path.exists()
    ]
    current = env.get("LD_LIBRARY_PATH")
    if current:
        existing.append(current)
    if existing:
        env["LD_LIBRARY_PATH"] = ":".join(existing)
    return env


def write_process_card(config: ProjectConfig, path: Path, force_output: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = config.output + (" -f" if force_output else "")
    lines: list[str] = []
    lines.extend(config.settings)
    lines.extend(config.pre_model_commands)
    lines.append(f"import model {config.model}")
    lines.extend(config.post_model_commands)
    lines.append(f"generate {config.generate}")
    lines.append(f"output {output}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_mg5(config: ProjectConfig, process_card: Path) -> int:
    command = [str(config.mg5_path / "bin" / "mg5_aMC"), str(process_card)]
    return _stream_subprocess(command, cwd=config.mg5_path, env=mg_runtime_env(config.mg5_path))


def launch_options(config: ProjectConfig) -> str:
    return config.scan.madgraph.launch_suffix()


def build_launch_block(config: ProjectConfig, point: ScanPoint) -> list[str]:
    ebeam = config.scan.energy_tev * 1000.0 / 2.0
    run_name = point.run_name(config)
    suffix = launch_options(config)
    launch_line = "launch " + run_name + ((" " + suffix) if suffix else "")
    lines = [
        launch_line,
        "0",
        f"set ebeam1 {ebeam}",
        f"set ebeam2 {ebeam}",
    ]
    for coupling, text in zip(config.couplings, point.texts):
        lines.append(f"set {coupling.parameter} {text}")
    lines.extend(config.scan.extra_set_commands)
    lines.append(f"set nevents {config.scan.nevents}")
    lines.append("0")
    lines.append("")
    return lines


def event_file(config: ProjectConfig, point: ScanPoint, run_number: str | None = None) -> Path:
    return config.process_dir / "Events" / point.run_name(config, run_number) / "unweighted_events.lhe.gz"


def write_madevent_card(
    config: ProjectConfig,
    points: Iterable[ScanPoint],
    path: Path,
    max_runs: int | None = None,
    force: bool = False,
) -> tuple[Path, list[ScanPoint]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected: list[ScanPoint] = []
    lines: list[str] = []
    for point in points:
        if max_runs is not None and len(selected) >= max_runs:
            break
        if not force and config.scan.skip_existing and event_file(config, point).exists():
            continue
        selected.append(point)
        lines.extend(build_launch_block(config, point))
    path.write_text("\n".join(lines), encoding="utf-8")
    return path, selected


def run_madevent(config: ProjectConfig, command_file: Path) -> int:
    command = [str(config.process_dir / "bin" / "madevent"), str(command_file)]
    return _stream_subprocess(command, cwd=config.process_dir, env=mg_runtime_env(config.mg5_path))


def _stream_subprocess(command: list[str], cwd: Path, env: dict[str, str]) -> int:
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    return proc.wait()
