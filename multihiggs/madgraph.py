from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from .config import ProjectConfig
from .grid import ScanPoint
from .results import event_count


ZERO_AMPLITUDE_MARKER = "Problem in the multi-channeling. All amp2 are zero but not the total matrix-element"


@dataclass(frozen=True)
class MadEventFailure:
    point: ScanPoint
    reason: str
    logs: tuple[Path, ...]


def mg_runtime_env(mg5_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    lib_paths = [
        mg5_path / "HEPTools" / "lib",
        mg5_path / "HEPTools" / "collier",
        mg5_path / "HEPTools" / "collier" / "COLLIER-1.2.9",
    ]
    paths = [
        str(path)
        for path in lib_paths
        if (path / "libcollier.so").exists() or path.exists()
    ]
    variables = ["LD_LIBRARY_PATH"]
    if sys.platform == "darwin":
        variables.append("DYLD_LIBRARY_PATH")
    for variable in variables:
        existing = list(paths)
        current = env.get(variable)
        if current:
            existing.append(current)
        if existing:
            env[variable] = ":".join(existing)
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
    mg5_path = config.mg5_path.resolve()
    command = [str(mg5_path / "bin" / "mg5_aMC"), str(process_card.resolve())]
    return _stream_subprocess(command, cwd=mg5_path, env=mg_runtime_env(mg5_path))


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
    if config.scan.no_cuts:
        lines.extend(config.scan.no_cut_commands)
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
    low_stat_existing: list[tuple[str, int]] = []
    lines: list[str] = []
    for point in points:
        if max_runs is not None and len(selected) >= max_runs:
            break
        if not force and config.scan.skip_existing:
            lhe_file = event_file(config, point)
            if lhe_file.exists():
                event_minimum = config.scan.event_minimum
                if event_minimum <= 0:
                    continue
                n_events = event_count(lhe_file)
                if n_events >= event_minimum:
                    continue
                low_stat_existing.append((point.run_name(config), n_events))
        selected.append(point)
        lines.extend(build_launch_block(config, point))
    path.write_text("\n".join(lines), encoding="utf-8")
    if low_stat_existing:
        event_minimum = config.scan.event_minimum
        print(
            "Warning: "
            f"{len(low_stat_existing)} existing run(s) have fewer than {event_minimum} events; "
            "scheduling them again"
        )
        for run_name, n_events in low_stat_existing[:10]:
            print(f"  {run_name}: {n_events} events")
        if len(low_stat_existing) > 10:
            print(f"  ... {len(low_stat_existing) - 10} additional run(s)")
    return path, selected


def run_madevent(config: ProjectConfig, command_file: Path) -> int:
    mg5_path = config.mg5_path.resolve()
    process_dir = config.process_dir.resolve()
    patch_macos_madloop_rpaths(process_dir)
    command = [str(process_dir / "bin" / "madevent"), str(command_file.resolve())]
    return _stream_subprocess(command, cwd=process_dir, env=mg_runtime_env(mg5_path))


def run_madevent_points(config: ProjectConfig, points: Iterable[ScanPoint], command_file: Path) -> int:
    active_card = command_file.with_name(command_file.stem + "_active.dcmd")
    failures: list[MadEventFailure] = []
    for point in points:
        _, selected = write_madevent_card(config, [point], active_card, force=True)
        if not selected:
            continue
        started_at = time.time()
        status = run_madevent(config, active_card)
        if status == 0:
            continue
        failure = diagnose_madevent_failure(config, point, since=started_at)
        if failure and failure.reason == "zero_amplitude_multichannel":
            print_zero_amplitude_warning(config, failure)
            failures.append(failure)
            continue
        return status
    if failures:
        print()
        print(
            "WARNING: "
            f"Skipped {len(failures)} MadEvent point(s) with zero-amplitude multichannel failures. "
            "The scan continued, but it is incomplete; inspect the warning(s) above before fitting."
        )
        return 2
    return 0


def diagnose_madevent_failure(
    config: ProjectConfig,
    point: ScanPoint,
    since: float | None = None,
) -> MadEventFailure | None:
    logs = find_failure_logs(config, point, ZERO_AMPLITUDE_MARKER, since=since)
    if logs:
        return MadEventFailure(point=point, reason="zero_amplitude_multichannel", logs=logs)
    return None


def print_zero_amplitude_warning(config: ProjectConfig, failure: MadEventFailure) -> None:
    point = failure.point
    print()
    print(f"WARNING: MadEvent failed for {point.run_name(config)} with a zero-amplitude multichannel issue.")
    print(
        "MadEvent reported that all channel amplitudes were zero while the total matrix element was not. "
        "This often happens at exact cancellation points where a coupling modifier removes the "
        "interaction needed for this process."
    )
    print("The pipeline is skipping this point and continuing with the remaining selected points.")
    print("Point: " + format_point(config, point))
    hint = zero_amplitude_hint(config, point)
    if hint:
        print("Possible cause: " + hint)
    if failure.logs:
        print("Relevant log(s):")
        for log in failure.logs[:5]:
            print(f"  {log}")


def zero_amplitude_hint(config: ProjectConfig, point: ScanPoint) -> str | None:
    values = {coupling.parameter: value for coupling, value in zip(config.couplings, point.values)}
    if config.model == "heft_loop_sm_restricted5" and abs(values.get("CT1", 999.0) + 1.0) < 1e-12:
        return (
            "in heft_loop_sm_restricted5, CT1 = -1 cancels the SM ttH Yukawa contribution. "
            "If CT2, D3, CT3, and D4 are also zero, ttbar+HHH can sit in a zero-amplitude corner."
        )
    return None


def format_point(config: ProjectConfig, point: ScanPoint) -> str:
    pieces = []
    for coupling, text in zip(config.couplings, point.texts):
        if coupling.name == coupling.parameter:
            pieces.append(f"{coupling.name}={text}")
        else:
            pieces.append(f"{coupling.name}={text} ({coupling.parameter})")
    return ", ".join(pieces)


def find_failure_logs(
    config: ProjectConfig,
    point: ScanPoint,
    marker: str,
    since: float | None = None,
) -> tuple[Path, ...]:
    process_dir = config.process_dir.resolve()
    run_name = point.run_name(config)
    direct_logs = [
        process_dir / f"{run_name}_tag_1_debug.log",
        process_dir / f"{run_name}_debug.log",
    ]
    matches = [path for path in direct_logs if file_contains(path, marker, since=since)]
    if matches:
        return tuple(matches)

    subproc_dir = process_dir / "SubProcesses"
    if not subproc_dir.exists():
        return ()
    for path in sorted(subproc_dir.glob("**/run*_app.log")):
        if file_contains(path, marker, since=since):
            matches.append(path)
    return tuple(matches)


def file_contains(path: Path, marker: str, since: float | None = None) -> bool:
    if not path.exists():
        return False
    if since is not None and path.stat().st_mtime < since - 1.0:
        return False
    return marker in path.read_text(encoding="utf-8", errors="ignore")


def patch_macos_madloop_rpaths(process_dir: Path, force: bool = False) -> None:
    if sys.platform != "darwin" and not force:
        return
    for makefile in (
        process_dir / "SubProcesses" / "makefile",
        process_dir / "SubProcesses" / "makefile_MadLoop",
    ):
        if not makefile.exists():
            continue
        lines = makefile.read_text(encoding="utf-8").splitlines()
        changed = False
        patched: list[str] = []
        for line in lines:
            if line.startswith("LINKLIBS =") and "$(RPATH_LIBS)" not in line:
                line = line + " $(RPATH_LIBS)"
                changed = True
            patched.append(line)
        if changed:
            makefile.write_text("\n".join(patched) + "\n", encoding="utf-8")


def _stream_subprocess(command: list[str], cwd: Path, env: dict[str, str]) -> int:
    with subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
        return proc.wait()
