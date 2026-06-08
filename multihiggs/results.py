from __future__ import annotations

import csv
import gzip
import re
from dataclasses import dataclass
from pathlib import Path

from .config import ProjectConfig


@dataclass(frozen=True)
class RunResult:
    run_name: str
    run_number: str
    values: tuple[float, ...]
    xsec_pb: float
    xerr_pb: float
    event_file: Path
    event_count: int | None = None

    def as_dict(self, config: ProjectConfig) -> dict[str, str | float]:
        row: dict[str, str | float] = {
            "run_name": self.run_name,
            "run_number": self.run_number,
            "xsec_pb": self.xsec_pb,
            "xerr_pb": self.xerr_pb,
            "event_file": str(self.event_file),
        }
        if self.event_count is not None:
            row["event_count"] = self.event_count
        for coupling, value in zip(config.couplings, self.values):
            row[coupling.name] = value
        return row


def discover_completed_runs(
    config: ProjectConfig,
    run_numbers: set[str] | None = None,
    exclude_run_numbers: set[str] | None = None,
    min_events: int | None = None,
) -> list[RunResult]:
    event_dir = config.process_dir / "Events"
    prefix = "run_" + config.name + "_"
    event_minimum = config.scan.event_minimum if min_events is None else min_events
    results: list[RunResult] = []
    if not event_dir.exists():
        return results

    exclude_run_numbers = exclude_run_numbers or set()
    skipped_low_stats = 0
    for rundir in sorted(event_dir.glob(prefix + "*")):
        if not rundir.is_dir():
            continue
        run_name = rundir.name
        rest = run_name[len(prefix):]
        parts = rest.split("_")
        expected = 1 + len(config.couplings)
        if len(parts) != expected:
            continue
        run_number = parts[0]
        if run_numbers is not None and run_number not in run_numbers:
            continue
        if run_number in exclude_run_numbers:
            continue
        try:
            values = tuple(float(item) for item in parts[1:])
        except ValueError:
            continue
        event_file = rundir / "unweighted_events.lhe.gz"
        if not event_file.exists():
            continue
        n_events = None
        if event_minimum > 0:
            n_events = event_count(event_file)
            if n_events < event_minimum:
                skipped_low_stats += 1
                continue
        xsec = read_xsec(event_file)
        xerr = read_integration_error(config.process_dir, run_name, xsec)
        results.append(RunResult(run_name, run_number, values, xsec, xerr, event_file, n_events))
    results.sort(key=lambda item: (item.run_number, item.values))
    if event_minimum > 0:
        if skipped_low_stats:
            print(f"Warning: skipped {skipped_low_stats} run(s) with fewer than {event_minimum} events")
        if not results:
            print(f"Warning: no completed runs found with at least {event_minimum} events")
    return results


def read_xsec(event_file: Path) -> float:
    pattern = re.compile(r"Integrated weight \(pb\)\s*:\s*([0-9.eE+-]+)")
    with gzip.open(event_file, "rt", errors="ignore") as stream:
        for line in stream:
            match = pattern.search(line)
            if match is not None:
                return float(match.group(1))
    raise RuntimeError(f"No integrated weight found in {event_file}")


def event_count(event_file: Path) -> int:
    count = 0
    with gzip.open(event_file, "rt", errors="ignore") as stream:
        for line in stream:
            if line.strip() == "<event>":
                count += 1
    return count


def read_integration_error(process_dir: Path, run_name: str, xsec: float) -> float:
    html_file = process_dir / "HTML" / run_name / "results.html"
    if html_file.exists():
        html = html_file.read_text(encoding="utf-8", errors="ignore")
        match = re.search(
            r"<b>s=\s*([0-9.eE+-]+)\s*(?:&#177|&plusmn;|\u00b1)\s*([0-9.eE+-]+)\s*\(pb\)</b>",
            html,
        )
        if match is not None:
            return float(match.group(2))
    return max(abs(float(xsec)) * 0.01, 1e-30)


def write_results_csv(config: ProjectConfig, results: list[RunResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_name", "run_number"] + [c.name for c in config.couplings] + [
        "xsec_pb",
        "xerr_pb",
        "event_count",
        "event_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.as_dict(config))
    return path


def read_results_csv(path: Path, config: ProjectConfig) -> list[RunResult]:
    results: list[RunResult] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            results.append(
                RunResult(
                    run_name=str(row["run_name"]),
                    run_number=str(row["run_number"]),
                    values=tuple(float(row[coupling.name]) for coupling in config.couplings),
                    xsec_pb=float(row["xsec_pb"]),
                    xerr_pb=float(row.get("xerr_pb") or 0.0),
                    event_file=Path(row["event_file"]),
                    event_count=None if not row.get("event_count") else int(row["event_count"]),
                )
            )
    return results
