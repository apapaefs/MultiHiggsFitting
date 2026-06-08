from __future__ import annotations

import csv
import gzip
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ObservableConfig, ProjectConfig
from .results import RunResult, discover_completed_runs, read_xsec


@dataclass(frozen=True)
class Particle:
    pid: int
    status: int
    px: float
    py: float
    pz: float
    energy: float
    mass: float

    @property
    def pt(self) -> float:
        return math.hypot(self.px, self.py)


def write_histogram_csv(
    config: ProjectConfig,
    output_path: Path,
    observable_names: set[str] | None = None,
    run_numbers: set[str] | None = None,
    min_events: int | None = None,
) -> Path:
    observables = [
        observable
        for observable in config.observables
        if observable_names is None or observable.name in observable_names
    ]
    if not observables:
        raise ValueError("No observables selected. Add [[observables]] entries to the config.")

    runs = discover_completed_runs(config, run_numbers=run_numbers, min_events=min_events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["run_name", "run_number"]
        + [coupling.name for coupling in config.couplings]
        + [
            "observable",
            "bin_index",
            "bin_low",
            "bin_high",
            "bin_xsec_pb",
            "bin_error_pb",
            "event_count",
            "event_file",
        ]
    )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for observable in observables:
                hist, err, n_events = histogram_lhe_with_count(run.event_file, observable)
                for index, (value, error) in enumerate(zip(hist, err)):
                    row = _run_row(config, run)
                    row.update(
                        {
                            "observable": observable.name,
                            "bin_index": index,
                            "bin_low": observable.bins[index],
                            "bin_high": observable.bins[index + 1],
                            "bin_xsec_pb": value,
                            "bin_error_pb": error,
                            "event_count": n_events,
                            "event_file": str(run.event_file),
                        }
                    )
                    writer.writerow(row)
    return output_path


def histogram_lhe(event_file: Path, observable: ObservableConfig) -> tuple[np.ndarray, np.ndarray]:
    hist, err, _ = histogram_lhe_with_count(event_file, observable)
    return hist, err


def histogram_lhe_with_count(
    event_file: Path,
    observable: ObservableConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    events = list(iter_lhe_events(event_file))
    if not events:
        zeros = np.zeros(len(observable.bins) - 1, dtype=float)
        return zeros, zeros, 0
    xsec = read_xsec(event_file)
    event_weight = xsec / len(events)
    hist = np.zeros(len(observable.bins) - 1, dtype=float)
    sumw2 = np.zeros(len(observable.bins) - 1, dtype=float)
    bins = np.asarray(observable.bins, dtype=float)
    for particles in events:
        for value in observable_values(particles, observable):
            index = int(np.searchsorted(bins, value, side="right") - 1)
            if index < 0 or index >= len(hist):
                continue
            hist[index] += event_weight
            sumw2[index] += event_weight * event_weight
    errors = np.sqrt(sumw2)
    errors = np.where(errors > 0.0, errors, abs(event_weight))
    return hist, errors, len(events)


def iter_lhe_events(event_file: Path):
    with gzip.open(event_file, "rt", errors="ignore") as stream:
        in_event = False
        remaining = 0
        particles: list[Particle] = []
        for line in stream:
            stripped = line.strip()
            if stripped == "<event>":
                in_event = True
                remaining = -1
                particles = []
                continue
            if not in_event:
                continue
            if stripped == "</event>":
                in_event = False
                yield particles
                continue
            if not stripped or stripped.startswith("#"):
                continue
            if remaining == -1:
                remaining = int(stripped.split()[0])
                continue
            if remaining > 0:
                pieces = stripped.split()
                particles.append(
                    Particle(
                        pid=int(pieces[0]),
                        status=int(pieces[1]),
                        px=float(pieces[6]),
                        py=float(pieces[7]),
                        pz=float(pieces[8]),
                        energy=float(pieces[9]),
                        mass=float(pieces[10]),
                    )
                )
                remaining -= 1


def observable_values(particles: list[Particle], observable: ObservableConfig) -> list[float]:
    if observable.kind == "pt":
        if observable.pdg_id is None:
            raise ValueError(f"Observable {observable.name} must define pdg_id for kind='pt'")
        selected = [p for p in particles if p.status == 1 and abs(p.pid) == abs(observable.pdg_id)]
        values = [p.pt for p in selected]
        return select_values(values, observable.which)
    if observable.kind == "invariant_mass":
        return invariant_mass_values(particles, observable)
    raise ValueError(f"Unsupported observable kind: {observable.kind}")


def invariant_mass_values(particles: list[Particle], observable: ObservableConfig) -> list[float]:
    if not observable.pdg_ids:
        raise ValueError(f"Observable {observable.name} must define pdg_ids for invariant_mass")
    selected_groups = []
    for pid in observable.pdg_ids:
        selected_groups.append([p for p in particles if p.status == 1 and abs(p.pid) == abs(pid)])
    values = []
    if len(set(abs(pid) for pid in observable.pdg_ids)) == 1:
        for combo in itertools.combinations(selected_groups[0], len(observable.pdg_ids)):
            values.append(invariant_mass(combo))
    else:
        for combo in itertools.product(*selected_groups):
            if len(set(id(p) for p in combo)) == len(combo):
                values.append(invariant_mass(combo))
    return select_values(values, observable.which)


def invariant_mass(particles: tuple[Particle, ...] | list[Particle]) -> float:
    energy = sum(p.energy for p in particles)
    px = sum(p.px for p in particles)
    py = sum(p.py for p in particles)
    pz = sum(p.pz for p in particles)
    mass2 = energy * energy - px * px - py * py - pz * pz
    return math.sqrt(max(mass2, 0.0))


def select_values(values: list[float], which: str) -> list[float]:
    if which == "all":
        return values
    if not values:
        return []
    if which == "leading":
        return [max(values)]
    if which == "subleading":
        sorted_values = sorted(values, reverse=True)
        return [sorted_values[1]] if len(sorted_values) > 1 else []
    if which.startswith("index:"):
        index = int(which.split(":", 1)[1])
        return [values[index]] if index < len(values) else []
    raise ValueError(f"Unsupported selector: {which}")


def _run_row(config: ProjectConfig, run: RunResult) -> dict[str, str | float]:
    row: dict[str, str | float] = {
        "run_name": run.run_name,
        "run_number": run.run_number,
    }
    for coupling, value in zip(config.couplings, run.values):
        row[coupling.name] = value
    return row
