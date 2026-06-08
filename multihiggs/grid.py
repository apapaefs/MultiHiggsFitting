from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import cos, pi
from typing import Iterable

from .config import ProjectConfig


@dataclass(frozen=True)
class ScanPoint:
    values: tuple[float, ...]
    texts: tuple[str, ...]

    def as_dict(self, config: ProjectConfig) -> dict[str, float | str]:
        row: dict[str, float | str] = {}
        for coupling, value, text in zip(config.couplings, self.values, self.texts):
            row[coupling.name] = value
            row[coupling.name + "_text"] = text
        return row

    def run_name(self, config: ProjectConfig, run_number: str | None = None) -> str:
        runnum = config.scan.run_number if run_number is None else str(run_number)
        return "run_" + config.name + "_" + runnum + "_" + "_".join(self.texts)


def chebyshev_lobatto_nodes(xmin: float, xmax: float, npoints: int) -> list[float]:
    if npoints < 2:
        return [0.5 * (xmin + xmax)]
    nodes = []
    for index in range(npoints):
        x = cos(pi * index / (npoints - 1))
        nodes.append(0.5 * (xmin + xmax) + 0.5 * (xmax - xmin) * x)
    return sorted(nodes)


def generate_scan_points(config: ProjectConfig) -> list[ScanPoint]:
    if config.scan.strategy != "chebyshev_lobatto":
        raise ValueError(f"Unsupported scan strategy: {config.scan.strategy}")

    fit_nodes = [
        chebyshev_lobatto_nodes(coupling.fit_range[0], coupling.fit_range[1], coupling.points)
        for coupling in config.couplings
    ]
    raw_points: list[tuple[float, ...]] = []
    for fit_values in product(*fit_nodes):
        raw_points.append(
            tuple(
                _clean_scan_value(coupling, coupling.scan_from_fit(fit_value))
                for coupling, fit_value in zip(config.couplings, fit_values)
            )
        )

    for point in config.extra_points:
        values = []
        for coupling in config.couplings:
            value = float(point.get(coupling.name, point.get(coupling.parameter, coupling.sm_value)))
            values.append(_clean_scan_value(coupling, value))
        raw_points.append(tuple(values))

    points = _deduplicate(raw_points)
    if config.scan.sort == "sm_first":
        sm = tuple(coupling.sm_value for coupling in config.couplings)
        points.sort(key=lambda values: (values != sm, sum(abs(v - s) for v, s in zip(values, sm)), values))
    elif config.scan.sort not in ("grid", "none"):
        raise ValueError(f"Unsupported scan sort mode: {config.scan.sort}")

    return [
        ScanPoint(
            values=values,
            texts=tuple(coupling.format_value(value) for coupling, value in zip(config.couplings, values)),
        )
        for values in points
    ]


def _deduplicate(points: Iterable[tuple[float, ...]]) -> list[tuple[float, ...]]:
    seen: set[tuple[float, ...]] = set()
    unique: list[tuple[float, ...]] = []
    for values in points:
        key = tuple(round(value, 12) for value in values)
        if key in seen:
            continue
        seen.add(key)
        unique.append(values)
    return unique


def _clean_scan_value(coupling, value: float) -> float:
    return float(coupling.format_value(value))
