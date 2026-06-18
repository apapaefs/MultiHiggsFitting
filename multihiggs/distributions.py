from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .basis import design_matrix
from .config import ObservableConfig, ProjectConfig
from .contours import format_axis_label, resolve_coupling, safe_filename_piece


DISTRIBUTION_TITLE_FONTSIZE = 13
SM_DISTRIBUTION_COLOR = "black"
PARAMETER_LINESTYLES = ("-", "--", "-.", ":")


@dataclass(frozen=True)
class DistributionSeries:
    label: str
    values: np.ndarray
    errors: np.ndarray
    parameter_values: dict[str, float]


@dataclass(frozen=True)
class DistributionData:
    observable: ObservableConfig
    bin_lows: np.ndarray
    bin_highs: np.ndarray
    bin_centers: np.ndarray
    bin_half_widths: np.ndarray
    series: list[DistributionSeries]
    density: bool


def parse_parameter_points(config: ProjectConfig, raw_points: list[str] | None) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for raw_point in raw_points or []:
        point: dict[str, float] = {}
        for raw_assignment in raw_point.split(","):
            assignment = raw_assignment.strip()
            if not assignment:
                continue
            if "=" not in assignment:
                raise ValueError(f"Parameter point {raw_point!r} must contain assignments like name=value")
            raw_name, raw_value = assignment.split("=", 1)
            name = raw_name.strip()
            coupling = resolve_coupling(config, name)
            try:
                point[coupling.name] = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"Parameter value for {name!r} is not a number: {raw_value!r}") from exc
        if not point:
            raise ValueError(f"Parameter point {raw_point!r} does not contain any assignments")
        points.append(point)
    return points


def build_distribution_data(
    config: ProjectConfig,
    fit_payload: dict[str, Any],
    observable_name: str,
    parameter_points: list[dict[str, float]] | None = None,
    *,
    include_sm: bool = True,
    density: bool = False,
) -> DistributionData:
    observable = find_observable(config, observable_name)
    fit_records = select_observable_fit_records(fit_payload, observable)
    if not parameter_points and not include_sm:
        raise ValueError("No distributions requested: provide --point or omit --no-sm")

    bin_edges = np.asarray(observable.bins, dtype=float)
    bin_lows = bin_edges[:-1]
    bin_highs = bin_edges[1:]
    widths = bin_highs - bin_lows
    bin_centers = 0.5 * (bin_lows + bin_highs)
    bin_half_widths = 0.5 * widths

    series: list[DistributionSeries] = []
    for point in parameter_points or []:
        label = format_parameter_point_label(config, point)
        values, errors = evaluate_observable_bins(config, fit_records, point)
        if density:
            values = values / widths
            errors = errors / widths
        series.append(DistributionSeries(label, values, errors, clean_parameter_point(config, point)))

    if include_sm:
        sm_point = {coupling.name: coupling.sm_value for coupling in config.couplings}
        values, errors = evaluate_observable_bins(config, fit_records, sm_point)
        if density:
            values = values / widths
            errors = errors / widths
        series.append(DistributionSeries("SM", values, errors, sm_point))

    return DistributionData(
        observable=observable,
        bin_lows=bin_lows,
        bin_highs=bin_highs,
        bin_centers=bin_centers,
        bin_half_widths=bin_half_widths,
        series=series,
        density=density,
    )


def find_observable(config: ProjectConfig, name: str) -> ObservableConfig:
    for observable in config.observables:
        if observable.name == name:
            return observable
    valid = ", ".join(observable.name for observable in config.observables)
    raise ValueError(f"Unknown observable {name!r}. Configured observables: {valid}")


def select_observable_fit_records(
    fit_payload: dict[str, Any],
    observable: ObservableConfig,
) -> list[dict[str, Any]]:
    fits = fit_payload.get("fits")
    if not isinstance(fits, list):
        raise ValueError("Fit JSON does not contain a fits list")
    pattern = re.compile(rf"^{re.escape(observable.name)}:bin([0-9]+)$")
    records_by_bin: dict[int, dict[str, Any]] = {}
    for record in fits:
        match = pattern.match(str(record.get("label", "")))
        if match is None:
            continue
        records_by_bin[int(match.group(1))] = record

    expected_bins = len(observable.bins) - 1
    missing = [index for index in range(expected_bins) if index not in records_by_bin]
    if missing:
        raise ValueError(
            f"Fit JSON is missing {observable.name} bin fit(s): "
            + ", ".join(f"bin{index}" for index in missing)
        )
    return [records_by_bin[index] for index in range(expected_bins)]


def evaluate_observable_bins(
    config: ProjectConfig,
    fit_records: list[dict[str, Any]],
    point: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    errors = []
    complete_values = complete_parameter_values(config, point)
    basis_row = design_matrix([complete_values], config)[0]
    for record in fit_records:
        coefficients = np.asarray(record["coefficients"], dtype=float)
        covariance = np.asarray(record["covariance"], dtype=float)
        value = float(np.dot(basis_row, coefficients))
        variance = float(np.dot(basis_row, np.dot(covariance, basis_row)))
        values.append(value)
        errors.append(math.sqrt(max(variance, 0.0)))
    return np.asarray(values, dtype=float), np.asarray(errors, dtype=float)


def complete_parameter_values(config: ProjectConfig, point: dict[str, float]) -> tuple[float, ...]:
    clean_point = clean_parameter_point(config, point)
    return tuple(clean_point.get(coupling.name, coupling.sm_value) for coupling in config.couplings)


def clean_parameter_point(config: ProjectConfig, point: dict[str, float]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for name, value in point.items():
        coupling = resolve_coupling(config, name)
        clean[coupling.name] = float(value)
    return clean


def format_parameter_point_label(config: ProjectConfig, point: dict[str, float]) -> str:
    clean = clean_parameter_point(config, point)
    pieces = []
    for coupling in config.couplings:
        if coupling.name in clean:
            pieces.append(f"{coupling.name}={format_number(clean[coupling.name])}")
    return ", ".join(pieces) if pieces else "SM"


def format_number(value: float) -> str:
    return f"{float(value):g}"


def write_distribution_plot(
    config: ProjectConfig,
    data: DistributionData,
    output_path: Path,
    *,
    log_y: bool = False,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "The distribution command requires matplotlib. Install it with "
            "`python3 -m pip install -e '.[plot]'` or `python3 -m pip install matplotlib`."
        ) from exc

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    parameter_style_index = 0
    baseline = distribution_outline_baseline(data, log_y=log_y)
    for index, series in enumerate(data.series):
        color, linestyle = distribution_series_style(series, parameter_style_index, color_cycle)
        draw_distribution_series(
            ax,
            data,
            series,
            color=color,
            marker=marker_for_index(index),
            linestyle=linestyle,
            baseline=baseline,
        )
        if not is_sm_distribution_series(series):
            parameter_style_index += 1

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(format_observable_axis_label(data.observable), fontsize=15)
    ax.set_ylabel(format_distribution_ylabel(data), fontsize=15)
    ax.set_title(make_distribution_title(config, data), fontsize=DISTRIBUTION_TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="major", color="0.88", linewidth=0.8)
    ax.grid(True, which="minor", color="0.94", linewidth=0.5)
    ax.legend(frameon=False, fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def histogram_step_xy(
    bin_lows: np.ndarray,
    bin_highs: np.ndarray,
    values: np.ndarray,
    *,
    baseline: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    numeric_values = np.asarray(values, dtype=float)
    edges = np.concatenate(([float(bin_lows[0])], np.asarray(bin_highs, dtype=float)))
    x_values = np.repeat(edges, 2)
    y_values = np.concatenate(([float(baseline)], np.repeat(numeric_values, 2), [float(baseline)]))
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def draw_distribution_series(
    ax,
    data: DistributionData,
    series: DistributionSeries,
    *,
    color: str,
    marker: str,
    linestyle: str = "-",
    baseline: float = 0.0,
) -> None:
    step_x, step_y = histogram_step_xy(data.bin_lows, data.bin_highs, series.values, baseline=baseline)
    ax.plot(
        step_x,
        step_y,
        linestyle=linestyle,
        linewidth=1.8,
        color=color,
        label=series.label,
    )
    ax.errorbar(
        data.bin_centers,
        series.values,
        yerr=series.errors,
        fmt=marker,
        linestyle="none",
        color=color,
        markersize=4.2,
        capsize=3.0,
        elinewidth=1.0,
    )


def distribution_outline_baseline(data: DistributionData, *, log_y: bool) -> float:
    if not log_y:
        return 0.0
    positive_values = [
        float(value)
        for series in data.series
        for value in series.values
        if float(value) > 0.0
    ]
    if not positive_values:
        return 1e-30
    return min(positive_values) * 0.5


def distribution_series_style(
    series: DistributionSeries,
    parameter_style_index: int,
    color_cycle: list[str],
) -> tuple[str, str]:
    if is_sm_distribution_series(series):
        return SM_DISTRIBUTION_COLOR, "-"
    color = color_cycle[parameter_style_index % len(color_cycle)] if color_cycle else f"C{parameter_style_index}"
    return color, PARAMETER_LINESTYLES[parameter_style_index % len(PARAMETER_LINESTYLES)]


def is_sm_distribution_series(series: DistributionSeries) -> bool:
    return series.label == "SM"


def marker_for_index(index: int) -> str:
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    return markers[index % len(markers)]


def format_observable_axis_label(observable: ObservableConfig) -> str:
    if observable.kind == "pt":
        subject = format_particle_label(observable.pdg_id)
        return rf"$p_{{T,{subject}}}$ [GeV]"
    if observable.kind == "invariant_mass":
        subject = "".join(format_particle_label(pid) for pid in observable.pdg_ids)
        return rf"$m_{{{subject}}}$ [GeV]"
    return format_axis_label(observable.name)


def format_particle_label(pid: int | None) -> str:
    if pid is None:
        return "x"
    labels = {
        6: "t",
        25: "h",
    }
    base = labels.get(abs(pid), str(abs(pid)))
    return base


def format_distribution_ylabel(data: DistributionData) -> str:
    if data.density:
        return r"$d\sigma/dx$ [pb/GeV]"
    return r"$\sigma_{\rm bin}$ [pb]"


def make_distribution_title(config: ProjectConfig, data: DistributionData) -> str:
    return f"{config.name} at {config.scan.energy_tev:g} TeV: {data.observable.name}"


def default_distribution_output(config: ProjectConfig, observable_name: str) -> Path:
    return config.output_dir / f"{safe_filename_piece(observable_name)}_distribution.png"
