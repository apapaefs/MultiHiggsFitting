from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .basis import chebyshev_row
from .config import CouplingConfig, ProjectConfig


PLOT_TITLE_FONTSIZE = 13


@dataclass(frozen=True)
class ContourData:
    x_name: str
    y_name: str
    x_values: np.ndarray
    y_values: np.ndarray
    ratio: np.ndarray
    fixed_values: dict[str, float]
    fit_label: str


@dataclass(frozen=True)
class VariationData:
    x_name: str
    x_values: np.ndarray
    ratio: np.ndarray
    fixed_values: dict[str, float]
    fit_label: str


def load_fit_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_fit_record(payload: dict[str, Any], label: str | None = None) -> dict[str, Any]:
    fits = payload.get("fits")
    if not isinstance(fits, list) or not fits:
        raise ValueError("Fit JSON does not contain any fit records")

    if label is not None:
        for record in fits:
            if record.get("label") == label:
                return record
        raise ValueError(f"No fit labeled {label!r} found")

    xsec_records = [record for record in fits if record.get("label") == "xsec"]
    if len(xsec_records) == 1:
        return xsec_records[0]
    if len(fits) == 1:
        return fits[0]

    labels = ", ".join(str(record.get("label")) for record in fits[:8])
    if len(fits) > 8:
        labels += ", ..."
    raise ValueError(f"Fit JSON contains multiple fits; choose one with --label. Available labels: {labels}")


def resolve_coupling(config: ProjectConfig, name: str) -> CouplingConfig:
    matches = [
        coupling
        for coupling in config.couplings
        if name in {coupling.name, coupling.parameter, coupling.fit_name}
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Coupling name {name!r} is ambiguous")
    valid = ", ".join(coupling.name for coupling in config.couplings)
    raise ValueError(f"Unknown coupling {name!r}. Configured couplings: {valid}")


def parse_fixed_values(
    config: ProjectConfig,
    raw_values: list[str] | None,
    axis_names: set[str] | None = None,
) -> dict[str, float]:
    fixed_values: dict[str, float] = {}
    axis_names = axis_names or set()
    for raw_value in raw_values or []:
        if "=" not in raw_value:
            raise ValueError(f"Fixed value {raw_value!r} must have the form name=value")
        raw_name, raw_number = raw_value.split("=", 1)
        name = raw_name.strip()
        if not name:
            raise ValueError(f"Fixed value {raw_value!r} must include a coupling name")
        coupling = resolve_coupling(config, name)
        if coupling.name in axis_names or coupling.parameter in axis_names or coupling.fit_name in axis_names:
            raise ValueError(f"Cannot fix axis variable {name!r}")
        try:
            fixed_values[coupling.name] = float(raw_number)
        except ValueError as exc:
            raise ValueError(f"Fixed value for {name!r} is not a number: {raw_number!r}") from exc
    return fixed_values


def build_contour_data(
    config: ProjectConfig,
    fit_record: dict[str, Any],
    x_name: str,
    y_name: str,
    fixed_values: dict[str, float] | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    x_points: int = 201,
    y_points: int = 201,
) -> ContourData:
    if x_points < 2 or y_points < 2:
        raise ValueError("Contour grids need at least two points on each axis")

    x_coupling = resolve_coupling(config, x_name)
    y_coupling = resolve_coupling(config, y_name)
    if x_coupling.name == y_coupling.name:
        raise ValueError("Contour axes must be two different configured variables")

    fixed_values = fixed_values or {}
    axis_names = {x_coupling.name, y_coupling.name}
    for fixed_name in fixed_values:
        fixed_coupling = resolve_coupling(config, fixed_name)
        if fixed_coupling.name in axis_names:
            raise ValueError(f"Cannot fix axis variable {fixed_name!r}")

    x_range = x_range or default_scan_range(x_coupling)
    y_range = y_range or default_scan_range(y_coupling)
    x_values = np.linspace(float(x_range[0]), float(x_range[1]), x_points)
    y_values = np.linspace(float(y_range[0]), float(y_range[1]), y_points)
    ratio = np.zeros((y_points, x_points), dtype=float)

    coefficients = np.asarray(fit_record["coefficients"], dtype=float)
    sigma_sm = float(fit_record.get("sigma_sm_pb") or 0.0)
    if not math.isfinite(sigma_sm) or abs(sigma_sm) < 1e-30:
        raise ValueError("Selected fit has zero or non-finite sigma_sm_pb")

    coupling_index = {coupling.name: index for index, coupling in enumerate(config.couplings)}
    x_index = coupling_index[x_coupling.name]
    y_index = coupling_index[y_coupling.name]
    base_values = [coupling.sm_value for coupling in config.couplings]
    clean_fixed_values = {resolve_coupling(config, name).name: float(value) for name, value in fixed_values.items()}
    for fixed_name, fixed_value in clean_fixed_values.items():
        base_values[coupling_index[fixed_name]] = fixed_value

    for row_index, y_value in enumerate(y_values):
        for col_index, x_value in enumerate(x_values):
            values = list(base_values)
            values[x_index] = float(x_value)
            values[y_index] = float(y_value)
            ratio[row_index, col_index] = float(np.dot(chebyshev_row(tuple(values), config), coefficients) / sigma_sm)

    return ContourData(
        x_name=x_coupling.name,
        y_name=y_coupling.name,
        x_values=x_values,
        y_values=y_values,
        ratio=ratio,
        fixed_values=clean_fixed_values,
        fit_label=str(fit_record.get("label", "fit")),
    )


def build_variation_data(
    config: ProjectConfig,
    fit_record: dict[str, Any],
    x_name: str,
    fixed_values: dict[str, float] | None = None,
    x_range: tuple[float, float] | None = None,
    points: int = 201,
) -> VariationData:
    if points < 2:
        raise ValueError("Variation scans need at least two points")

    x_coupling = resolve_coupling(config, x_name)
    fixed_values = fixed_values or {}
    for fixed_name in fixed_values:
        fixed_coupling = resolve_coupling(config, fixed_name)
        if fixed_coupling.name == x_coupling.name:
            raise ValueError(f"Cannot fix axis variable {fixed_name!r}")

    x_range = x_range or default_scan_range(x_coupling)
    x_values = np.linspace(float(x_range[0]), float(x_range[1]), points)
    ratio = np.zeros(points, dtype=float)

    coefficients = np.asarray(fit_record["coefficients"], dtype=float)
    sigma_sm = float(fit_record.get("sigma_sm_pb") or 0.0)
    if not math.isfinite(sigma_sm) or abs(sigma_sm) < 1e-30:
        raise ValueError("Selected fit has zero or non-finite sigma_sm_pb")

    coupling_index = {coupling.name: index for index, coupling in enumerate(config.couplings)}
    x_index = coupling_index[x_coupling.name]
    base_values = [coupling.sm_value for coupling in config.couplings]
    clean_fixed_values = {resolve_coupling(config, name).name: float(value) for name, value in fixed_values.items()}
    for fixed_name, fixed_value in clean_fixed_values.items():
        base_values[coupling_index[fixed_name]] = fixed_value

    for index, x_value in enumerate(x_values):
        values = list(base_values)
        values[x_index] = float(x_value)
        ratio[index] = float(np.dot(chebyshev_row(tuple(values), config), coefficients) / sigma_sm)

    return VariationData(
        x_name=x_coupling.name,
        x_values=x_values,
        ratio=ratio,
        fixed_values=clean_fixed_values,
        fit_label=str(fit_record.get("label", "fit")),
    )


def default_scan_range(coupling: CouplingConfig) -> tuple[float, float]:
    return (
        coupling.scan_from_fit(coupling.fit_range[0]),
        coupling.scan_from_fit(coupling.fit_range[1]),
    )


def write_contour_plot(
    config: ProjectConfig,
    contour: ContourData,
    output_path: Path,
    *,
    log_scale: bool = True,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.colors as colors
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "The contour command requires matplotlib. Install it with "
            "`python3 -m pip install -e '.[plot]'` or `python3 -m pip install matplotlib`."
        ) from exc

    x_grid, y_grid = np.meshgrid(contour.x_values, contour.y_values)
    ratio = contour.ratio
    positive = ratio[np.isfinite(ratio) & (ratio > 0.0)]
    use_log = log_scale and len(positive) > 0

    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    if use_log:
        levels = make_log_levels(positive)
        ratio_for_fill = np.ma.masked_where(ratio <= 0.0, ratio)
        filled = ax.contourf(
            x_grid,
            y_grid,
            ratio_for_fill,
            levels=levels,
            norm=colors.LogNorm(vmin=levels[0], vmax=levels[-1]),
            cmap="viridis",
            extend="both",
        )
        line_levels = make_line_levels(levels)
        if line_levels:
            lines = ax.contour(x_grid, y_grid, ratio_for_fill, levels=line_levels, colors="white", linewidths=0.55)
            ax.clabel(lines, fmt=format_level, inline=True, fontsize=10)
        if np.any(ratio <= 0.0):
            ax.contourf(x_grid, y_grid, ratio <= 0.0, levels=[0.5, 1.5], colors=["0.75"], alpha=0.8)
    else:
        finite = ratio[np.isfinite(ratio)]
        if len(finite) == 0:
            raise RuntimeError("No finite normalized fit values found")
        min_value = float(np.min(finite))
        max_value = float(np.max(finite))
        if min_value == max_value:
            padding = max(abs(min_value) * 0.1, 1e-12)
            min_value -= padding
            max_value += padding
        levels = np.linspace(min_value, max_value, 24)
        filled = ax.contourf(x_grid, y_grid, ratio, levels=levels, cmap="viridis", extend="both")
        lines = ax.contour(x_grid, y_grid, ratio, levels=levels[::4], colors="white", linewidths=0.55)
        ax.clabel(lines, fmt=format_level, inline=True, fontsize=10)

    x_coupling = resolve_coupling(config, contour.x_name)
    y_coupling = resolve_coupling(config, contour.y_name)
    ax.plot(
        [x_coupling.sm_value],
        [y_coupling.sm_value],
        marker="o",
        color="white",
        markeredgecolor="black",
        markersize=5,
    )
    ax.set_xlim((float(contour.x_values[0]), float(contour.x_values[-1])))
    ax.set_ylim((float(contour.y_values[0]), float(contour.y_values[-1])))
    ax.set_xlabel(format_axis_label(contour.x_name), fontsize=18)
    ax.set_ylabel(format_axis_label(contour.y_name), fontsize=18)
    ax.set_title(make_plot_title(config, contour), fontsize=PLOT_TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=14)

    cbar = fig.colorbar(filled, ax=ax)
    cbar.set_label(r"$\sigma/\sigma_{\rm SM}$", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def write_variation_plot(
    config: ProjectConfig,
    variation: VariationData,
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
            "The variation command requires matplotlib. Install it with "
            "`python3 -m pip install -e '.[plot]'` or `python3 -m pip install matplotlib`."
        ) from exc

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.plot(
        variation.x_values,
        variation.ratio,
        color="#1f77b4",
        linewidth=2.0,
        label=variation.fit_label,
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.1, label="SM")

    if log_y:
        ax.set_yscale("log")
    else:
        finite = variation.ratio[np.isfinite(variation.ratio)]
        if len(finite) > 0:
            min_value = min(float(np.min(finite)), 1.0)
            max_value = max(float(np.max(finite)), 1.0)
            if min_value == max_value:
                padding = max(abs(min_value) * 0.1, 1e-12)
            else:
                padding = 0.08 * (max_value - min_value)
            ax.set_ylim(min_value - padding, max_value + padding)

    x_coupling = resolve_coupling(config, variation.x_name)
    ax.set_xlim((float(variation.x_values[0]), float(variation.x_values[-1])))
    ax.axvline(x_coupling.sm_value, color="0.45", linestyle=":", linewidth=1.0)
    ax.set_xlabel(format_axis_label(variation.x_name), fontsize=15)
    ax.set_ylabel(r"$\sigma/\sigma_{\rm SM}$", fontsize=15)
    ax.set_title(make_variation_plot_title(config, variation), fontsize=PLOT_TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="major", color="0.88", linewidth=0.8)
    ax.grid(True, which="minor", color="0.94", linewidth=0.5)
    ax.legend(frameon=False, fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def make_log_levels(positive_values: np.ndarray) -> np.ndarray:
    min_value = float(np.min(positive_values))
    max_value = float(np.max(positive_values))
    if min_value == max_value:
        min_value /= 10.0
        max_value *= 10.0
    lo = math.floor(math.log10(min_value))
    hi = math.ceil(math.log10(max_value))
    nlevels = min(max((hi - lo) * 4 + 1, 12), 80)
    return np.logspace(lo, hi, nlevels)


def make_line_levels(filled_levels: np.ndarray) -> list[float]:
    lo = math.floor(math.log10(float(filled_levels[0])))
    hi = math.ceil(math.log10(float(filled_levels[-1])))
    return [
        10.0**power
        for power in range(lo, hi + 1)
        if float(filled_levels[0]) <= 10.0**power <= float(filled_levels[-1])
    ]


def format_level(value: float) -> str:
    value = float(value)
    if abs(value) >= 1000.0 or (abs(value) > 0.0 and abs(value) < 0.01):
        return f"{value:.0e}"
    if abs(value) >= 10.0:
        return f"{value:.0f}"
    return f"{value:.2g}"


def format_axis_label(name: str) -> str:
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        return "$" + name.replace("_", r"\_") + "$"
    return name


def make_plot_title(config: ProjectConfig, contour: ContourData) -> str:
    title = f"{config.name} at {config.scan.energy_tev:g} TeV: {contour.fit_label}"
    if contour.fixed_values:
        fixed_text = ", ".join(f"{name}={value:g}" for name, value in sorted(contour.fixed_values.items()))
        title += f" ({fixed_text})"
    return title


def make_variation_plot_title(config: ProjectConfig, variation: VariationData) -> str:
    title = f"{config.name} at {config.scan.energy_tev:g} TeV: {variation.fit_label}"
    if variation.fixed_values:
        fixed_text = ", ".join(f"{name}={value:g}" for name, value in sorted(variation.fixed_values.items()))
        title += f" ({fixed_text})"
    return title


def safe_filename_piece(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return clean.strip("_") or "fit"
