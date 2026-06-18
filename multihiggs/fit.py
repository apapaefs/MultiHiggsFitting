from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .basis import (
    design_matrix,
    monomial_basis,
    monomial_labels,
    monomial_transform,
    physical_variable_names,
    physical_variable_values,
    resolved_fit_term_map,
)
from .config import ProjectConfig
from .results import RunResult, read_results_csv
from .term_maps import (
    format_factored_power_label,
    format_power_label,
    resolve_term_map,
    transform_coefficients,
    transform_mapped_coefficients_to_sources,
)


@dataclass(frozen=True)
class FitResult:
    label: str
    coefficients: np.ndarray
    covariance: np.ndarray
    rank: int
    condition: float
    chi2_dof: float
    sigma_sm: float
    fit_monomial_powers: list[tuple[int, ...]]
    fit_monomial_coefficients: np.ndarray
    fit_monomial_covariance: np.ndarray
    normalized_fit_monomial_coefficients: np.ndarray
    normalized_fit_monomial_covariance: np.ndarray
    monomial_powers: list[tuple[int, ...]]
    monomial_coefficients: np.ndarray
    monomial_covariance: np.ndarray
    normalized_monomial_coefficients: np.ndarray
    normalized_monomial_covariance: np.ndarray

    def to_dict(self, config: ProjectConfig) -> dict[str, object]:
        fit_label_variable = "physical" if config.fit.basis == "physical_monomial" else "fit"
        fit_mono_labels = monomial_labels(config, self.fit_monomial_powers, variable=fit_label_variable)
        mono_labels = monomial_labels(config, self.monomial_powers, variable="scan")
        return {
            "label": self.label,
            "basis": config.fit.basis,
            "terms": [list(term) for term in config.fit.terms],
            "coefficients": clean_json_value(self.coefficients),
            "coefficient_errors": clean_json_value(np.sqrt(np.abs(np.diag(self.covariance)))),
            "covariance": clean_json_value(self.covariance),
            "rank": self.rank,
            "condition": clean_json_value(self.condition),
            "chi2_dof": clean_json_value(self.chi2_dof),
            "sigma_sm_pb": clean_json_value(self.sigma_sm),
            "couplings": [coupling.name for coupling in config.couplings],
            "fit_variables": [coupling.fit_name for coupling in config.couplings],
            "fit_monomial": [
                {
                    "label": label,
                    "powers": list(powers),
                    "coefficient_pb": clean_json_value(coeff),
                    "error_pb": clean_json_value(err),
                    "normalized": clean_json_value(norm),
                    "normalized_error": clean_json_value(norm_err),
                }
                for label, powers, coeff, err, norm, norm_err in zip(
                    fit_mono_labels,
                    self.fit_monomial_powers,
                    self.fit_monomial_coefficients.tolist(),
                    np.sqrt(np.abs(np.diag(self.fit_monomial_covariance))).tolist(),
                    self.normalized_fit_monomial_coefficients.tolist(),
                    np.sqrt(np.abs(np.diag(self.normalized_fit_monomial_covariance))).tolist(),
                )
            ],
            "monomial": [
                {
                    "label": label,
                    "powers": list(powers),
                    "coefficient_pb": clean_json_value(coeff),
                    "error_pb": clean_json_value(err),
                    "normalized": clean_json_value(norm),
                    "normalized_error": clean_json_value(norm_err),
                }
                for label, powers, coeff, err, norm, norm_err in zip(
                    mono_labels,
                    self.monomial_powers,
                    self.monomial_coefficients.tolist(),
                    np.sqrt(np.abs(np.diag(self.monomial_covariance))).tolist(),
                    self.normalized_monomial_coefficients.tolist(),
                    np.sqrt(np.abs(np.diag(self.normalized_monomial_covariance))).tolist(),
                )
            ],
        }


def fit_runs(config: ProjectConfig, results: Iterable[RunResult], label: str = "xsec") -> FitResult:
    rows = list(results)
    values = [row.values for row in rows]
    y = np.asarray([row.xsec_pb for row in rows], dtype=float)
    yerr = np.asarray([row.xerr_pb for row in rows], dtype=float)
    return fit_values(config, values, y, yerr, label)


def fit_values(
    config: ProjectConfig,
    values: list[tuple[float, ...]],
    y: np.ndarray,
    yerr: np.ndarray | None,
    label: str,
) -> FitResult:
    design = design_matrix(values, config)
    if yerr is None:
        errors = np.maximum(np.abs(y) * 0.01, 1e-30)
    else:
        fallback = zero_error_fallback(y, yerr)
        errors = np.where(yerr > 0.0, yerr, fallback)
    fit_design = design / errors[:, None]
    fit_values_array = y / errors
    coefficients, _, rank, _ = np.linalg.lstsq(fit_design, fit_values_array, rcond=None)
    residuals = y - np.dot(design, coefficients)
    dof = max(len(y) - len(coefficients), 1)
    chi2_dof = float(np.dot(residuals / errors, residuals / errors) / dof)
    covariance = max(chi2_dof, 1.0) * np.linalg.pinv(np.dot(fit_design.T, fit_design))
    condition = float(np.linalg.cond(design))

    if config.fit.basis == "chebyshev":
        fit_powers, fit_transform = monomial_transform(config, variable="fit")
        fit_monomial_coefficients = np.dot(fit_transform, coefficients)
        fit_monomial_covariance = np.dot(fit_transform, np.dot(covariance, fit_transform.T))
        sm_fit_values = tuple(coupling.fit_from_scan(coupling.sm_value) for coupling in config.couplings)
        sm_fit_basis = monomial_basis(sm_fit_values, fit_powers)
        sigma_sm_fit = float(np.dot(sm_fit_basis, fit_monomial_coefficients))
        normalized_fit_coeffs, normalized_fit_cov = normalize_coefficients(
            fit_monomial_coefficients,
            fit_monomial_covariance,
            sm_fit_basis,
            sigma_sm_fit,
        )

        powers, transform = monomial_transform(config, variable="scan")
        monomial_coefficients = np.dot(transform, coefficients)
        monomial_covariance = np.dot(transform, np.dot(covariance, transform.T))
        sm_values = tuple(coupling.sm_value for coupling in config.couplings)
        sm_basis = monomial_basis(sm_values, powers)
        sigma_sm = float(np.dot(sm_basis, monomial_coefficients))
        normalized_coeffs, normalized_cov = normalize_coefficients(
            monomial_coefficients,
            monomial_covariance,
            sm_basis,
            sigma_sm,
        )
    elif config.fit.basis == "physical_monomial":
        fit_powers = list(config.fit.terms)
        fit_monomial_coefficients = coefficients
        fit_monomial_covariance = covariance
        sm_values = tuple(coupling.sm_value for coupling in config.couplings)
        sm_fit_values = physical_variable_values(sm_values, config)
        sm_fit_basis = monomial_basis(sm_fit_values, fit_powers)
        sigma_sm_fit = float(np.dot(sm_fit_basis, fit_monomial_coefficients))
        normalized_fit_coeffs, normalized_fit_cov = normalize_coefficients(
            fit_monomial_coefficients,
            fit_monomial_covariance,
            sm_fit_basis,
            sigma_sm_fit,
        )

        term_map = resolved_fit_term_map(config)
        powers, monomial_coefficients, monomial_covariance = transform_mapped_coefficients_to_sources(
            fit_powers,
            coefficients,
            covariance,
            term_map,
        )
        sm_basis = monomial_basis(sm_values, powers)
        sigma_sm = float(np.dot(sm_basis, monomial_coefficients))
        normalized_coeffs, normalized_cov = normalize_coefficients(
            monomial_coefficients,
            monomial_covariance,
            sm_basis,
            sigma_sm,
        )
    else:
        raise ValueError(f"Unsupported fit basis: {config.fit.basis}")
    return FitResult(
        label=label,
        coefficients=coefficients,
        covariance=covariance,
        rank=int(rank),
        condition=condition,
        chi2_dof=chi2_dof,
        sigma_sm=sigma_sm,
        fit_monomial_powers=fit_powers,
        fit_monomial_coefficients=fit_monomial_coefficients,
        fit_monomial_covariance=fit_monomial_covariance,
        normalized_fit_monomial_coefficients=normalized_fit_coeffs,
        normalized_fit_monomial_covariance=normalized_fit_cov,
        monomial_powers=powers,
        monomial_coefficients=monomial_coefficients,
        monomial_covariance=monomial_covariance,
        normalized_monomial_coefficients=normalized_coeffs,
        normalized_monomial_covariance=normalized_cov,
    )


def normalize_coefficients(
    coefficients: np.ndarray,
    covariance: np.ndarray,
    sm_basis: np.ndarray,
    sigma_sm: float,
) -> tuple[np.ndarray, np.ndarray]:
    if abs(sigma_sm) < 1e-30:
        return np.full_like(coefficients, math.nan), np.full_like(covariance, math.nan)
    jacobian = np.zeros((len(coefficients), len(coefficients)), dtype=float)
    for index in range(len(coefficients)):
        jacobian[index, index] = 1.0 / sigma_sm
        jacobian[index, :] -= coefficients[index] * sm_basis / (sigma_sm**2)
    return coefficients / sigma_sm, np.dot(jacobian, np.dot(covariance, jacobian.T))


def zero_error_fallback(y: np.ndarray, yerr: np.ndarray) -> np.ndarray:
    positive_errors = yerr[yerr > 0.0]
    if len(positive_errors) > 0:
        zero_scale = float(np.median(positive_errors))
    else:
        positive_values = np.abs(y[np.abs(y) > 0.0])
        zero_scale = float(np.median(positive_values) * 0.1) if len(positive_values) > 0 else 1.0
    zero_scale = max(zero_scale, 1e-30)
    value_scale = np.maximum(np.abs(y) * 0.01, zero_scale)
    return np.where(np.abs(y) > 0.0, value_scale, zero_scale)


def fit_results_csv(
    config: ProjectConfig,
    input_path: Path,
    output_path: Path,
    min_events: int | None = None,
) -> list[FitResult]:
    with input_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
    if "bin_xsec_pb" in fieldnames:
        fit_results = fit_histogram_csv(config, input_path, min_events=min_events)
    else:
        results = read_results_csv(input_path, config)
        results = filter_run_results_by_min_events(results, min_events)
        fit_results = [fit_runs(config, results, label="xsec")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "config": str(config.path),
        "process": config.name,
        "fits": [result.to_dict(config) for result in fit_results],
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return fit_results


def fit_histogram_csv(
    config: ProjectConfig,
    input_path: Path,
    min_events: int | None = None,
) -> list[FitResult]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = {}
    skipped_low_stats = 0
    missing_event_count = 0
    with input_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if min_events is not None:
                raw_count = row.get("event_count")
                if not raw_count:
                    missing_event_count += 1
                    continue
                if int(raw_count) < min_events:
                    skipped_low_stats += 1
                    continue
            key = (row["observable"], int(row["bin_index"]))
            groups.setdefault(key, []).append(row)
    if min_events is not None:
        if missing_event_count:
            print(
                "Warning: skipped "
                f"{missing_event_count} row(s) with no event_count column/value; "
                "regenerate the CSV with the current hist command to filter by event count"
            )
        if skipped_low_stats:
            print(f"Warning: skipped {skipped_low_stats} row(s) with fewer than {min_events} events")
        if not groups:
            raise RuntimeError(f"No histogram rows found with at least {min_events} events")

    fit_results: list[FitResult] = []
    for (observable, bin_index), rows in sorted(groups.items()):
        values = [
            tuple(float(row[coupling.name]) for coupling in config.couplings)
            for row in rows
        ]
        y = np.asarray([float(row["bin_xsec_pb"]) for row in rows], dtype=float)
        yerr = np.asarray([float(row.get("bin_error_pb") or 0.0) for row in rows], dtype=float)
        label = f"{observable}:bin{bin_index}"
        fit_results.append(fit_values(config, values, y, yerr, label))
    return fit_results


def filter_run_results_by_min_events(
    results: list[RunResult],
    min_events: int | None,
) -> list[RunResult]:
    if min_events is None:
        return results
    filtered = []
    skipped_low_stats = 0
    missing_event_count = 0
    for result in results:
        if result.event_count is None:
            missing_event_count += 1
            continue
        if result.event_count < min_events:
            skipped_low_stats += 1
            continue
        filtered.append(result)
    if missing_event_count:
        print(
            "Warning: skipped "
            f"{missing_event_count} run(s) with no event_count column/value; "
            "regenerate the CSV with the current collect command to filter by event count"
        )
    if skipped_low_stats:
        print(f"Warning: skipped {skipped_low_stats} run(s) with fewer than {min_events} events")
    if not filtered:
        raise RuntimeError(f"No run rows found with at least {min_events} events")
    return filtered


def format_polynomial_report(
    config: ProjectConfig,
    result: FitResult,
    term_map_name: str | None = None,
    expand_term_map: bool = False,
) -> str:
    lines = []
    if result.label != "xsec":
        lines.append(f"Fit label: {result.label}")
    if config.fit.basis == "chebyshev":
        lines.append("Absolute Chebyshev coefficients:")
        for label, coeff, err in zip(
            chebyshev_labels(config),
            result.coefficients,
            np.sqrt(np.abs(np.diag(result.covariance))),
        ):
            lines.append(f"{label} {round_sig(coeff, 6)} +- {round_sig(err, 3)}")

        lines.append("Chebyshev coefficients normalized to the constant term:")
        constant = result.coefficients[0]
        for label, coeff, err in zip(
            chebyshev_labels(config),
            result.coefficients,
            np.sqrt(np.abs(np.diag(result.covariance))),
        ):
            if abs(constant) < 1e-30:
                lines.append(f"{label} nan +- nan")
            else:
                lines.append(f"{label} {round_sig(coeff / constant, 3)} +- {round_sig(err / abs(constant), 3)}")
        fit_variable_text = ",".join(coupling.fit_name for coupling in config.couplings)
        fit_label_variable = "fit"
    elif config.fit.basis == "physical_monomial":
        lines.append("Absolute physical-basis coefficients:")
        lines.extend(
            coefficient_lines(
                config,
                result.fit_monomial_powers,
                result.fit_monomial_coefficients,
                result.fit_monomial_covariance,
                variable="physical",
                sig=6,
            )
        )
        fit_variable_text = ",".join(physical_variable_names(config))
        fit_label_variable = "physical"
    else:
        raise ValueError(f"Unsupported fit basis: {config.fit.basis}")
    lines.append(f"Physical monomial coefficients in {fit_variable_text}:")
    lines.extend(
        coefficient_lines(
            config,
            result.fit_monomial_powers,
            result.fit_monomial_coefficients,
            result.fit_monomial_covariance,
            variable=fit_label_variable,
            sig=6,
        )
    )

    sm_conditions = ",".join(f"{coupling.name}={coupling.sm_value:g}" for coupling in config.couplings)
    lines.append(f"Fitted sigma({sm_conditions}): {round_sig(result.sigma_sm, 6)}")
    lines.append(f"Physical monomial function normalized to sigma({sm_conditions}):")
    lines.extend(
        coefficient_lines(
            config,
            result.fit_monomial_powers,
            result.normalized_fit_monomial_coefficients,
            result.normalized_fit_monomial_covariance,
            variable=fit_label_variable,
            sig=6,
        )
    )

    scan_variable_text = ",".join(coupling.name for coupling in config.couplings)
    lines.append(f"Physical polynomial coefficients in {scan_variable_text}:")
    lines.extend(
        coefficient_lines(
            config,
            result.monomial_powers,
            result.monomial_coefficients,
            result.monomial_covariance,
            variable="scan",
            sig=6,
        )
    )

    lines.append(f"Physical {scan_variable_text} function normalized to sigma({sm_conditions}):")
    lines.extend(
        coefficient_lines(
            config,
            result.monomial_powers,
            result.normalized_monomial_coefficients,
            result.normalized_monomial_covariance,
            variable="scan",
            sig=6,
        )
    )
    if term_map_name is not None:
        term_map = resolve_term_map(config, term_map_name)
        mapped_variable_text = ",".join(term_map.names)
        lines.append(f"Physical polynomial coefficients in minimal term map {term_map.name} ({mapped_variable_text}):")
        mapping_text = term_map.mapping_text()
        if mapping_text:
            lines.append(f"Mapping: {mapping_text}")
        lines.extend(
            coefficient_lines_for_labels(
                [
                    format_factored_power_label(power, term_map)
                    for power in result.monomial_powers
                ],
                result.monomial_coefficients,
                result.monomial_covariance,
                sig=6,
            )
        )

        lines.append(
            f"Physical minimal {mapped_variable_text} function normalized to sigma({sm_conditions}):"
        )
        lines.extend(
            coefficient_lines_for_labels(
                [
                    format_factored_power_label(power, term_map)
                    for power in result.monomial_powers
                ],
                result.normalized_monomial_coefficients,
                result.normalized_monomial_covariance,
                sig=6,
            )
        )
        if expand_term_map:
            mapped_powers, mapped_coefficients, mapped_covariance = transform_coefficients(
                result.monomial_powers,
                result.monomial_coefficients,
                result.monomial_covariance,
                term_map,
            )
            mapped_normalized_powers, mapped_normalized_coefficients, mapped_normalized_covariance = transform_coefficients(
                result.monomial_powers,
                result.normalized_monomial_coefficients,
                result.normalized_monomial_covariance,
                term_map,
            )
            lines.append(f"Expanded physical polynomial coefficients in term map {term_map.name} ({mapped_variable_text}):")
            lines.extend(
                coefficient_lines_for_labels(
                    [
                        format_power_label(power, term_map.names)
                        for power in mapped_powers
                    ],
                    mapped_coefficients,
                    mapped_covariance,
                    sig=6,
                )
            )

            lines.append(
                f"Expanded physical {mapped_variable_text} function normalized to sigma({sm_conditions}):"
            )
            lines.extend(
                coefficient_lines_for_labels(
                    [
                        format_power_label(power, term_map.names)
                        for power in mapped_normalized_powers
                    ],
                    mapped_normalized_coefficients,
                    mapped_normalized_covariance,
                    sig=6,
                )
            )
    return "\n".join(lines)


def coefficient_lines(
    config: ProjectConfig,
    powers: list[tuple[int, ...]],
    coefficients: np.ndarray,
    covariance: np.ndarray,
    variable: str,
    sig: int,
) -> list[str]:
    labels = monomial_labels(config, powers, variable=variable)
    return coefficient_lines_for_labels(labels, coefficients, covariance, sig)


def coefficient_lines_for_labels(
    labels: list[str],
    coefficients: np.ndarray,
    covariance: np.ndarray,
    sig: int,
) -> list[str]:
    errors = np.sqrt(np.abs(np.diag(covariance)))
    return [
        f"{label} {round_sig(coeff, sig)} +- {round_sig(err, 3)}"
        for label, coeff, err in zip(labels, coefficients, errors)
    ]


def chebyshev_labels(config: ProjectConfig) -> list[str]:
    return [format_chebyshev_term(term, config) for term in config.fit.terms]


def format_chebyshev_term(term: tuple[int, ...], config: ProjectConfig) -> str:
    pieces = []
    for power, coupling in zip(term, config.couplings):
        if power > 0:
            pieces.append(f"T{power}({chebyshev_axis_name(coupling.fit_name)})")
    return "*".join(pieces) if pieces else "1"


def chebyshev_axis_name(fit_name: str) -> str:
    if fit_name.startswith("k") and len(fit_name) > 1:
        return "x" + fit_name[1:]
    return "x_" + fit_name


def round_sig(value: float, sig: int) -> float:
    value = float(value)
    if not math.isfinite(value):
        return value
    if value == 0.0:
        return 0.0
    return round(value, sig - int(math.floor(math.log10(abs(value)))) - 1)


def clean_json_value(value):
    if isinstance(value, np.ndarray):
        return clean_json_value(value.tolist())
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value
