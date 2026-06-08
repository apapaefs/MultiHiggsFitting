from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .basis import design_matrix, monomial_basis, monomial_labels, monomial_transform
from .config import ProjectConfig
from .results import RunResult, read_results_csv


@dataclass(frozen=True)
class FitResult:
    label: str
    coefficients: np.ndarray
    covariance: np.ndarray
    rank: int
    condition: float
    chi2_dof: float
    sigma_sm: float
    monomial_powers: list[tuple[int, ...]]
    monomial_coefficients: np.ndarray
    monomial_covariance: np.ndarray
    normalized_monomial_coefficients: np.ndarray
    normalized_monomial_covariance: np.ndarray

    def to_dict(self, config: ProjectConfig) -> dict[str, object]:
        mono_labels = monomial_labels(config, self.monomial_powers)
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
    if config.fit.basis != "chebyshev":
        raise ValueError(f"Unsupported fit basis: {config.fit.basis}")
    design = design_matrix(values, config)
    if yerr is None:
        errors = np.maximum(np.abs(y) * 0.01, 1e-30)
    else:
        fallback = np.maximum(np.abs(y) * 0.01, 1e-30)
        errors = np.where(yerr > 0.0, yerr, fallback)
    fit_design = design / errors[:, None]
    fit_values_array = y / errors
    coefficients, _, rank, _ = np.linalg.lstsq(fit_design, fit_values_array, rcond=None)
    residuals = y - np.dot(design, coefficients)
    dof = max(len(y) - len(coefficients), 1)
    chi2_dof = float(np.dot(residuals / errors, residuals / errors) / dof)
    covariance = max(chi2_dof, 1.0) * np.linalg.pinv(np.dot(fit_design.T, fit_design))
    condition = float(np.linalg.cond(design))

    powers, transform = monomial_transform(config)
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
    return FitResult(
        label=label,
        coefficients=coefficients,
        covariance=covariance,
        rank=int(rank),
        condition=condition,
        chi2_dof=chi2_dof,
        sigma_sm=sigma_sm,
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


def fit_results_csv(config: ProjectConfig, input_path: Path, output_path: Path) -> list[FitResult]:
    with input_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
    if "bin_xsec_pb" in fieldnames:
        fit_results = fit_histogram_csv(config, input_path)
    else:
        fit_results = [fit_runs(config, read_results_csv(input_path, config), label="xsec")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "config": str(config.path),
        "process": config.name,
        "fits": [result.to_dict(config) for result in fit_results],
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return fit_results


def fit_histogram_csv(config: ProjectConfig, input_path: Path) -> list[FitResult]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = {}
    with input_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            key = (row["observable"], int(row["bin_index"]))
            groups.setdefault(key, []).append(row)

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
