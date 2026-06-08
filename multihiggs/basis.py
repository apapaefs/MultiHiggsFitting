from __future__ import annotations

from math import prod

import numpy as np

from .config import CouplingConfig, ProjectConfig


def scale_to_chebyshev(value: float, value_range: tuple[float, float]) -> float:
    xmin, xmax = value_range
    return (2.0 * value - xmin - xmax) / (xmax - xmin)


def chebyshev_t(order: int, x: float) -> float:
    if order == 0:
        return 1.0
    if order == 1:
        return float(x)
    previous = 1.0
    current = float(x)
    for _ in range(2, order + 1):
        previous, current = current, 2.0 * x * current - previous
    return current


def chebyshev_row(values: tuple[float, ...], config: ProjectConfig) -> np.ndarray:
    fit_values = [
        coupling.fit_from_scan(value)
        for coupling, value in zip(config.couplings, values)
    ]
    scaled = [
        scale_to_chebyshev(value, coupling.fit_range)
        for coupling, value in zip(config.couplings, fit_values)
    ]
    return np.asarray(
        [
            prod(chebyshev_t(power, x) for power, x in zip(term, scaled))
            for term in config.fit.terms
        ],
        dtype=float,
    )


def design_matrix(values: list[tuple[float, ...]], config: ProjectConfig) -> np.ndarray:
    return np.asarray([chebyshev_row(point, config) for point in values], dtype=float)


def univariate_chebyshev_polynomial(
    order: int,
    coupling: CouplingConfig,
    variable: str,
) -> np.ndarray:
    xmin, xmax = coupling.fit_range
    a = 2.0 / (xmax - xmin)
    if variable == "fit":
        b = -(xmin + xmax) / (xmax - xmin)
    elif variable == "scan":
        b = (2.0 * coupling.fit_offset - xmin - xmax) / (xmax - xmin)
    else:
        raise ValueError(f"Unsupported monomial variable convention: {variable}")
    x_poly = np.asarray([b, a], dtype=float)
    if order == 0:
        return np.asarray([1.0], dtype=float)
    if order == 1:
        return x_poly
    previous = np.asarray([1.0], dtype=float)
    current = x_poly
    for _ in range(2, order + 1):
        next_poly = poly_add(2.0 * np.convolve(x_poly, current), -previous)
        previous, current = current, next_poly
    return current


def monomial_transform(
    config: ProjectConfig,
    variable: str = "scan",
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    expansions: list[dict[tuple[int, ...], float]] = []
    all_powers: set[tuple[int, ...]] = set()
    ndims = len(config.couplings)
    for term in config.fit.terms:
        expansion: dict[tuple[int, ...], float] = {tuple([0] * ndims): 1.0}
        for dim, (power, coupling) in enumerate(zip(term, config.couplings)):
            poly = univariate_chebyshev_polynomial(power, coupling, variable)
            next_expansion: dict[tuple[int, ...], float] = {}
            for powers, coeff in expansion.items():
                for out_power, out_coeff in enumerate(poly):
                    next_powers = list(powers)
                    next_powers[dim] += out_power
                    next_key = tuple(next_powers)
                    next_expansion[next_key] = next_expansion.get(next_key, 0.0) + coeff * out_coeff
            expansion = next_expansion
        expansions.append(expansion)
        all_powers.update(expansion)

    preferred = []
    seen_preferred = set()
    for term in config.fit.terms:
        if term in all_powers and term not in seen_preferred:
            preferred.append(term)
            seen_preferred.add(term)
    remaining = sorted(all_powers - seen_preferred, key=lambda powers: (sum(powers), powers))
    powers_list = preferred + remaining
    row_index = {powers: index for index, powers in enumerate(powers_list)}
    transform = np.zeros((len(powers_list), len(config.fit.terms)), dtype=float)
    for col, expansion in enumerate(expansions):
        for powers, coeff in expansion.items():
            transform[row_index[powers], col] = coeff
    return powers_list, transform


def monomial_basis(values: tuple[float, ...], powers: list[tuple[int, ...]]) -> np.ndarray:
    return np.asarray(
        [prod(value ** power for value, power in zip(values, term)) for term in powers],
        dtype=float,
    )


def monomial_labels(
    config: ProjectConfig,
    powers: list[tuple[int, ...]],
    variable: str = "scan",
) -> list[str]:
    labels = []
    for term in powers:
        pieces = []
        for coupling, power in zip(config.couplings, term):
            name = coupling.fit_name if variable == "fit" else coupling.name
            if power == 1:
                pieces.append(name)
            elif power > 1:
                pieces.append(f"{name}^{power}")
        labels.append("*".join(pieces) if pieces else "1")
    return labels


def poly_add(poly_a: np.ndarray, poly_b: np.ndarray) -> np.ndarray:
    size = max(len(poly_a), len(poly_b))
    result = np.zeros(size, dtype=float)
    result[: len(poly_a)] += poly_a
    result[: len(poly_b)] += poly_b
    return result
