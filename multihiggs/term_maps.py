from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Iterable, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .config import ProjectConfig


Power = tuple[int, ...]


@dataclass(frozen=True)
class TermMapVariable:
    source: str
    name: str
    offset: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TermMapVariable":
        return cls(
            source=str(data["source"]),
            name=str(data["name"]),
            offset=float(data.get("offset", 0.0)),
        )


@dataclass(frozen=True)
class TermMap:
    name: str
    description: str = ""
    variables: tuple[TermMapVariable, ...] = ()

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "TermMap":
        variables = tuple(
            TermMapVariable.from_dict(item)
            for item in data.get("variables", [])
        )
        if not variables:
            raise ValueError(f"Term map {name!r} must define at least one variable")
        return cls(
            name=name,
            description=str(data.get("description", "")),
            variables=variables,
        )


@dataclass(frozen=True)
class ResolvedTermMap:
    term_map: TermMap
    names: tuple[str, ...]
    offsets: tuple[float, ...]
    active_variables: tuple[TermMapVariable | None, ...]

    @property
    def name(self) -> str:
        return self.term_map.name

    def mapping_text(self) -> str:
        pieces = []
        for variable in self.active_variables:
            if variable is None:
                continue
            pieces.append(format_mapping(variable))
        return ", ".join(pieces)


BUILTIN_TERM_MAPS: dict[str, TermMap] = {
    "mheft_kappa": TermMap(
        name="mheft_kappa",
        description="MHEFT deviations rewritten as kappa modifiers",
        variables=(
            TermMapVariable(source="D3", name="K3", offset=1.0),
            TermMapVariable(source="D4", name="K4", offset=1.0),
            TermMapVariable(source="CT1", name="KT", offset=1.0),
        ),
    ),
}


def parse_term_maps(data: dict[str, Any] | None) -> dict[str, TermMap]:
    data = data or {}
    return {
        str(name): TermMap.from_dict(str(name), value)
        for name, value in data.items()
    }


def available_term_map_names(config: "ProjectConfig") -> list[str]:
    names = set(BUILTIN_TERM_MAPS)
    names.update(config.term_maps)
    return sorted(names)


def resolve_term_map(
    config: "ProjectConfig",
    name: str,
    source_names: Sequence[str] | None = None,
) -> ResolvedTermMap:
    term_map = config.term_maps.get(name, BUILTIN_TERM_MAPS.get(name))
    if term_map is None:
        available = ", ".join(available_term_map_names(config)) or "none"
        raise ValueError(f"Unknown term map {name!r}. Available term maps: {available}")

    if source_names is None:
        alias_sets = [
            (coupling.parameter, coupling.name, coupling.fit_name)
            for coupling in config.couplings
        ]
        fallback_names = [coupling.name for coupling in config.couplings]
    else:
        alias_sets = [(str(source),) for source in source_names]
        fallback_names = [str(source) for source in source_names]

    names: list[str] = []
    offsets: list[float] = []
    active_variables: list[TermMapVariable | None] = []
    for aliases, fallback_name in zip(alias_sets, fallback_names):
        variable = find_variable(term_map, aliases)
        if variable is None:
            names.append(fallback_name)
            offsets.append(0.0)
            active_variables.append(None)
        else:
            names.append(variable.name)
            offsets.append(variable.offset)
            active_variables.append(variable)

    if not any(variable is not None for variable in active_variables):
        raise ValueError(f"Term map {name!r} does not match any configured coupling")

    return ResolvedTermMap(
        term_map=term_map,
        names=tuple(names),
        offsets=tuple(offsets),
        active_variables=tuple(active_variables),
    )


def find_variable(term_map: TermMap, aliases: Sequence[str]) -> TermMapVariable | None:
    alias_set = set(aliases)
    for variable in term_map.variables:
        if variable.source in alias_set:
            return variable
    return None


def format_mapping(variable: TermMapVariable) -> str:
    offset = variable.offset
    if abs(offset) < 1e-15:
        return f"{variable.name}={variable.source}"
    if offset > 0.0:
        return f"{variable.name}={offset:g}+{variable.source}"
    return f"{variable.name}={variable.source}{offset:g}"


def transformed_support(powers: Sequence[Power], term_map: ResolvedTermMap) -> tuple[Power, ...]:
    mapped: set[Power] = set()
    for power in powers:
        mapped.update(expand_power(power, term_map.offsets))
    return sort_powers(mapped)


def transform_coefficients(
    powers: Sequence[Power],
    coefficients: np.ndarray,
    covariance: np.ndarray,
    term_map: ResolvedTermMap,
) -> tuple[list[Power], np.ndarray, np.ndarray]:
    mapped_powers, transform = affine_transform_matrix(powers, term_map.offsets)
    mapped_coefficients = np.dot(transform, coefficients)
    mapped_covariance = np.dot(transform, np.dot(covariance, transform.T))
    return mapped_powers, mapped_coefficients, mapped_covariance


def transform_mapped_coefficients_to_sources(
    powers: Sequence[Power],
    coefficients: np.ndarray,
    covariance: np.ndarray,
    term_map: ResolvedTermMap,
) -> tuple[list[Power], np.ndarray, np.ndarray]:
    mapped_powers, transform = affine_transform_matrix(
        powers,
        [-offset for offset in term_map.offsets],
    )
    mapped_coefficients = np.dot(transform, coefficients)
    mapped_covariance = np.dot(transform, np.dot(covariance, transform.T))
    return mapped_powers, mapped_coefficients, mapped_covariance


def affine_transform_matrix(
    powers: Sequence[Power],
    offsets: Sequence[float],
) -> tuple[list[Power], np.ndarray]:
    expansions = [expand_power(power, offsets) for power in powers]
    mapped_powers = list(sort_powers(power for expansion in expansions for power in expansion))
    row_index = {power: index for index, power in enumerate(mapped_powers)}
    transform = np.zeros((len(mapped_powers), len(powers)), dtype=float)
    for col, expansion in enumerate(expansions):
        for power, coeff in expansion.items():
            transform[row_index[power], col] += coeff
    return mapped_powers, transform


def expand_power(power: Power, offsets: Sequence[float]) -> dict[Power, float]:
    expansion: dict[Power, float] = {tuple([0] * len(power)): 1.0}
    for dim, exponent in enumerate(power):
        one_dim = {
            out_power: comb(exponent, out_power) * ((-offsets[dim]) ** (exponent - out_power))
            for out_power in range(exponent + 1)
        }
        next_expansion: dict[Power, float] = {}
        for powers, coeff in expansion.items():
            for out_power, out_coeff in one_dim.items():
                if abs(out_coeff) < 1e-15:
                    continue
                next_powers = list(powers)
                next_powers[dim] += out_power
                key = tuple(next_powers)
                next_expansion[key] = next_expansion.get(key, 0.0) + coeff * out_coeff
        expansion = next_expansion
    return {
        powers: coeff
        for powers, coeff in expansion.items()
        if abs(coeff) > 1e-15
    }


def sort_powers(powers: Iterable[Power]) -> tuple[Power, ...]:
    return tuple(sorted(set(powers), key=lambda power: (sum(power), tuple(-item for item in power))))


def format_power_label(power: Power, names: Sequence[str]) -> str:
    pieces = []
    for name, exponent in zip(names, power):
        if exponent == 0:
            continue
        if exponent == 1:
            pieces.append(name)
        else:
            pieces.append(f"{name}^{exponent}")
    return "*".join(pieces) if pieces else "1"


def format_factored_power_label(power: Power, term_map: ResolvedTermMap) -> str:
    pieces = []
    for exponent, name, variable in zip(power, term_map.names, term_map.active_variables):
        if exponent == 0:
            continue
        factor = name if variable is None else inverse_factor_label(variable)
        if exponent == 1:
            pieces.append(factor)
        else:
            pieces.append(f"{factor}^{exponent}")
    return "*".join(pieces) if pieces else "1"


def inverse_factor_label(variable: TermMapVariable) -> str:
    offset = variable.offset
    if abs(offset) < 1e-15:
        return variable.name
    if offset > 0.0:
        return f"({variable.name}-{offset:g})"
    return f"({variable.name}+{-offset:g})"
