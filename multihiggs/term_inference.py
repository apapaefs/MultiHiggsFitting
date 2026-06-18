from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .term_maps import (
    ResolvedTermMap,
    expand_power,
    format_factored_power_label,
    format_power_label,
    transformed_support,
)


Power = tuple[int, ...]
Support = set[Power]
StateKey = tuple[str, str]
CoefficientKey = tuple[str, ...]
Coefficient = dict[CoefficientKey, float]
SymbolicPolynomial = dict[Power, Coefficient]


class TermInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class InferredTerms:
    process_dir: Path
    coupling_names: tuple[str, ...]
    amplitude_terms: tuple[Power, ...]
    cross_section_terms: tuple[Power, ...]
    amplitude_support_counts: dict[tuple[Power, ...], int]
    coupling_supports: dict[str, tuple[Power, ...]]
    matrix_files: tuple[Path, ...]
    physical_basis: bool = False
    source_names: tuple[str, ...] = ()
    term_map_name: str | None = None
    mheft_squared_order_cap: int | None = None
    amplitude_basis: str | None = None


def infer_terms_from_process_dir(
    process_dir: Path,
    coupling_names: Iterable[str],
    term_map: ResolvedTermMap | None = None,
    physical_basis: bool = False,
    mheft_squared_order_cap: int | None = None,
    amplitude_basis: str | None = None,
) -> InferredTerms:
    process_dir = Path(process_dir)
    source_names = tuple(str(name) for name in coupling_names)
    if not process_dir.exists():
        raise TermInferenceError(f"Generated process directory does not exist: {process_dir}")

    matrix_files = find_matrix_files(process_dir)
    if not matrix_files:
        raise TermInferenceError(f"No generated matrix source files found under {process_dir}")

    if physical_basis:
        if term_map is None:
            raise TermInferenceError("physical-basis inference requires a term map")
        coupling_names = term_map.names
        coupling_values = read_coupling_values(process_dir)
        used_couplings = collect_used_couplings(matrix_files, set(coupling_values))
        coupling_supports = parse_physical_coupling_supports(
            coupling_values,
            source_names,
            term_map,
            used_couplings,
        )
    else:
        coupling_names = source_names
        coupling_supports = read_coupling_supports(process_dir, coupling_names)

    zero = zero_power(coupling_names)
    amplitude_support_counts: Counter[tuple[Power, ...]] = Counter()
    amplitude_terms: set[Power] = set()
    for matrix_group in matrix_file_groups(matrix_files):
        state: dict[StateKey, Support] = {}
        for matrix_file in matrix_group:
            for support in infer_matrix_amplitude_supports(matrix_file, coupling_supports, zero, state):
                clean_support = sort_powers(support)
                amplitude_support_counts[clean_support] += 1
                amplitude_terms.update(clean_support)

    amplitude_terms = project_amplitude_terms(amplitude_terms, coupling_names, amplitude_basis)
    if not amplitude_terms:
        amplitude_terms.add(zero)
    cross_section_terms = {
        add_powers(left, right)
        for left in amplitude_terms
        for right in amplitude_terms
    }
    if mheft_squared_order_cap is not None:
        cross_section_terms = {
            power
            for power in cross_section_terms
            if sum(power) <= mheft_squared_order_cap
        }
    return InferredTerms(
        process_dir=process_dir,
        coupling_names=coupling_names,
        amplitude_terms=sort_powers(amplitude_terms),
        cross_section_terms=sort_powers(cross_section_terms),
        amplitude_support_counts=dict(sorted(amplitude_support_counts.items())),
        coupling_supports={
            name: sort_powers(support)
            for name, support in sorted(coupling_supports.items())
            if support != {zero}
        },
        matrix_files=tuple(matrix_files),
        physical_basis=physical_basis,
        source_names=source_names,
        term_map_name=None if term_map is None else term_map.name,
        mheft_squared_order_cap=mheft_squared_order_cap,
        amplitude_basis=amplitude_basis,
    )


def read_coupling_supports(process_dir: Path, coupling_names: tuple[str, ...]) -> dict[str, Support]:
    coupling_values = read_coupling_values(process_dir)
    if not coupling_values:
        model_dir = process_dir / "bin" / "internal" / "ufomodel"
        raise TermInferenceError(f"No UFO coupling definitions found under {model_dir}")
    return {
        name: polynomial_support(value, coupling_names)
        for name, value in coupling_values.items()
    }


def read_coupling_values(process_dir: Path) -> dict[str, str]:
    model_dir = process_dir / "bin" / "internal" / "ufomodel"
    coupling_files = [model_dir / "couplings.py", model_dir / "CT_couplings.py"]
    values: dict[str, str] = {}
    for path in coupling_files:
        if path.exists():
            values.update(parse_ufo_coupling_values(path))
    return values


def parse_ufo_coupling_values(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"Coupling\s*\(\s*name\s*=\s*['\"](?P<name>[^'\"]+)['\"].*?"
        r"value\s*=\s*['\"](?P<value>[^'\"]*)['\"]",
        re.DOTALL,
    )
    return {
        match.group("name"): match.group("value")
        for match in pattern.finditer(text)
    }


def parse_physical_coupling_supports(
    coupling_values: dict[str, str],
    source_names: tuple[str, ...],
    term_map: ResolvedTermMap,
    used_couplings: set[str],
) -> dict[str, Support]:
    supports: dict[str, Support] = {}
    for name, value in coupling_values.items():
        try:
            supports[name] = physical_polynomial_support(value, source_names, term_map)
        except TermInferenceError as exc:
            raise TermInferenceError(f"Could not infer physical support for {name}: {exc}") from exc
    if not supports:
        raise TermInferenceError("No UFO coupling definitions found")
    apply_restricted5_physical_aliases(supports, term_map, used_couplings)
    return supports


def parse_ufo_coupling_file(path: Path, coupling_names: tuple[str, ...]) -> dict[str, Support]:
    supports: dict[str, Support] = {}
    for name, value in parse_ufo_coupling_values(path).items():
        try:
            supports[name] = polynomial_support(value, coupling_names)
        except TermInferenceError as exc:
            raise TermInferenceError(f"Could not infer support for {name} in {path}: {exc}") from exc
    return supports


def polynomial_support(expression: str, coupling_names: tuple[str, ...]) -> Support:
    zero = zero_power(coupling_names)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise TermInferenceError(f"cannot parse expression {expression!r}") from exc
    return support_for_node(tree.body, coupling_names, zero)


def physical_polynomial_support(
    expression: str,
    source_names: tuple[str, ...],
    term_map: ResolvedTermMap,
) -> Support:
    polynomial = symbolic_polynomial(expression, source_names)
    transformed: SymbolicPolynomial = {}
    for power, coefficient in polynomial.items():
        for mapped_power, numeric in expand_power(power, term_map.offsets).items():
            transformed[mapped_power] = coefficient_add(
                transformed.get(mapped_power, coefficient_zero()),
                coefficient_scale(coefficient, numeric),
            )
    return {
        power
        for power, coefficient in transformed.items()
        if not coefficient_is_zero(coefficient)
    }


def symbolic_polynomial(expression: str, variable_names: tuple[str, ...]) -> SymbolicPolynomial:
    zero = zero_power(variable_names)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise TermInferenceError(f"cannot parse expression {expression!r}") from exc
    return symbolic_support_for_node(tree.body, variable_names, zero)


def symbolic_support_for_node(
    node: ast.AST,
    variable_names: tuple[str, ...],
    zero: Power,
) -> SymbolicPolynomial:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return {zero: coefficient_number(float(node.value))}
        return {zero: coefficient_factor(canonical_node(node))}
    if isinstance(node, ast.Name):
        if node.id in variable_names:
            power = [0] * len(variable_names)
            power[variable_names.index(node.id)] = 1
            return {tuple(power): coefficient_one()}
        return {zero: coefficient_factor(node.id)}
    if isinstance(node, ast.Attribute):
        return {zero: coefficient_factor(canonical_node(node))}
    if isinstance(node, ast.Call):
        if node_contains_variables(node, variable_names):
            raise TermInferenceError("function call with scan-dependent argument is not polynomial")
        return {zero: coefficient_factor(canonical_node(node))}
    if isinstance(node, ast.UnaryOp):
        operand = symbolic_support_for_node(node.operand, variable_names, zero)
        if isinstance(node.op, ast.USub):
            return symbolic_poly_scale(operand, -1.0)
        if isinstance(node.op, ast.UAdd):
            return operand
        return {zero: coefficient_factor(canonical_node(node))}
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            return symbolic_poly_add(
                symbolic_support_for_node(node.left, variable_names, zero),
                symbolic_support_for_node(node.right, variable_names, zero),
            )
        if isinstance(node.op, ast.Sub):
            return symbolic_poly_add(
                symbolic_support_for_node(node.left, variable_names, zero),
                symbolic_poly_scale(symbolic_support_for_node(node.right, variable_names, zero), -1.0),
            )
        if isinstance(node.op, ast.Mult):
            return symbolic_poly_multiply(
                symbolic_support_for_node(node.left, variable_names, zero),
                symbolic_support_for_node(node.right, variable_names, zero),
            )
        if isinstance(node.op, ast.Div):
            if node_contains_variables(node.right, variable_names):
                raise TermInferenceError("division by scan-dependent expression is not polynomial")
            left = symbolic_support_for_node(node.left, variable_names, zero)
            divisor = numeric_constant(node.right)
            if divisor is not None:
                return symbolic_poly_scale(left, 1.0 / divisor)
            return symbolic_poly_multiply(
                left,
                {zero: coefficient_factor(f"1/({canonical_node(node.right)})")},
            )
        if isinstance(node.op, ast.Pow):
            exponent = integer_exponent(node.right)
            if exponent is None:
                if node_contains_variables(node.left, variable_names):
                    raise TermInferenceError("non-integer scan-dependent power")
                return {zero: coefficient_factor(canonical_node(node))}
            if node_contains_variables(node.left, variable_names):
                if exponent < 0:
                    raise TermInferenceError("negative scan-dependent power")
                result = {zero: coefficient_one()}
                base = symbolic_support_for_node(node.left, variable_names, zero)
                for _ in range(exponent):
                    result = symbolic_poly_multiply(result, base)
                return result
            constant = numeric_constant(node.left)
            if constant is not None:
                return {zero: coefficient_number(constant**exponent)}
            return {zero: coefficient_factor(canonical_node(node))}
    raise TermInferenceError(f"unsupported expression node: {type(node).__name__}")


def node_contains_variables(node: ast.AST, variable_names: tuple[str, ...]) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in variable_names:
            return True
    return False


def numeric_constant(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = numeric_constant(node.operand)
        return None if value is None else -value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return numeric_constant(node.operand)
    return None


def canonical_node(node: ast.AST) -> str:
    return ast.unparse(node).replace(" ", "")


def symbolic_poly_add(left: SymbolicPolynomial, right: SymbolicPolynomial) -> SymbolicPolynomial:
    result = dict(left)
    for power, coefficient in right.items():
        result[power] = coefficient_add(result.get(power, coefficient_zero()), coefficient)
    return {
        power: coefficient
        for power, coefficient in result.items()
        if not coefficient_is_zero(coefficient)
    }


def symbolic_poly_scale(polynomial: SymbolicPolynomial, scale: float) -> SymbolicPolynomial:
    return {
        power: scaled
        for power, coefficient in polynomial.items()
        if not coefficient_is_zero(scaled := coefficient_scale(coefficient, scale))
    }


def symbolic_poly_multiply(left: SymbolicPolynomial, right: SymbolicPolynomial) -> SymbolicPolynomial:
    result: SymbolicPolynomial = {}
    for left_power, left_coefficient in left.items():
        for right_power, right_coefficient in right.items():
            power = add_powers(left_power, right_power)
            coefficient = coefficient_multiply(left_coefficient, right_coefficient)
            result[power] = coefficient_add(result.get(power, coefficient_zero()), coefficient)
    return {
        power: coefficient
        for power, coefficient in result.items()
        if not coefficient_is_zero(coefficient)
    }


def coefficient_zero() -> Coefficient:
    return {}


def coefficient_one() -> Coefficient:
    return {(): 1.0}


def coefficient_number(value: float) -> Coefficient:
    if abs(value) < 1e-15:
        return {}
    return {(): float(value)}


def coefficient_factor(factor: str) -> Coefficient:
    return {(factor,): 1.0}


def coefficient_add(left: Coefficient, right: Coefficient) -> Coefficient:
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, 0.0) + value
        if abs(result[key]) < 1e-12:
            del result[key]
    return result


def coefficient_scale(coefficient: Coefficient, scale: float) -> Coefficient:
    if abs(scale) < 1e-15:
        return {}
    return {
        key: value * scale
        for key, value in coefficient.items()
        if abs(value * scale) > 1e-12
    }


def coefficient_multiply(left: Coefficient, right: Coefficient) -> Coefficient:
    result: Coefficient = {}
    for left_key, left_value in left.items():
        for right_key, right_value in right.items():
            key = tuple(sorted(left_key + right_key))
            result[key] = result.get(key, 0.0) + left_value * right_value
            if abs(result[key]) < 1e-12:
                del result[key]
    return result


def coefficient_is_zero(coefficient: Coefficient) -> bool:
    return not coefficient


def apply_restricted5_physical_aliases(
    supports: dict[str, Support],
    term_map: ResolvedTermMap,
    used_couplings: set[str],
) -> None:
    aliases = {
        "CT1": ("GC_37", "GC_TTH_MHEFT"),
        "D3": ("GC_30", "GC_HHH_MHEFT"),
        "D4": ("GC_HHHH", "GC_HHHH_MHEFT"),
    }
    for dim, variable in enumerate(term_map.active_variables):
        if variable is None:
            continue
        pair = aliases.get(variable.source)
        if pair is None:
            continue
        if not all(name in supports and name in used_couplings for name in pair):
            continue
        power = [0] * len(term_map.names)
        power[dim] = 1
        support = {tuple(power)}
        for name in pair:
            supports[name] = support


def support_for_node(node: ast.AST, coupling_names: tuple[str, ...], zero: Power) -> Support:
    if isinstance(node, ast.Constant):
        return {zero}
    if isinstance(node, ast.Name):
        if node.id in coupling_names:
            power = [0] * len(coupling_names)
            power[coupling_names.index(node.id)] = 1
            return {tuple(power)}
        return {zero}
    if isinstance(node, ast.UnaryOp):
        return support_for_node(node.operand, coupling_names, zero)
    if isinstance(node, ast.BinOp):
        left = support_for_node(node.left, coupling_names, zero)
        right = support_for_node(node.right, coupling_names, zero)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return left | right
        if isinstance(node.op, ast.Mult):
            return multiply_supports(left, right)
        if isinstance(node.op, ast.Div):
            if right != {zero}:
                raise TermInferenceError("division by scan-dependent expression is not polynomial")
            return left
        if isinstance(node.op, ast.Pow):
            exponent = integer_exponent(node.right)
            if exponent is None or exponent < 0:
                if left == {zero}:
                    return {zero}
                raise TermInferenceError("non-integer or negative scan-dependent power")
            result = {zero}
            for _ in range(exponent):
                result = multiply_supports(result, left)
            return result
    if isinstance(node, ast.Call):
        supports = [support_for_node(arg, coupling_names, zero) for arg in node.args]
        for keyword in node.keywords:
            supports.append(support_for_node(keyword.value, coupling_names, zero))
        if all(support == {zero} for support in supports):
            return {zero}
        raise TermInferenceError("function call with scan-dependent argument is not polynomial")
    if isinstance(node, ast.Attribute):
        return {zero}
    raise TermInferenceError(f"unsupported expression node: {type(node).__name__}")


def integer_exponent(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = integer_exponent(node.operand)
        return None if value is None else -value
    return None


def find_matrix_files(process_dir: Path) -> list[Path]:
    subprocess_dir = process_dir / "SubProcesses"
    preferred = sorted(subprocess_dir.glob("P*/matrix*_orig.f"))
    if preferred:
        return preferred
    candidates = [
        path
        for path in subprocess_dir.glob("P*/matrix*.f")
        if not path.name.startswith("template_")
    ]
    candidates.extend(subprocess_dir.glob("PV*/helas_calls_amp*.f"))
    candidates.extend(subprocess_dir.glob("PV*/coef_construction*.f"))
    return sorted(candidates, key=matrix_file_sort_key)


def matrix_file_sort_key(path: Path) -> tuple[str, int, str]:
    name = path.name
    if name.startswith("helas_calls_amp"):
        order = 0
    elif name.startswith("loop_CT_calls"):
        order = 1
    elif name.startswith("coef_construction"):
        order = 2
    else:
        order = 0
    return (str(path.parent), order, name)


def matrix_file_groups(matrix_files: Iterable[Path]) -> tuple[tuple[Path, ...], ...]:
    groups: dict[Path, list[Path]] = {}
    for matrix_file in sorted(matrix_files, key=matrix_file_sort_key):
        groups.setdefault(matrix_file.parent, []).append(matrix_file)
    return tuple(tuple(files) for files in groups.values())


def infer_matrix_amplitude_supports(
    matrix_file: Path,
    coupling_supports: dict[str, Support],
    zero: Power,
    state: dict[StateKey, Support] | None = None,
) -> list[Support]:
    state = {} if state is None else state
    amplitudes: list[Support] = []
    for statement in iter_fortran_call_statements(matrix_file):
        tokens = call_tokens(statement, coupling_supports)
        if not tokens:
            continue
        if is_loop_coefficient_creation(statement):
            support = first_state_support(tokens, state, zero)
            if support is not None:
                amplitudes.append(support)
            continue
        output_index = output_token_index(tokens)
        if output_index is None:
            continue
        output_type, output_name = tokens[output_index]
        if is_loop_wavefunction_update(statement):
            support = coefs_support(tokens[:output_index], state, zero)
        else:
            inputs = input_supports(tokens[:output_index], state, coupling_supports, zero)
            support = product_supports(inputs, zero)
        if output_type == "AMP":
            amplitudes.append(support)
        else:
            state[(output_type, output_name)] = support
            if tokens[-1][0] == "COEFS":
                state[tokens[-1]] = support
    return amplitudes


def is_loop_coefficient_creation(statement: str) -> bool:
    return fortran_call_name(statement).endswith("CREATE_LOOP_COEFS")


def is_loop_wavefunction_update(statement: str) -> bool:
    return "_UPDATE_WL_" in fortran_call_name(statement)


def fortran_call_name(statement: str) -> str:
    match = re.search(r"\bCALL\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)", statement, re.IGNORECASE)
    return "" if match is None else match.group("name").upper()


def output_token_index(tokens: list[tuple[str, str]]) -> int | None:
    if tokens[-1][0] in {"AMP", "W", "PL", "WL"}:
        return len(tokens) - 1
    if tokens[-1][0] == "COEFS":
        for index in range(len(tokens) - 2, -1, -1):
            if tokens[index][0] in {"W", "PL", "WL"}:
                return index
    return None


def first_state_support(
    tokens: list[tuple[str, str]],
    state: dict[StateKey, Support],
    zero: Power,
) -> Support | None:
    for token_type, token_name in tokens:
        if token_type in {"W", "PL", "WL", "COEFS"}:
            return state.get((token_type, token_name), {zero})
    return None


def input_supports(
    tokens: Iterable[tuple[str, str]],
    state: dict[StateKey, Support],
    coupling_supports: dict[str, Support],
    zero: Power,
) -> list[Support]:
    supports: list[Support] = []
    for token_type, token_name in tokens:
        if token_type in {"W", "PL", "WL", "COEFS"}:
            supports.append(state.get((token_type, token_name), {zero}))
        elif token_type == "COUPLING":
            supports.append(coupling_supports[token_name])
    return supports


def coefs_support(
    tokens: Iterable[tuple[str, str]],
    state: dict[StateKey, Support],
    zero: Power,
) -> Support:
    for token_type, token_name in tokens:
        if token_type == "COEFS":
            return state.get((token_type, token_name), {zero})
    return {zero}


def iter_fortran_call_statements(path: Path):
    current: str | None = None
    balance = 0
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = strip_fortran_line(raw_line)
        if not line:
            continue
        if current is None:
            if not re.search(r"\bCALL\b", line, re.IGNORECASE):
                continue
            current = line
            balance = paren_balance(line)
        else:
            current += " " + line
            balance += paren_balance(line)
        if current is not None and balance <= 0:
            yield current
            current = None
            balance = 0


def strip_fortran_line(raw_line: str) -> str:
    if raw_line and raw_line[0] in ("C", "c", "*", "!"):
        return ""
    stripped = raw_line.strip()
    if not stripped:
        return ""
    if stripped.startswith("$"):
        stripped = stripped[1:].strip()
    return stripped


def paren_balance(text: str) -> int:
    return text.count("(") - text.count(")")


def call_tokens(statement: str, coupling_supports: dict[str, Support]) -> list[tuple[str, str]]:
    match = re.search(r"\bCALL\s+\w+\s*\((?P<args>.*)\)\s*$", statement, re.IGNORECASE)
    if not match:
        return []
    args = match.group("args")
    token_pattern = re.compile(
        r"W\(\s*1\s*,\s*(?P<w>\d+)\s*\)"
        r"|PL\(\s*0\s*,\s*(?P<pl>-?\d+)\s*\)"
        r"|WL\(\s*1\s*,\s*0\s*,\s*1\s*,\s*(?P<wl>-?\d+)\s*\)"
        r"|AMP\(\s*(?P<amp>\d+)\s*\)"
        r"|AMPL\(\s*(?P<ampl>\d+\s*,\s*\d+)\s*\)"
        r"|(?P<coefs>\bCOEFS\b)"
        r"|(?P<name>\b[A-Za-z_][A-Za-z0-9_]*\b)"
    )
    tokens: list[tuple[str, str]] = []
    for token in token_pattern.finditer(args):
        if token.group("w"):
            tokens.append(("W", token.group("w")))
        elif token.group("pl"):
            tokens.append(("PL", token.group("pl")))
        elif token.group("wl"):
            tokens.append(("WL", token.group("wl")))
        elif token.group("amp") or token.group("ampl"):
            tokens.append(("AMP", token.group("amp") or token.group("ampl").replace(" ", "")))
        elif token.group("coefs"):
            tokens.append(("COEFS", "COEFS"))
        else:
            name = token.group("name")
            if name in coupling_supports:
                tokens.append(("COUPLING", name))
    return tokens


def collect_used_couplings(matrix_files: Iterable[Path], coupling_names: set[str]) -> set[str]:
    used: set[str] = set()
    support_placeholders = {name: set() for name in coupling_names}
    for matrix_file in matrix_files:
        for statement in iter_fortran_call_statements(matrix_file):
            for token_type, token_name in call_tokens(statement, support_placeholders):
                if token_type == "COUPLING":
                    used.add(token_name)
    return used


def product_supports(supports: Iterable[Support], zero: Power) -> Support:
    result = {zero}
    for support in supports:
        result = multiply_supports(result, support)
    return result


def multiply_supports(left: Support, right: Support) -> Support:
    return {add_powers(left_power, right_power) for left_power in left for right_power in right}


def add_powers(left: Power, right: Power) -> Power:
    return tuple(a + b for a, b in zip(left, right))


def zero_power(coupling_names: tuple[str, ...]) -> Power:
    return tuple(0 for _ in coupling_names)


def sort_powers(powers: Iterable[Power]) -> tuple[Power, ...]:
    return tuple(sorted(powers, key=lambda power: (sum(power), tuple(-item for item in power))))


def project_amplitude_terms(
    amplitude_terms: Iterable[Power],
    coupling_names: tuple[str, ...],
    amplitude_basis: str | None,
) -> set[Power]:
    terms = set(amplitude_terms)
    if amplitude_basis is None:
        return terms
    if amplitude_basis == "sm_like_hhh":
        return project_sm_like_hhh_amplitude_terms(terms, coupling_names)
    raise TermInferenceError(f"Unknown amplitude basis {amplitude_basis!r}")


def project_sm_like_hhh_amplitude_terms(
    amplitude_terms: Iterable[Power],
    coupling_names: tuple[str, ...],
) -> set[Power]:
    sm_like_names = ("KT", "K3", "K4")
    present_sm = tuple(
        (name, coupling_names.index(name), sm_position)
        for sm_position, name in enumerate(sm_like_names)
        if name in coupling_names
    )
    sm_indices = {index for _, index, _ in present_sm}
    full_allowed_sm_subpowers = {
        (3, 0, 0),
        (2, 1, 0),
        (1, 2, 0),
        (1, 0, 1),
    }
    allowed_sm_subpowers = {
        tuple(full_power[sm_position] for _, _, sm_position in present_sm)
        for full_power in full_allowed_sm_subpowers
    }
    projected: set[Power] = set()
    for power in amplitude_terms:
        has_non_sm_variable = any(
            exponent != 0
            for index, exponent in enumerate(power)
            if index not in sm_indices
        )
        if has_non_sm_variable:
            projected.add(power)
            continue
        subpower = tuple(power[index] for _, index, _ in present_sm)
        if subpower in allowed_sm_subpowers:
            projected.add(power)
    return projected


def format_inferred_terms(
    result: InferredTerms,
    term_map: ResolvedTermMap | None = None,
    expand_term_map: bool = False,
) -> str:
    coupling_text = ",".join(result.source_names if result.physical_basis else result.coupling_names)
    lines = [
        f"Process directory: {result.process_dir}",
        f"Scanned parameters: {coupling_text}",
    ]
    if result.mheft_squared_order_cap is not None:
        lines.append(f"Applied MHEFT^2 order cap: <= {result.mheft_squared_order_cap}")
    if result.amplitude_basis is not None:
        lines.append(f"Applied amplitude basis: {result.amplitude_basis}")
    if result.physical_basis:
        lines.append(f"Physical-basis fit variables: {','.join(result.coupling_names)}")
        if term_map is not None:
            mapping_text = term_map.mapping_text()
            if mapping_text:
                lines.append(f"Mapping: {mapping_text}")
    lines.append("Inferred amplitude support:")
    for power in result.amplitude_terms:
        lines.append(f"  {format_power(power)}")
    lines.append("Generated AMP support groups:")
    for support, count in sorted(
        result.amplitude_support_counts.items(),
        key=lambda item: (len(item[0]), item[0]),
    ):
        support_text = ", ".join(format_power(power) for power in support)
        lines.append(f"  {count} amplitude(s): {support_text}")
    lines.append("Scan-dependent UFO couplings used:")
    if result.coupling_supports:
        for name, support in result.coupling_supports.items():
            support_text = ", ".join(format_power(power) for power in support)
            lines.append(f"  {name}: {support_text}")
    else:
        lines.append("  none")
    if result.physical_basis:
        lines.append("Inferred physical-basis cross-section [fit].terms:")
    else:
        lines.append("Inferred cross-section [fit].terms:")
    lines.append("terms = [")
    for power in result.cross_section_terms:
        lines.append(f"  {format_power(power)},")
    lines.append("]")
    if result.physical_basis:
        lines.append("Inferred physical-basis cross-section polynomial powers:")
    else:
        lines.append("Inferred cross-section polynomial powers:")
    for power in result.cross_section_terms:
        lines.append(f"  {format_monomial(power, result.coupling_names)}")
    lines.append(f"Number of polynomial terms: {len(result.cross_section_terms)}")
    if term_map is not None and not result.physical_basis:
        variable_text = ",".join(term_map.names)
        lines.append(f"Inferred cross-section minimal polynomial powers in term map {term_map.name} ({variable_text}):")
        mapping_text = term_map.mapping_text()
        if mapping_text:
            lines.append(f"Mapping: {mapping_text}")
        for power in result.cross_section_terms:
            lines.append(f"  {format_factored_power_label(power, term_map)}")
        lines.append(f"Number of minimal polynomial terms: {len(result.cross_section_terms)}")
    if term_map is not None and expand_term_map and not result.physical_basis:
        variable_text = ",".join(term_map.names)
        expanded_terms = transformed_support(result.cross_section_terms, term_map)
        lines.append(f"Inferred expanded cross-section polynomial powers in term map {term_map.name} ({variable_text}):")
        for power in expanded_terms:
            lines.append(f"  {format_power_label(power, term_map.names)}")
        lines.append(f"Number of expanded polynomial terms: {len(expanded_terms)}")
    lines.append("WARNING: CHECK inferred fit terms before using them.")
    return "\n".join(lines)


def format_monomial(power: Power, coupling_names: tuple[str, ...]) -> str:
    pieces = []
    for name, exponent in zip(coupling_names, power):
        if exponent == 0:
            continue
        if exponent == 1:
            pieces.append(name)
        else:
            pieces.append(f"{name}^{exponent}")
    if not pieces:
        return "1"
    return "*".join(pieces)


def format_terms_block(terms: Iterable[Power]) -> str:
    lines = ["terms = ["]
    for power in terms:
        lines.append(f"  {format_power(power)},")
    lines.append("]")
    return "\n".join(lines)


def update_config_fit_terms(
    config_path: Path,
    terms: Iterable[Power],
    basis: str | None = None,
    term_map_name: str | None = None,
) -> Path:
    config_path = Path(config_path)
    text = config_path.read_text(encoding="utf-8")
    fit_header = re.search(r"(?m)^\[fit\]\s*$", text)
    if not fit_header:
        raise TermInferenceError(f"Config has no [fit] section: {config_path}")

    section_start = fit_header.end()
    next_header = re.search(r"(?m)^\[", text[section_start:])
    section_end = section_start + next_header.start() if next_header else len(text)
    section = text[section_start:section_end]
    terms_match = re.search(r"(?m)^[ \t]*terms[ \t]*=", section)
    new_block = format_terms_block(terms)

    if terms_match:
        terms_start = section_start + terms_match.start()
        value_start = section_start + terms_match.end()
        bracket_start = text.find("[", value_start, section_end)
        if bracket_start == -1:
            raise TermInferenceError(f"Could not find [fit].terms list in {config_path}")
        bracket_end = matching_bracket_end(text, bracket_start)
        text = text[:terms_start] + new_block + text[bracket_end:]
    else:
        insertion = section_start
        text = text[:insertion] + "\n" + new_block + "\n" + text[insertion:]

    if basis is not None:
        text = upsert_fit_string(text, "basis", basis)
    if term_map_name is not None:
        text = upsert_fit_string(text, "term_map", term_map_name)

    config_path.write_text(text, encoding="utf-8")
    return config_path


def upsert_fit_string(text: str, key: str, value: str) -> str:
    fit_header = re.search(r"(?m)^\[fit\]\s*$", text)
    if not fit_header:
        raise TermInferenceError("Config has no [fit] section")
    section_start = fit_header.end()
    next_header = re.search(r"(?m)^\[", text[section_start:])
    section_end = section_start + next_header.start() if next_header else len(text)
    section = text[section_start:section_end]
    line = f'{key} = "{value}"'
    match = re.search(rf"(?m)^[ \t]*{re.escape(key)}[ \t]*=.*$", section)
    if match:
        start = section_start + match.start()
        end = section_start + match.end()
        return text[:start] + line + text[end:]
    return text[:section_start] + "\n" + line + text[section_start:]


def matching_bracket_end(text: str, bracket_start: int) -> int:
    depth = 0
    for index in range(bracket_start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index + 1
    raise TermInferenceError("Could not find end of terms list")


def format_power(power: Power) -> str:
    return "[" + ", ".join(str(item) for item in power) + "]"
