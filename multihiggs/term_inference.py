from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


Power = tuple[int, ...]
Support = set[Power]


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


def infer_terms_from_process_dir(
    process_dir: Path,
    coupling_names: Iterable[str],
) -> InferredTerms:
    process_dir = Path(process_dir)
    coupling_names = tuple(str(name) for name in coupling_names)
    if not process_dir.exists():
        raise TermInferenceError(f"Generated process directory does not exist: {process_dir}")

    coupling_supports = read_coupling_supports(process_dir, coupling_names)
    matrix_files = find_matrix_files(process_dir)
    if not matrix_files:
        raise TermInferenceError(f"No generated matrix source files found under {process_dir}")

    zero = zero_power(coupling_names)
    amplitude_support_counts: Counter[tuple[Power, ...]] = Counter()
    amplitude_terms: set[Power] = set()
    for matrix_file in matrix_files:
        for support in infer_matrix_amplitude_supports(matrix_file, coupling_supports, zero):
            clean_support = sort_powers(support)
            amplitude_support_counts[clean_support] += 1
            amplitude_terms.update(clean_support)

    if not amplitude_terms:
        amplitude_terms.add(zero)
    cross_section_terms = {
        add_powers(left, right)
        for left in amplitude_terms
        for right in amplitude_terms
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
    )


def read_coupling_supports(process_dir: Path, coupling_names: tuple[str, ...]) -> dict[str, Support]:
    model_dir = process_dir / "bin" / "internal" / "ufomodel"
    coupling_files = [model_dir / "couplings.py", model_dir / "CT_couplings.py"]
    supports: dict[str, Support] = {}
    for path in coupling_files:
        if not path.exists():
            continue
        supports.update(parse_ufo_coupling_file(path, coupling_names))
    if not supports:
        raise TermInferenceError(f"No UFO coupling definitions found under {model_dir}")
    return supports


def parse_ufo_coupling_file(path: Path, coupling_names: tuple[str, ...]) -> dict[str, Support]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"Coupling\s*\(\s*name\s*=\s*['\"](?P<name>[^'\"]+)['\"].*?"
        r"value\s*=\s*['\"](?P<value>[^'\"]*)['\"]",
        re.DOTALL,
    )
    supports: dict[str, Support] = {}
    for match in pattern.finditer(text):
        name = match.group("name")
        value = match.group("value")
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
    return sorted(
        path
        for path in subprocess_dir.glob("P*/matrix*.f")
        if not path.name.startswith("template_")
    )


def infer_matrix_amplitude_supports(
    matrix_file: Path,
    coupling_supports: dict[str, Support],
    zero: Power,
) -> list[Support]:
    wavefunctions: dict[str, Support] = {}
    amplitudes: list[Support] = []
    for statement in iter_fortran_call_statements(matrix_file):
        tokens = call_tokens(statement, coupling_supports)
        if not tokens or tokens[-1][0] not in ("W", "AMP"):
            continue
        output_type, output_name = tokens[-1]
        inputs: list[Support] = []
        for token_type, token_name in tokens[:-1]:
            if token_type == "W":
                inputs.append(wavefunctions.get(token_name, {zero}))
            elif token_type == "COUPLING":
                inputs.append(coupling_supports[token_name])
        support = product_supports(inputs, zero)
        if output_type == "W":
            wavefunctions[output_name] = support
        else:
            amplitudes.append(support)
    return amplitudes


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
        r"|AMP\(\s*(?P<amp>\d+)\s*\)"
        r"|(?P<name>\b[A-Za-z_][A-Za-z0-9_]*\b)"
    )
    tokens: list[tuple[str, str]] = []
    for token in token_pattern.finditer(args):
        if token.group("w"):
            tokens.append(("W", token.group("w")))
        elif token.group("amp"):
            tokens.append(("AMP", token.group("amp")))
        else:
            name = token.group("name")
            if name in coupling_supports:
                tokens.append(("COUPLING", name))
    return tokens


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


def format_inferred_terms(result: InferredTerms) -> str:
    coupling_text = ",".join(result.coupling_names)
    lines = [
        f"Process directory: {result.process_dir}",
        f"Scanned parameters: {coupling_text}",
        "Inferred amplitude support:",
    ]
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
    lines.append("Inferred cross-section [fit].terms:")
    lines.append("terms = [")
    for power in result.cross_section_terms:
        lines.append(f"  {format_power(power)},")
    lines.append("]")
    lines.append("WARNING: CHECK inferred fit terms before using them.")
    return "\n".join(lines)


def format_terms_block(terms: Iterable[Power]) -> str:
    lines = ["terms = ["]
    for power in terms:
        lines.append(f"  {format_power(power)},")
    lines.append("]")
    return "\n".join(lines)


def update_config_fit_terms(config_path: Path, terms: Iterable[Power]) -> Path:
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

    config_path.write_text(text, encoding="utf-8")
    return config_path


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
