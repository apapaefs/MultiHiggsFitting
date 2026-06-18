from __future__ import annotations

import re


def restricted_mheft_squared_order_cap(config) -> int | None:
    explicit = explicit_mheft_squared_order_cap(config.generate)
    if explicit is not None:
        return explicit
    if config.model != "heft_loop_sm_restricted5":
        return None
    final_state = primary_final_state(config.generate)
    higgs_count = sum(1 for token in process_tokens(final_state) if token.lower() == "h")
    if higgs_count == 0:
        return None
    return 2 * higgs_count


def generate_with_restricted_mheft_cap(config) -> str:
    if config.model != "heft_loop_sm_restricted5":
        return config.generate
    cap = restricted_mheft_squared_order_cap(config)
    if cap is None:
        return config.generate
    return normalize_restricted_mheft_generate(config.generate, cap)


def explicit_mheft_squared_order_cap(generate: str) -> int | None:
    match = re.search(r"\bMHEFT\s*\^\s*2\s*<=\s*(\d+)", generate, re.IGNORECASE)
    return None if match is None else int(match.group(1))


def normalize_restricted_mheft_generate(generate: str, cap: int) -> str:
    without_cap = remove_mheft_squared_order_cap(generate)
    with_noborn = ensure_mheft_noborn(without_cap)
    return insert_mheft_squared_order_cap(with_noborn, cap)


def remove_mheft_squared_order_cap(generate: str) -> str:
    cleaned = re.sub(r"\s*\bMHEFT\s*\^\s*2\s*<=\s*\d+\s*", " ", generate, flags=re.IGNORECASE)
    return normalize_spaces(cleaned)


def ensure_mheft_noborn(generate: str) -> str:
    match = re.search(r"\[(?P<body>[^\]]*)\]", generate)
    if match is None:
        return generate
    body = match.group("body")
    noborn = re.search(r"\bnoborn\s*=\s*(?P<orders>[^,\]]+)", body, re.IGNORECASE)
    if noborn is None:
        return generate
    orders = noborn.group("orders").split()
    if not any(order.upper() == "MHEFT" for order in orders):
        orders.append("MHEFT")
    replacement = body[:noborn.start("orders")] + " ".join(orders) + body[noborn.end("orders"):]
    return generate[:match.start()] + "[" + replacement + "]" + generate[match.end():]


def insert_mheft_squared_order_cap(generate: str, cap: int) -> str:
    order = f"MHEFT^2<={cap}"
    bracket = generate.find("[")
    if bracket == -1:
        return f"{generate.rstrip()} {order}"
    bracket_end = generate.find("]", bracket)
    if bracket_end == -1:
        return f"{generate.rstrip()} {order}"
    before = generate[: bracket_end + 1].rstrip()
    after = generate[bracket_end + 1 :].strip()
    return f"{before} {order}" + (f" {after}" if after else "")


def primary_final_state(generate: str) -> str:
    without_brackets = re.sub(r"\[[^\]]*\]", " ", generate)
    if ">" not in without_brackets:
        return without_brackets
    final_state = without_brackets.split(">", 1)[1]
    return final_state.split(",", 1)[0]


def process_tokens(process_fragment: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_~+-]*", process_fragment)


def normalize_spaces(text: str) -> str:
    return " ".join(text.split())
