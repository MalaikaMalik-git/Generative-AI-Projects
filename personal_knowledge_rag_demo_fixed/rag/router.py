from __future__ import annotations

import re


DATE_PATTERN = r"\d{4}-\d{2}-\d{2}"


def detect_tool(query: str) -> str:
    q = query.lower().strip()

    # -------- DATE ROUTES FIRST --------
    if "today" in q and "date" in q:
        return "today"

    if re.search(DATE_PATTERN, q) and (
        "day after" in q
        or "days after" in q
        or "day before" in q
        or "days before" in q
        or "week after" in q
        or "weeks after" in q
        or "week before" in q
        or "weeks before" in q
    ):
        return "date"

    # -------- CALCULATOR ROUTE --------
    # Only match real arithmetic expressions, not ISO dates.
    normalized = q.replace("×", "*").replace("÷", "/")
    if re.search(r"\d+\s*[\+\*\/]\s*\d+", normalized):
        return "calculator"

    # subtraction-only expressions like "10 - 3" but avoid matching YYYY-MM-DD
    if re.search(r"\d+\s-\s\d+", normalized):
        return "calculator"

    return "rag"


def extract_math_expression(query: str) -> str | None:
    q = query.lower().strip()
    q = q.replace("×", "*").replace("÷", "/")

    # Remove ISO dates completely so they are never treated as math.
    q = re.sub(DATE_PATTERN, " ", q)

    # Keep only characters that can appear in arithmetic expressions.
    candidates = re.findall(r"[0-9\.\+\-\*\/\(\)\s]+", q)

    cleaned_candidates: list[str] = []
    for candidate in candidates:
        expr = candidate.strip()

        if not expr:
            continue

        # Must contain at least one operator and at least two numbers.
        if not re.search(r"[\+\-\*\/]", expr):
            continue

        if len(re.findall(r"\d+(?:\.\d+)?", expr)) < 2:
            continue

        # Avoid expressions that start/end with an operator.
        if re.match(r"^[\+\*\/]", expr) or re.search(r"[\+\-\*\/]$", expr):
            continue

        cleaned_candidates.append(expr)

    if not cleaned_candidates:
        return None

    return max(cleaned_candidates, key=len)


def extract_date_info(query: str) -> dict | None:
    q = query.lower().strip()

    if "today" in q and "date" in q:
        return {"mode": "today"}

    date_match = re.search(DATE_PATTERN, q)
    if not date_match:
        return None

    number_match = re.search(r"\b(\d+)\b", q)
    if not number_match:
        return None

    amount = int(number_match.group(1))

    if "week" in q:
        amount *= 7

    if "before" in q:
        amount = -amount

    return {
        "mode": "offset",
        "base_date": date_match.group(0),
        "offset_days": amount,
    }
