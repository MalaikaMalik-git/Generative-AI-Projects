from __future__ import annotations

import ast
import operator as op
from datetime import datetime, timedelta


_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.Num):  # compatibility
        return node.n

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        operand = _safe_eval(node.operand)
        return _ALLOWED_OPERATORS[type(node.op)](operand)

    raise ValueError("Unsupported expression")


def calculator_tool(expression: str) -> dict:
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval(parsed.body)
        return {"result": result}
    except Exception as exc:
        return {"error": str(exc)}


def date_tool(base_date: str, offset_days: int) -> dict:
    try:
        dt = datetime.strptime(base_date, "%Y-%m-%d")
        new_date = dt + timedelta(days=offset_days)
        return {"result_date": new_date.strftime("%Y-%m-%d")}
    except Exception as exc:
        return {"error": str(exc)}


def today_tool() -> dict:
    return {"today": datetime.now().strftime("%Y-%m-%d")}
