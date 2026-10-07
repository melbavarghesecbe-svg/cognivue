"""Safe arithmetic over named variables. ast whitelist, never eval()."""
from __future__ import annotations

import ast
import math
import operator

from .schema import Calc


class CalcError(Exception):
    pass


BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _pct_change(old: float, new: float) -> float:
    return (new - old) / old * 100


def _cagr(start: float, end: float, years: float) -> float:
    return ((end / start) ** (1 / years) - 1) * 100


FUNCS = {
    "round": round, "abs": abs, "min": min, "max": max,
    "pct_change": _pct_change, "cagr": _cagr, "sqrt": math.sqrt,
}


def safe_eval(expr: str, variables: dict[str, float]) -> float:
    if len(expr) > 300:
        raise CalcError("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"syntax error: {e.msg}") from e
    return float(_eval(tree.body, variables))


def _eval(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise CalcError(f"unknown variable '{node.id}'")
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
        left, right = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise CalcError("exponent too large")
        try:
            return BIN_OPS[type(node.op)](left, right)
        except ZeroDivisionError as e:
            raise CalcError("division by zero") from e
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        return UNARY_OPS[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS and not node.keywords:
        args = [_eval(a, env) for a in node.args]
        try:
            return FUNCS[node.func.id](*args)
        except (TypeError, ValueError, ZeroDivisionError) as e:
            raise CalcError(f"{node.func.id}: {e}") from e
    raise CalcError(f"disallowed expression: {type(node).__name__}")


def run_calcs(calcs: list[Calc], variables: dict[str, float], var_sources: dict[str, list[str]]) -> list[Calc]:
    """Evaluate in order; each result becomes a variable for later calcs. Errors are recorded, not raised."""
    env = dict(variables)
    sources = {k: list(v) for k, v in var_sources.items()}
    out = []
    for c in calcs:
        names = sorted({n.id for n in ast.walk(_safe_parse(c.expr)) if isinstance(n, ast.Name)} - FUNCS.keys())
        done = c.model_copy(update={
            "inputs": {n: env[n] for n in names if n in env},
            "sources": sorted({s for n in names for s in sources.get(n, [])}),
        })
        try:
            done.result = safe_eval(c.expr, env)
            env[c.name] = done.result
            sources[c.name] = done.sources
        except CalcError as e:
            done.error = str(e)
        out.append(done)
    return out


def _safe_parse(expr: str) -> ast.AST:
    try:
        return ast.parse(expr, mode="eval")
    except SyntaxError:
        return ast.parse("0", mode="eval")
