"""ProofGraph: Graphviz DOT built only from the recorded trace of a Result."""
from __future__ import annotations

from .schema import Result

COLORS = {"supported": "#2e7d32", "partial": "#f9a825", "unsupported": "#c62828", "unchecked": "#757575"}


def _q(s: str, n: int = 48) -> str:
    s = s.replace('"', "'").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def build_dot(res: Result) -> str:
    steps = {t.step: t for t in res.trace}
    lines = ['digraph G {', 'rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f5f5f5", fontsize=10];']
    lines.append(f'Q [label="Q: {_q(res.question)}", fillcolor="#e3f2fd"];')
    if "plan" in steps:
        for i, sq in enumerate(steps["plan"].data.get("sub_questions", [])):
            lines.append(f'SQ{i} [label="{_q(sq, 36)}"]; Q -> SQ{i};')
    if "premise_guard" in steps:
        v = steps["premise_guard"].data.get("violations", [])
        lines.append(f'PG [label="Premise Guard: {"VIOLATED" if v else "ok"}", fillcolor="{"#ffcdd2" if v else "#c8e6c9"}"]; Q -> PG;')
    hits = [h["id"] for h in steps["retrieve"].data.get("hits", [])] if "retrieve" in steps else []
    for h in hits:
        lines.append(f'"{h}" [shape=note, fillcolor="#fff8e1"]; Q -> "{h}" [style=dotted];')
    for h in steps["reason"].data.get("relook", []) if "reason" in steps else []:
        lines.append(f'"{h}" -> R [label="visual re-look", color="#6a1b9a", fontcolor="#6a1b9a"];')
    if "reason" in steps:
        lines.append(f'R [label="Reasoner ({_q(steps["reason"].detail, 30)})", fillcolor="#ede7f6"];')
    for i, c in enumerate(steps["calculate"].data.get("calcs", []) if "calculate" in steps else []):
        res_txt = "error" if c.get("error") else f"{c.get('result'):.4g}"
        lines.append(f'C{i} [label="{_q(c["name"] + " = " + c["expr"], 40)}\\n= {res_txt}", shape=hexagon, fillcolor="#e0f2f1"];')
        for s in c.get("sources", []):
            lines.append(f'"{s}" -> C{i};')
    for i, v in enumerate(steps["verify"].data.get("verdicts", []) if "verify" in steps else []):
        col = COLORS.get(v["verdict"], "#757575")
        lines.append(f'CL{i} [label="{_q(v["text"], 40)}\\n[{v["verdict"]}]", color="{col}", penwidth=2];')
        for s in v.get("cites", []):
            lines.append(f'"{s}" -> CL{i} [color="{col}"];')
    for i, name in enumerate(steps["triangulate"].data.get("conflicts", []) if "triangulate" in steps else []):
        lines.append(f'X{i} [label="CONFLICT: {_q(name)}", fillcolor="#ffcdd2"];')
    end = "Refused" if res.answer.refused else f"Answer (conf {res.answer.confidence:.2f})"
    lines.append(f'A [label="{end}", fillcolor="{"#ffcdd2" if res.answer.refused else "#c8e6c9"}"];')
    lines.append("}")
    return "\n".join(lines)
