"""Question -> Result. Every step appends to the trace that the ProofGraph is drawn from."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import confidence as conf
from .calc import run_calcs
from .conflicts import find_conflicts, premise_violations
from .index import Index, load_embedder
from .ledger import Ledger
from .reasoner import ReasonerOut, reason, resolve_variable
from .retrieve import build_pack, plan
from .schema import Answer, Calc, Claim, Element, Result, TraceStep
from .store import Store
from .verify import fill_placeholders, number_supported, verify_claims


@dataclass
class Mode:
    ledger: bool = True
    relook: bool = True
    verify: bool = True

    @classmethod
    def full(cls) -> "Mode":
        return cls()

    @classmethod
    def baseline(cls) -> "Mode":
        return cls(ledger=False, relook=False, verify=False)


class Pipeline:
    def __init__(self, store: Store, llm, settings, index: Optional[Index] = None):
        self.store, self.llm, self.settings = store, llm, settings
        self.ledger = Ledger(store)
        self.index = index
        self.embed_name = ""

    def build_index(self, embedder=None) -> None:
        if embedder is None:
            embedder, self.embed_name = load_embedder(self.settings)
        self.index = Index(embedder, str(self.settings.chroma_dir))
        self.index.build(self.store.all_elements())

    # -- main entry -------------------------------------------------------
    def ask(self, question: str, mode: Mode = Mode()) -> Result:
        res = Result(question=question)
        llm_log: list = []
        try:
            self._run(question, mode, res, llm_log)
        except Exception as e:  # soft failure: the UI always gets a Result
            res.errors.append(f"{type(e).__name__}: {e}")
            res.answer = Answer(refused=True, refusal_reason=f"Internal error: {e}")
        res.trace.append(TraceStep(step="llm_calls", data={"calls": llm_log}))
        return res

    def _step(self, res: Result, step: str, detail: str = "", **data) -> None:
        res.trace.append(TraceStep(step=step, detail=detail, data=data))

    def _refuse(self, res: Result, reason: str, retrieval: float = 0.0) -> None:
        res.answer = Answer(refused=True, refusal_reason=reason, confidence=0.0,
                            confidence_parts={"retrieval": retrieval})
        self._step(res, "refuse", reason)

    def _run(self, question: str, mode: Mode, res: Result, llm_log: list) -> None:
        if self.index is None:
            self.build_index()
        p = plan(self.llm, question, llm_log)
        self._step(res, "plan", f"{len(p.sub_questions)} sub-questions", sub_questions=p.sub_questions,
                   premises=[x.model_dump() for x in p.premises])

        hits = build_pack(self.index, question, p.sub_questions, self.settings.top_k, self.settings.min_retrieval_score)
        pack = self.store.get_many([h.id for h in hits])
        res.evidence = pack
        top = max((h.dense for h in hits), default=0.0)
        self._step(res, "retrieve", f"{len(pack)} elements, top score {top:.2f}",
                   hits=[{"id": h.id, "rrf": round(h.rrf, 4), "dense": round(h.dense, 3)} for h in hits])

        if reason_ := conf.retrieval_gate(top, self.settings.min_retrieval_score):
            return self._refuse(res, reason_, top)

        if mode.ledger:
            violations = premise_violations(self.ledger, p.premises)
            self._step(res, "premise_guard", "violated" if violations else "ok", violations=violations)
            if violations:
                return self._refuse(res, "False premise. " + " ".join(violations), top)

        facts = self.ledger.for_sources([e.id for e in pack]) if mode.ledger else []
        out = reason(self.llm, question, p.sub_questions, pack, facts, relook=mode.relook, trace=llm_log)
        relooked = [e.id for e in pack if e.kind in ("figure", "table")][:3] if mode.relook else []
        self._step(res, "reason", f"sufficient={out.sufficient}", relook=relooked, claims=len(out.claims),
                   facts=len(facts))
        if reason_ := conf.evidence_gate(out.sufficient, out.missing):
            return self._refuse(res, reason_, top)

        by_id = {e.id: e for e in pack}
        calcs = self._calculate(out, by_id, res)
        claims = [Claim(text=fill_placeholders(c.text, calcs), cites=c.cites, quote=c.quote) for c in out.claims]
        claims = verify_claims(self.llm, claims, by_id, calcs, llm_log) if mode.verify else claims
        kept = [c for c in claims if c.verdict != "unsupported"]
        self._step(res, "verify", f"{len(kept)}/{len(claims)} claims kept",
                   verdicts=[{"text": c.text, "verdict": c.verdict, "cites": c.cites, "reasons": c.reasons} for c in claims])
        if reason_ := conf.verification_gate(kept):
            return self._refuse(res, reason_, top)

        conflicts, agreement = find_conflicts(self.ledger, [x for c in kept for x in c.cites]) if mode.ledger else ([], 0.5)
        about = (question + " " + " ".join(c.text for c in kept)).lower()
        conflicts = [c for c in conflicts if all(w in about for w in c.metric.split())]  # only metrics this answer is about
        self._step(res, "triangulate", f"{len(conflicts)} conflicts, agreement {agreement:.2f}",
                   conflicts=[c.metric + " " + c.period for c in conflicts])
        res.answer = self._answer(out, kept, claims, calcs, conflicts, agreement, top)
        self._step(res, "confidence", f"{res.answer.confidence:.2f}", parts=res.answer.confidence_parts)

    def _calculate(self, out: ReasonerOut, by_id: dict[str, Element], res: Result) -> list[Calc]:
        values, sources, bad = {}, {}, []
        for v in out.variables:
            el = by_id.get(v.cite)
            val = resolve_variable(v, el)
            if el is None or val is None or not number_supported(val, [el], []):
                bad.append(v.name)
                continue
            values[v.name], sources[v.name] = val, [v.cite]
        calcs = run_calcs([Calc(name=c.name, expr=c.expr) for c in out.calcs], values, sources)
        self._step(res, "calculate", f"{len(calcs)} calcs, {len(bad)} rejected inputs", rejected_inputs=bad,
                   calcs=[c.model_dump() for c in calcs])
        return calcs

    def _answer(self, out: ReasonerOut, kept: list[Claim], claims: list[Claim], calcs: list[Calc],
                conflicts, agreement: float, top: float) -> Answer:
        cited = self.store.get_many(sorted({x for c in kept for x in c.cites}))
        quality = sum(self.store.page_quality(e.doc_id, e.page) for e in cited) / max(len(cited), 1)
        estimated = any(c.estimated for c in kept)
        score, parts = conf.confidence(top, conf.verified_ratio(claims), agreement, quality, estimated, bool(conflicts))
        text = fill_placeholders(out.answer, calcs)
        if len(kept) < len(claims):
            text = " ".join(c.text for c in kept)  # never show prose built on a removed claim
        return Answer(
            text=text, claims=kept, calcs=calcs, conflicts=conflicts, confidence=score, confidence_parts=parts,
            modalities=sorted({e.modality for e in cited}), estimated=estimated,
        )
