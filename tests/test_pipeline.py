"""Retrieval -> reasoning -> verification, offline with a scripted FakeProvider and hash embeddings."""
import pytest

from cognivue.confidence import confidence
from cognivue.conflicts import find_conflicts
from cognivue.index import hash_embedder
from cognivue.ingest import ingest_pdf
from cognivue.ledger import Ledger, norm_metric
from cognivue.llm import LLM
from cognivue.pipeline import Mode, Pipeline
from cognivue.schema import Calc, Claim, Element, Fact
from cognivue.store import Store
from cognivue.verify import code_checks, fill_placeholders
from scripts.make_demo_docs import main as make_docs
from tests.conftest import FakeProvider

PLAN = {"sub_questions": ["Revenue FY23", "Revenue FY24"], "premises": []}
REASON = {
    "sufficient": True,
    "answer": "Revenue grew {growth} from FY23 to FY24.",
    "claims": [
        {"text": "Revenue grew {growth} from Rs 1,210 crore in FY23 to Rs 1,450 crore in FY24.",
         "cites": ["D1-p2-T1"], "quote": "1,450"},
        {"text": "Revenue was Rs 1,500 crore in FY24.", "cites": ["D1-p2-T1"], "quote": ""},
        {"text": "The CEO is happy.", "cites": ["D9-p1-X1"], "quote": ""},
    ],
    "variables": [{"name": "a", "cite": "D1-p2-T1", "value_text": "1,210"},
                  {"name": "b", "cite": "D1-p2-T1", "value_text": "1,450"}],
    "calcs": [{"name": "growth", "expr": "pct_change(a, b)"}],
}
ENTAIL = {"results": [{"idx": 0, "label": "entailed"}]}


@pytest.fixture(scope="module")
def demo_paths(tmp_path_factory):
    return make_docs(tmp_path_factory.mktemp("docs"))


def make_pipeline(settings, demo_paths, replies):
    settings.min_retrieval_score = 0.05
    store = Store(settings.db_path)
    offline = LLM([], settings.cache_dir, cache_only=True)
    for p in demo_paths:
        ingest_pdf(p, store, offline, settings)
    llm = LLM([FakeProvider(replies)], settings.cache_dir / "q", backoff_s=0)
    pipe = Pipeline(store, llm, settings)
    pipe.build_index(hash_embedder)
    return pipe


REPLIES = {"Split the user question": PLAN, "You answer questions ONLY": REASON, "judge whether": ENTAIL}


def test_full_pipeline_verifies_and_computes(settings, demo_paths):
    res = make_pipeline(settings, demo_paths, REPLIES).ask("How much did revenue grow from FY23 to FY24?")
    a = res.answer
    assert not a.refused, res.errors
    assert len(a.claims) == 1 and a.claims[0].verdict == "supported"
    assert "19.8%" in a.claims[0].text and "19.8%" in a.text
    assert a.calcs[0].sources == ["D1-p2-T1"]
    assert 0 < a.confidence <= 1
    steps = [t.step for t in res.trace]
    assert steps[:3] == ["plan", "retrieve", "premise_guard"] and "verify" in steps


def test_false_premise_refused(settings, demo_paths):
    replies = dict(REPLIES)
    replies["Split the user question"] = {"sub_questions": ["revenue FY24"],
                                          "premises": [{"metric": "revenue", "period": "FY24", "value_text": "Rs 900 crore"}]}
    res = make_pipeline(settings, demo_paths, replies).ask("Why did revenue fall to Rs 900 crore in FY24?")
    assert res.answer.refused and "False premise" in res.answer.refusal_reason


def test_insufficient_evidence_refused(settings, demo_paths):
    replies = dict(REPLIES)
    replies["You answer questions ONLY"] = {"sufficient": False, "missing": "no salary data"}
    res = make_pipeline(settings, demo_paths, replies).ask("What is the CEO's salary?")
    assert res.answer.refused and "salary" in res.answer.refusal_reason


def test_baseline_skips_verifier(settings, demo_paths):
    res = make_pipeline(settings, demo_paths, REPLIES).ask("How much did revenue grow?", Mode.baseline())
    assert len(res.answer.claims) == 3  # nothing removed


def test_llm_outage_is_soft(settings, demo_paths):
    pipe = make_pipeline(settings, demo_paths, {})
    pipe.llm = LLM([], settings.cache_dir / "none", cache_only=True)
    res = pipe.ask("revenue?")
    assert res.answer.refused and res.errors


# -- unit checks -------------------------------------------------------------

def _el(id_, text, kind="text", **meta):
    return Element(id=id_, doc_id=id_.split("-")[0], page=1, kind=kind, text=text, meta=meta)


def test_code_checks_number_and_ids():
    pack = {"D1-p1-X1": _el("D1-p1-X1", "Repair cost was Rs 38 lakh.")}
    ok = code_checks(Claim(text="Repair cost was Rs 38 lakh.", cites=["D1-p1-X1"], quote="Rs 38 lakh"), pack, [])
    assert ok.verdict == "supported"
    bad = code_checks(Claim(text="Repair cost was Rs 40 lakh.", cites=["D1-p1-X1"]), pack, [])
    assert bad.verdict == "unsupported"
    ghost = code_checks(Claim(text="x", cites=["D7-p1-X1"]), pack, [])
    assert ghost.verdict == "unsupported" and ghost.cites == []


def test_fill_placeholders():
    calcs = [Calc(name="g", expr="pct_change(a,b)", result=19.834), Calc(name="d", expr="b-a", result=2.4e9)]
    assert fill_placeholders("{g} / {d} / {zz}", calcs) == "19.8% / Rs 240.00 crore / {zz}"


def test_confidence_formula():
    score, _ = confidence(1, 1, 1, 1, chart_estimated=False, has_conflict=False)
    assert score == 1.0
    assert confidence(1, 1, 1, 1, chart_estimated=True, has_conflict=False)[0] == 0.9
    assert confidence(1, 1, 1, 1, chart_estimated=False, has_conflict=True)[0] == 0.6


def test_conflict_detected(settings):
    store = Store(settings.db_path)
    led = Ledger(store)
    mk = lambda v, src, mod: Fact(metric="Revenue", metric_norm=norm_metric("Revenue"), period="FY24", value=v,
                                   unit="INR", raw=str(v), source_id=src, doc_id=src[:2], modality=mod)
    led.add([mk(1.45e10, "D1-p2-T1", "table"), mk(1.48e10, "D4-p1-X3", "text")])
    conflicts, agreement = find_conflicts(led, ["D1-p2-T1"])
    assert len(conflicts) == 1 and len(conflicts[0].facts) == 2 and agreement == 0
