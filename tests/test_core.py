import json

import pytest
from pydantic import BaseModel

from cognivue.calc import CalcError, run_calcs, safe_eval
from cognivue.ledger import facts_from_table, norm_metric, norm_period
from cognivue.llm import LLM, CacheMiss, LLMError, extract_json
from cognivue.numparse import find_numbers, parse_number
from cognivue.schema import Calc, Element, Result, make_element_id
from tests.conftest import FakeProvider


class Out(BaseModel):
    x: int


# -- schema -------------------------------------------------------------------

def test_element_id_format():
    assert make_element_id("D1", 7, "figure", 3) == "D1-p7-F3"


def test_result_roundtrip():
    r = Result(question="q")
    assert Result.model_validate_json(r.model_dump_json()) == r


# -- llm ----------------------------------------------------------------------

def test_llm_caches(tmp_path):
    p = FakeProvider({"hello": {"x": 1}})
    llm = LLM([p], tmp_path, backoff_s=0)
    assert llm.call_json("t", "hello", Out).x == 1
    assert llm.call_json("t", "hello", Out).x == 1
    assert p.calls == 1


def test_llm_cache_only_miss(tmp_path):
    llm = LLM([FakeProvider()], tmp_path, cache_only=True)
    with pytest.raises(CacheMiss):
        llm.call_json("t", "hello", Out)


def test_llm_retry_then_fallback(tmp_path):
    bad = FakeProvider(fail_times=99, name="primary")
    good = FakeProvider({"hello": {"x": 2}}, name="fallback")
    trace = []
    llm = LLM([bad, good], tmp_path, retries=2, backoff_s=0)
    assert llm.call_json("t", "hello", Out, trace=trace).x == 2
    assert bad.calls == 2 and trace[-1]["model"] == "fallback"


def test_llm_invalid_json_fails_softly(tmp_path):
    llm = LLM([FakeProvider({"hello": "not json"})], tmp_path, retries=2, backoff_s=0)
    with pytest.raises(LLMError):
        llm.call_json("t", "hello", Out)


def test_extract_json_strips_fences():
    assert json.loads(extract_json('sure ```json\n{"x": 1}\n```')) == {"x": 1}


# -- numbers ------------------------------------------------------------------

@pytest.mark.parametrize("text,value", [
    ("Rs 1,23,45,678", 12345678), ("12.5 crore", 125000000), ("Rs 38 lakh", 3800000),
    ("(1,234)", -1234), ("88%", 88), ("₹1,450 crore", 14500000000),
])
def test_parse_number(text, value):
    assert parse_number(text) == pytest.approx(value)


def test_find_numbers_multiple():
    assert [v for _, v in find_numbers("from 980 to 1,210")] == [980, 1210]


# -- calculator ---------------------------------------------------------------

def test_safe_eval_basic():
    assert safe_eval("(b - a) / a * 100", {"a": 1210, "b": 1450}) == pytest.approx(19.8347, rel=1e-4)
    assert safe_eval("pct_change(a, b)", {"a": 100, "b": 120}) == pytest.approx(20)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('x')", "a.__class__", "open('f')", "[1,2]", "lambda: 1", "2 ** 1000", "a if a else b",
])
def test_safe_eval_rejects(expr):
    with pytest.raises(CalcError):
        safe_eval(expr, {"a": 1, "b": 2})


def test_run_calcs_chains_and_tracks_sources():
    calcs = [Calc(name="growth", expr="b - a"), Calc(name="pct", expr="growth / a * 100"), Calc(name="bad", expr="a / 0")]
    out = run_calcs(calcs, {"a": 100, "b": 150}, {"a": ["D1-p2-T1"], "b": ["D1-p2-T1", "D4-p1-X2"]})
    assert out[1].result == pytest.approx(50)
    assert out[1].sources == ["D1-p2-T1", "D4-p1-X2"]
    assert out[2].error == "division by zero"


# -- ledger -------------------------------------------------------------------

def test_norm_period_and_metric():
    assert norm_period("FY 2023-24") == "FY24"
    assert norm_period("FY24") == "FY24"
    assert norm_period("Q3 FY24") == "Q3 FY24"
    assert norm_metric("Revenue from operations (Rs crore)") == "revenue"


def test_facts_from_table_periods_in_header():
    el = Element(id="D1-p2-T1", doc_id="D1", page=2, kind="table", meta={"caption": "Table 1: Key financials (Rs crore)"})
    rows = [["Metric", "FY23", "FY24"], ["Revenue", "1,210", "1,450"]]
    facts = facts_from_table(el, rows)
    assert {(f.period, f.value) for f in facts} == {("FY23", 1.21e10), ("FY24", 1.45e10)}
    assert all(f.unit == "INR" for f in facts)


def test_facts_from_table_entity_rows_indian_grouping():
    el = Element(id="D2-p1-T1", doc_id="D2", page=1, kind="table", meta={"caption": "Table 1: by plant, FY24"})
    rows = [["Plant", "Capacity (units)", "Utilisation (%)"], ["Pune", "1,20,000", "80"]]
    facts = {f.metric_norm: f for f in facts_from_table(el, rows)}
    assert facts["pune capacity"].value == 120000 and facts["pune capacity"].period == "FY24"
    assert facts["pune utilisation"].unit == "%"
