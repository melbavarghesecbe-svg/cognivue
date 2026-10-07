"""Ingest the generated demo docs fully offline (VLM calls soft-fail via CACHE_ONLY)."""
import pytest

from cognivue.ingest import ingest_pdf
from cognivue.ledger import Ledger
from cognivue.store import Store
from scripts.make_demo_docs import main as make_docs


@pytest.fixture(scope="module")
def demo_paths(tmp_path_factory):
    return make_docs(tmp_path_factory.mktemp("docs"))


@pytest.fixture
def ingested(demo_paths, settings, offline_llm):
    store = Store(settings.db_path)
    errors = []
    for p in demo_paths:
        _, errs = ingest_pdf(p, store, offline_llm, settings)
        errors += errs
    return store, errors


def test_ingest_all_docs_without_crashing(ingested):
    store, errors = ingested
    assert errors == []
    assert [d["id"] for d in store.docs()] == ["D1", "D2", "D3", "D4"]


def test_annual_report_elements(ingested):
    store, _ = ingested
    els = store.all_elements("D1")
    kinds = {e.kind for e in els}
    assert {"heading", "text", "table", "figure"} <= kinds
    table = next(e for e in els if e.kind == "table")
    assert table.id == "D1-p2-T1" and "1,450" in table.text
    assert table.meta["caption"].startswith("Table 1")
    fig = next(e for e in els if e.kind == "figure")
    assert fig.id.startswith("D1-p3-F") and fig.image_path
    assert "data not extracted" not in fig.text.lower()


def test_scan_is_triaged_and_ocrd(ingested):
    store, _ = ingested
    page = store.pages("D3")[0]
    assert page["kind"] == "scan"
    text = " ".join(e.text for e in store.all_elements("D3"))
    assert "96 hours" in text or "coolant" in text.lower()


def test_table_facts_in_ledger(ingested):
    store, _ = ingested
    led = Ledger(store)
    rev = led.find("revenue", "FY24")
    assert any(f.value == 1.45e10 and f.source_id == "D1-p2-T1" for f in rev)
    assert led.find("pune capacity")[0].value == 120000


def test_reingest_replaces(demo_paths, ingested, settings, offline_llm):
    store, _ = ingested
    n = len(store.all_elements())
    doc_id, _ = ingest_pdf(demo_paths[0], store, offline_llm, settings)
    assert doc_id == "D1" and len(store.all_elements()) == n
