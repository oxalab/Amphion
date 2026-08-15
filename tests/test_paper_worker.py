"""Fake-first test for PaperWorker: no network, no LLM spend.

Overrides `_fetch_url_content` to return canned bytes (httpx can't open file://
URLs, and we want the test network-free). A stub model_client returns a canned
summary. Proves the full 2-step slice (fetch -> summarize) end-to-end, including
input resolution.
"""

import sqlite3
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from src.models.artifact import ArtifactKind
from src.models.event import EventType
from src.models.research_task import Priority, ResearchTask, ResearchTaskStatus
from src.models.step import Step, StepKind
from src.registry.artifact_store import ArtifactStore
from src.registry.event_bus import EventBus
from src.registry.registry import TaskRegistry
from src.workers.paper_worker import PaperWorker


def _fresh_stores():
    conn = sqlite3.connect(":memory:")
    return conn, TaskRegistry(conn), EventBus(conn), ArtifactStore(conn, base_dir=tempfile.mkdtemp())


def _make_step(step_id="step-1", task_id="task-1", kind=StepKind.FETCH_PAPER,
               deps=None, params=None):
    return Step(
        id=step_id, task_id=task_id, kind=kind,
        dependencies=deps or [], input_artifacts=[], params=params or {},
    )


def _make_task(task_id="task-1", steps=None):
    now = datetime.now(UTC)
    return ResearchTask(
        id=task_id, objective="summarize a paper", status=ResearchTaskStatus.PENDING,
        priority=Priority.MEDIUM, owner="llm.reason", fingerprint_id=uuid4(),
        steps=steps or [_make_step()], artifacts=[], retry_count=0, retry_budget=3,
        created_at=now, updated_at=now,
    )


def _fake_model_client(canned_text: str):
    """OpenAI-compatible stub: client.chat.completions.create(...) -> canned response."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=canned_text))]
    )
    client = SimpleNamespace()
    client.chat = SimpleNamespace()
    client.chat.completions = SimpleNamespace()
    client.chat.completions.create = lambda **kw: response
    return client


def test_paper_worker_fetch_and_summarize():
    _, registry, bus, artifacts = _fresh_stores()
    client = _fake_model_client("Method: flash-attn. Dataset: wikitext. Metrics: 2x speedup.")
    canned_paper = b"This paper introduces a novel fused attention kernel."

    # httpx can't open file:// URLs, so override the fetch to stay network-free.
    class _TestablePaperWorker(PaperWorker):
        def _fetch_url_content(self, url):
            return canned_paper

    worker = _TestablePaperWorker(
        registry=registry, bus=bus, artifacts=artifacts,
        kinds=[StepKind.FETCH_PAPER, StepKind.EXTRACT_SUMMARY],
        worker_id="paper-1", model_client=client,
    )

    a = _make_step(step_id="fetch", kind=StepKind.FETCH_PAPER, params={"url": "http://fake/paper"})
    b = _make_step(step_id="extract", kind=StepKind.EXTRACT_SUMMARY, deps=["fetch"])
    registry.create_task(_make_task(task_id="t1", steps=[a, b]))

    worker._execute(a)
    worker._execute(b)

    # fetch -> a PAPER artifact 
    fetch_after = registry.get_step("fetch")
    print(f"DEBUG - fetch_after: {fetch_after}")
    if fetch_after:
        print(f"DEBUG - output_artifact: {fetch_after.output_artifact}")

    assert fetch_after is not None and fetch_after.output_artifact is not None

    fetched_artifact = artifacts.get(fetch_after.output_artifact)
    assert fetched_artifact is not None, f"Artifact ID {fetch_after.output_artifact}"
    assert fetched_artifact.kind == ArtifactKind.PAPER

    # extract -> a SUMMARY artifact containing the fake LLM output
    extract_after = registry.get_step("extract")
    assert extract_after is not None and extract_after.output_artifact is not None
    
    summary = artifacts.read(extract_after.output_artifact)

    assert summary is not None, "Summary data is missing from the artifact store."
    assert b"flash-attn" in summary

    # both bookend events fired (fetch completed + extract completed)
    assert len(bus.replay(EventType.STEP_COMPLETED)) == 2


_TESTS = [test_paper_worker_fetch_and_summarize]

if __name__ == "__main__":
    passed = 0
    for t in _TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
        except RuntimeError as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(_TESTS)} passed")
    sys.exit(0 if passed == len(_TESTS) else 1)
