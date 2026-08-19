"""Smoke test for the three Phase 2 stores + the Event model.

Runs two ways:
  - directly:   `python tests/test_stores.py`
  - via pytest: `pytest tests/test_stores.py`  (after `uv add --dev pytest`)

Each test gets a fresh in-memory DB + temp artifact dir, so they're isolated.
"""

import sqlite3
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

# Bootstrap project root onto sys.path so `src.*` imports resolve when run
# directly. Remove this once the package is properly installed/configured.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.artifact import ArtifactConfidence, ArtifactKind
from src.models.event import EventType
from src.models.research_task import Priority, ResearchTask, ResearchTaskStatus
from src.models.step import Step, StepKind, StepStatus
from src.registry.artifact_store import ArtifactStore
from src.registry.event_bus import EventBus
from src.registry.registry import TaskRegistry

# --- fixtures / factories ---

def _fresh_stores() -> tuple[sqlite3.Connection, TaskRegistry, EventBus, ArtifactStore]:
    """One shared in-memory connection across all three stores + a temp artifact dir."""
    conn = sqlite3.connect(":memory:")
    return (
        conn,
        TaskRegistry(conn),
        EventBus(conn),
        ArtifactStore(conn, base_dir=tempfile.mkdtemp()),
    )


def _make_step(
    step_id: str = "step-1",
    task_id: str = "task-1",
    kind: StepKind = StepKind.FETCH_PAPER,
    deps: list[str] | None = None,
) -> Step:
    return Step(
        id=step_id,
        task_id=task_id,
        kind=kind,
        dependencies=deps or [],
        input_artifacts=[],
    )


def _make_task(task_id: str = "task-1", steps: list[Step] | None = None) -> ResearchTask:
    now = datetime.now(UTC)
    return ResearchTask(
        id=task_id,
        objective="reproduce FlashAttention-3",
        status=ResearchTaskStatus.PENDING,
        priority=Priority.MEDIUM,
        owner="gpu.benchmark",
        fingerprint_id=uuid4(),
        steps=steps or [_make_step()],
        artifacts=[],
        retry_count=0,
        retry_budget=3,
        created_at=now,
        updated_at=now,
    )


# --- tests ---

def test_task_round_trip():
    """create_task → get_task rehydrates every field; claim_step is exclusive."""
    _, registry, _, _ = _fresh_stores()
    task = _make_task(steps=[_make_step(step_id="s1")])
    registry.create_task(task)

    got = registry.get_task(task.id)
    assert got is not None
    assert got.status == ResearchTaskStatus.PENDING
    assert got.priority == Priority.MEDIUM
    assert got.owner == "gpu.benchmark"
    assert len(got.steps) == 1
    assert got.steps[0].kind == StepKind.FETCH_PAPER
    assert got.steps[0].requires_reasoning is False  # FetchPaper is an executor
    # enums + UUID + datetime survive the round trip
    assert got.fingerprint_id == task.fingerprint_id
    assert got.created_at == task.created_at

    # claim is exclusive: first wins, second loses
    assert registry.claim_step("s1", "worker-1") is True
    assert registry.claim_step("s1", "worker-2") is False

    # cannot complete a step you don't own / that isn't Running
    assert registry.complete_step("s1", "artifact-x") is True
    # already completed → second complete fails (no longer Running)
    assert registry.complete_step("s1", "artifact-y") is False
    print("  task round-trip + exclusive claim + status guard OK")


def test_dag_guard():
    """get_pending_steps only returns steps whose dependencies are Completed."""
    _, registry, _, _ = _fresh_stores()
    a = _make_step(step_id="A", kind=StepKind.BUILD_IMAGE)
    b = _make_step(step_id="B", kind=StepKind.RUN_BENCHMARK, deps=["A"])
    registry.create_task(_make_task(steps=[a, b]))

    # Initially only A is claimable — B is blocked by unfinished A.
    pending = registry.get_pending_steps()
    pending_ids = {s.id for s in pending}
    assert pending_ids == {"A"}, f"expected only A, got {pending_ids}"

    # Claim + complete A.
    assert registry.claim_step("A", "worker-1") is True
    assert registry.complete_step("A", "img-hash") is True

    # Now B is claimable.
    pending = registry.get_pending_steps()
    pending_ids = {s.id for s in pending}
    assert pending_ids == {"B"}, f"expected only B, got {pending_ids}"
    print("  DAG guard: dependent step hidden until dependency completes OK")


def test_artifact_store():
    """put is content-addressed + idempotent; read returns the bytes."""
    _, _, _, artifacts = _fresh_stores()

    art1 = artifacts.put(
        b"hello world",
        kind=ArtifactKind.SUMMARY,
        task_id="task-1",
        produced_by="step-1",
        content_type="text/plain",
    )
    # same content → same id, no duplicate
    art2 = artifacts.put(
        b"hello world",
        kind=ArtifactKind.SUMMARY,
        task_id="task-1",
        produced_by="step-1",
        content_type="text/plain",
    )
    assert art1.id == art2.id
    assert artifacts.exists(art1.id)

    # different content → different id
    art3 = artifacts.put(
        b"goodbye",
        kind=ArtifactKind.SUMMARY,
        task_id="task-1",
        produced_by="step-1",
        content_type="text/plain",
    )
    assert art3.id != art1.id

    # read round-trips the bytes
    assert artifacts.read(art1.id) == b"hello world"
    assert artifacts.read("nonexistent") is None

    # id is a 64-char hex sha256
    assert len(art1.id) == 64 and all(c in "0123456789abcdef" for c in art1.id)
    print("  artifact store: content-addressing + idempotency + read OK")


def test_event_bus():
    """emit is idempotent by event_id; replay reads the dedup'd log."""
    _, _, bus, _ = _fresh_stores()
    calls: list[str] = []

    bus.subscribe(EventType.STEP_COMPLETED, lambda e: calls.append(e.id))

    _, was_new_1 = bus.emit(EventType.STEP_COMPLETED, producer_id="s1", event_id="ev-1")
    _, was_new_2 = bus.emit(EventType.STEP_COMPLETED, producer_id="s1", event_id="ev-1")  # dup
    _, was_new_3 = bus.emit(EventType.STEP_COMPLETED, producer_id="s1", event_id="ev-2")  # new

    assert was_new_1 is True
    assert was_new_2 is False, "duplicate event_id should be absorbed"
    assert was_new_3 is True
    assert len(calls) == 2, f"subscribers should fire twice, got {len(calls)}"

    logged = bus.replay(EventType.STEP_COMPLETED)
    assert len(logged) == 2, "log should contain only the 2 dedup'd events"
    print("  event bus: idempotent dispatch + replay OK")


def test_cross_store():
    """End-to-end: claim → produce artifact → complete step → emit event."""
    _, registry, bus, artifacts = _fresh_stores()
    task = _make_task(steps=[_make_step(step_id="s1", kind=StepKind.RUN_BENCHMARK)])
    registry.create_task(task)

    # a worker claims the step
    assert registry.claim_step("s1", "worker-1") is True

    # produces a benchmark result artifact
    art = artifacts.put(
        b"latency=42ms,throughput=1200tps",
        kind=ArtifactKind.RESULT,
        task_id=task.id,
        produced_by="s1",
        content_type="text/csv",
        confidence=ArtifactConfidence.DEDICATED,
    )

    # completes the step with the artifact id + emits the event
    assert registry.complete_step("s1", art.id) is True
    bus.emit(EventType.STEP_COMPLETED, producer_id="s1")

    # verify the wiring held together
    got = registry.get_task(task.id)
    assert got is not None
    step = next(s for s in got.steps if s.id == "s1")
    assert step.status == StepStatus.COMPLETED
    assert step.output_artifact == art.id
    assert artifacts.read(art.id) == b"latency=42ms,throughput=1200tps"
    assert len(bus.replay(EventType.STEP_COMPLETED)) == 1
    print("  cross-store: claim -> put -> complete -> emit OK")


# --- runner so it works without pytest ---

_TESTS = [
    test_task_round_trip,
    test_dag_guard,
    test_artifact_store,
    test_event_bus,
    test_cross_store,
]


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
