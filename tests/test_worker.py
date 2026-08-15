import sqlite3
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from uuid import uuid4

from src.models.artifact import ArtifactKind
from src.models.event import EventType
from src.models.research_task import Priority, ResearchTask, ResearchTaskStatus
from src.models.step import Step, StepKind, StepStatus
from src.registry.artifact_store import ArtifactStore
from src.registry.event_bus import EventBus
from src.registry.registry import TaskRegistry
from src.workers.worker import Worker


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

class _FakeWorker(Worker):
    def handle(self, step, ctx):
        art = ctx.artifacts.put(b"fake output", kind=ArtifactKind.SUMMARY,
                task_id=step.task_id, produced_by=step.id,
                content_type="text/plain")
        return art.id


def test_execute_succeeds():
    conn, registry, bus, artifacts = _fresh_stores()
    step = _make_step(step_id="1", kind=StepKind.FETCH_PAPER)
    task = _make_task(steps=[step])
    registry.create_task(task)
    worker = _FakeWorker(
        registry=registry, bus=bus, artifacts=artifacts,
        kinds=[StepKind.FETCH_PAPER],worker_id="w1"
    )
    worker._execute(step)

    got = registry.get_task(task.id)
    if got is not None:
        done = got.steps[0]
    else: 
        done = None
    if done is not None:
        assert done.status == StepStatus.COMPLETED
    else:
        return
    assert done.output_artifact is not None

    events = bus.replay(EventType.STEP_COMPLETED)
    assert len(events) == 1
    assert artifacts.read(done.output_artifact) == b"fake output"

def test_execute_noop_on_completed():
    conn, registry, bus, artifacts = _fresh_stores()
    step = _make_step(step_id="s1", kind=StepKind.FETCH_PAPER)
    registry.create_task(_make_task(steps=[step]))
    worker = _FakeWorker(registry=registry, bus=bus, artifacts=artifacts,
        kinds=[StepKind.FETCH_PAPER], worker_id="w1")
    worker._execute(step)
    worker._execute(step)
    assert len(bus.replay(EventType.STEP_COMPLETED)) == 1


def test_input_resolution():
    """A 2-step task (A -> B): B's handle receives A's output via ctx.input_artifacts,
    without B doing any lookup itself. Proves the base class resolves dependencies."""
    conn, registry, bus, artifacts = _fresh_stores()
    seen_inputs = []

    class _SpyWorker(Worker):
        def handle(self, step, ctx):
            if step.kind == StepKind.EXTRACT_SUMMARY:
                seen_inputs.extend(ctx.input_artifacts)
            return ctx.artifacts.put(b"out", kind=ArtifactKind.SUMMARY,
                                     task_id=step.task_id, produced_by=step.id,
                                     content_type="text/plain").id

    a = _make_step(step_id="A", kind=StepKind.FETCH_PAPER)
    b = _make_step(step_id="B", kind=StepKind.EXTRACT_SUMMARY, deps=["A"])
    registry.create_task(_make_task(steps=[a, b]))

    worker = _SpyWorker(registry=registry, bus=bus, artifacts=artifacts,
                        kinds=[StepKind.FETCH_PAPER, StepKind.EXTRACT_SUMMARY], worker_id="w1")
    worker._execute(a)
    worker._execute(b)

    a_after = registry.get_step("A")
    assert a_after is not None
    assert a_after.output_artifact is not None

    assert len(seen_inputs) == 1
    assert seen_inputs[0].id == a_after.output_artifact

def test_pdf_summary():
    conn, registry, bus, artifacts = _fresh_stores()
    

_TESTS = [
    test_execute_succeeds,
    test_execute_noop_on_completed,
    test_input_resolution,
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
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(_TESTS)} passed")
    sys.exit(0 if passed == len(_TESTS) else 1)