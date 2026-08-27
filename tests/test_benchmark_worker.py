""" Fake-first BenchmarkWorker test: no GPU, no subprocess, no network.

Overrides _run_subprocess with canned JSON (your real fp16 run + a plausible fp32)
and stubs snapshot_download. Proves the fan-out DAG end to end:
    FetchWeights -> RunBenchmark  x2 -> Analyze comparing both.

"""

import json
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
from src.registry.knowledge_store import KnowledgeStore
from src.registry.registry import TaskRegistry
from src.workers import benchmark_worker
from src.workers.benchmark_worker import BenchmarkWorker

# canned by dtype -- fp16 line us your real measured run
CANNED = {
        "float16": json.dumps({"model": "Qwen/Qwen3-0.6B", "dtype": "float16", "batch_size": 1,
                             "n_tokens": 32, "ttft_ms": 119.9, "tokens_per_sec": 5.64,
                             "peak_vram_mb": 1187.83, "total_time_ms": 5736.04}),
        "float32": json.dumps({"model": "Qwen/Qwen3-0.6B", "dtype": "float32", "batch_size": 1,
                             "n_tokens": 32, "ttft_ms": 152.3, "tokens_per_sec": 2.5,
                             "peak_vram_mb": 2401.5, "total_time_ms": 12752.1}),
}

class _TestableBenchmarkWorker(BenchmarkWorker):
    def _run_subprocess(self, cmd):
        dtype = cmd[cmd.index("--dtype") + 1]
        return CANNED[dtype]

def test_benchmark_fanout_dag():
    conn = sqlite3.connect(":memory:")
    registry = TaskRegistry(conn)
    bus = EventBus(conn)
    artifacts = ArtifactStore(conn, base_dir=tempfile.mkdtemp())
    knowledge = KnowledgeStore(conn)

    # stub the network fetch: no download, return a fake cache path
    benchmark_worker.snapshot_download = lambda repo_id: f"/fake/cache/{repo_id}"

    # build the fan-out task inline (same shape as cli.build_benchmark_task)
    task_id, fetch_id = str(uuid4()), str(uuid4())
    bench_a, bench_b = str(uuid4()), str(uuid4())
    now = datetime.now(UTC)

    def _step(sid, kind, deps, params=None):
        return Step(id=sid, task_id=task_id, kind=kind, dependencies=deps,
            input_artifacts=[], params=params or {})

    task = ResearchTask(
        id=task_id, objective="bench fp16 vs fp32", status=ResearchTaskStatus.PENDING,
        priority=Priority.MEDIUM, owner="gpu.benchmark", fingerprint_id=uuid4(),
        steps=[
            _step(fetch_id, StepKind.FETCH_WEIGHTS, [], {"repo_id": "Qwen/Qwen3-0.6B"}),
            _step(bench_a, StepKind.RUN_BENCHMARK, [fetch_id], {"dtype": "float16", "n_tokens": 32}),
            _step(bench_b, StepKind.RUN_BENCHMARK, [fetch_id], {"dtype": "float32", "n_tokens": 32}),
            _step("analyze", StepKind.ANALYZE, [bench_a, bench_b]),
        ],
        artifacts=[], retry_count=0, retry_budget=3, created_at=now, updated_at=now,
    )
    registry.create_task(task)

    worker = _TestableBenchmarkWorker(
        registry=registry, bus=bus, artifacts=artifacts,
        kinds=[StepKind.FETCH_WEIGHTS, StepKind.RUN_BENCHMARK, StepKind.ANALYZE],
        worker_id="bench-1", knowledge_store=knowledge
    )

    # the driver loop
    while pending := registry.get_pending_steps():
        for step in pending:
            if step.kind in worker.kinds:
                worker._execute(step)

    # all four steps completed
    done = registry.get_task(task_id)
    if done:
        assert all(s.status == StepStatus.COMPLETED for s in done.steps), \
            [ (s.kind, s.status) for s in done.steps]


    # artifacts: 1 WEIGHTS + 2 RESULT + 1 ANALYSIS
    task_arts = artifacts.list_by_task(task_id)
    kinds = sorted(a.kind for a in task_arts)
    assert kinds == [ArtifactKind.ANALYSIS, ArtifactKind.RESULT, ArtifactKind.RESULT, ArtifactKind.WEIGHTS], kinds

    # the analysis computed the real deltas from the canned runs
    analysis_art = next(a for a in task_arts if a.kind == ArtifactKind.ANALYSIS)
    raw_analysis = artifacts.read(analysis_art.id)
    if raw_analysis is None:
        raise ValueError(f"Analysis artifact not found: {analysis_art.id}")
    analysis = json.loads(raw_analysis)
    assert analysis["tokens_per_sec_ratio"] == round(5.64 / 2.5, 3)
    assert analysis["ttft_diff_ms"] == round(119.9 - 152.3, 2)
    assert analysis["peak_vram_diff_mb"] == round(1187.83 - 2401.5, 2)

    # engine tag stamped on results
    for a in task_arts:
        if a.kind == ArtifactKind.RESULT:
            read_art = artifacts.read(a.id)
            assert read_art is not None, (
                f"Did not find the required task artifact: {a.id}"
            )
            assert json.loads(read_art)["engine"] == "transformers"

    # bookend events: 4 steps x (started + completed)
    assert len(bus.replay(EventType.STEP_COMPLETED)) == 4


def test_benchmark_writes_kg():
    conn = sqlite3.connect(":memory:")
    registry = TaskRegistry(conn)
    bus = EventBus(conn)
    artifacts = ArtifactStore(conn, base_dir=tempfile.mkdtemp())
    knowledge=KnowledgeStore(conn)

    benchmark_worker.snapshot_download = lambda repo_id: f"/fake/cache/{repo_id}"

    task_id, fetch_id, analyze_id, update_id, critique_id = (str(uuid4()) for _ in range(5))
    bench_ids = [str(uuid4()), str(uuid4())]
    now = datetime.now(UTC)

    def _step(sid, kind, deps, params=None):
        return Step(id=sid, task_id=task_id, kind=kind, dependencies=deps,
            input_artifacts=[], params=params or {})

    task = ResearchTask(
        id=task_id, objective="bench -> kg", status=ResearchTaskStatus.PENDING,
        priority=Priority.MEDIUM, owner="gpu.benchmark", fingerprint_id=uuid4(),
        steps=[
            _step(fetch_id, StepKind.FETCH_WEIGHTS, [], {"repo_id": "Qwen/Qwen3-0.6B"}),
            _step(bench_ids[0], StepKind.RUN_BENCHMARK, [fetch_id], {"dtype": "float16", "n_tokens": 32}),
            _step(bench_ids[1], StepKind.RUN_BENCHMARK, [fetch_id], {"dtype": "float32", "n_tokens": 32}),
            _step(analyze_id, StepKind.ANALYZE, bench_ids),
            _step(update_id, StepKind.UPDATE_GRAPH, [analyze_id], {"card": "GTX-1650-Ti"}),
            _step(critique_id, StepKind.CRITIQUE, [*bench_ids, analyze_id, update_id]),
        ],
        artifacts=[], retry_count=0, retry_budget=3, created_at=now, updated_at=now,
    )
    registry.create_task(task)

    worker = _TestableBenchmarkWorker(
        registry=registry, bus=bus, artifacts=artifacts,
        kinds=[StepKind.FETCH_WEIGHTS, StepKind.RUN_BENCHMARK, 
            StepKind.ANALYZE, StepKind.UPDATE_GRAPH],
        worker_id="bench-1", knowledge_store=knowledge,
    )
    while pending := registry.get_pending_steps():
        for step in pending:
            if step.kind in worker.kinds:
                worker._execute(step)

    done = registry.get_task(task_id)
    statuses = {s.kind: s.status for s in done.steps} if done else {}
    assert statuses[StepKind.UPDATE_GRAPH] == StepStatus.COMPLETED

    found = knowledge.find_findings(
        metric="tokens_per_sec", model="Qwen/Qwen3-0.6B",
        engine="transformers", card="GTX-1650-Ti",
    )
    assert len(found) == 2, f"expected 2 arm-findings, got {len(found)}"
    assert {f.value for f in found} == {5.64, 2.5}
    assert {f.value for f in found} == {"dtype=float16", "dtype=float32"}

    for metric in ("ttft_ms", "peak_vram_mb"):
        assert len(knowledge.find_findings(
            metric=metric, model="Qwen/Qwen3-0.6B",
            engine="transformers", card="GTX-1650-Ti"
        )) == 2

    kinds = [a.kind for a in artifacts.list_by_task(task_id=task_id)]
    assert ArtifactKind.GRAPH_DELTA in kinds

_TESTS = [test_benchmark_fanout_dag]

if __name__ == "__main__":
    passed = 0
    for t in _TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
        except RuntimeError as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(_TESTS)} passed")
    sys.exit(0 if passed == len(_TESTS) else 1)