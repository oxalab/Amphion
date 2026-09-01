""" Fake-first BenchmarkWorker test: no GPU, no subprocess, no network.

Overrides _run_subprocess with canned JSON (3 distinct replicates per dtype,
built around the real fp16 runs) and stubs snapshot_download. Proves the
replicated fan-out DAG end to end:
    FetchWeights -> RunBenchmark x6 (2 arms x 3 replicates) -> Analyze -> UpdateGraph.

"""

import json
import sqlite3
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from uuid import uuid4

from src.cli import build_cache_task
from src.models.artifact import ArtifactKind
from src.models.event import EventType
from src.models.research_task import Priority, ResearchTask, ResearchTaskStatus
from src.models.step import Step, StepKind, StepStatus
from src.registry.artifact_store import ArtifactStore
from src.registry.event_bus import EventBus
from src.registry.knowledge_store import KnowledgeStore
from src.registry.lesson_store import LessonStore
from src.registry.registry import TaskRegistry
from src.workers import benchmark_worker
from src.workers.benchmark_worker import BenchmarkWorker
from src.workers.critic_worker import CriticWorker

# canned by dtype -- fp16 line us your real measured run
CANNED = {
    ("float16", False): [
            {"model": "Qwen/Qwen3-0.6B", "dtype": "float16", "batch_size": 1,
            "n_tokens": 128, "ttft_ms": 119.9, "tokens_per_sec": 5.64,
            "peak_vram_mb": 1187.83, "total_time_ms": 5736.04},
            {"model": "Qwen/Qwen3-0.6B", "dtype": "float16", "batch_size": 1,
            "n_tokens": 128, "ttft_ms": 124.1, "tokens_per_sec": 5.55,
            "peak_vram_mb": 1187.83, "total_time_ms": 5812.20},
            {"model": "Qwen/Qwen3-0.6B", "dtype": "float16", "batch_size": 1,
            "n_tokens": 128, "ttft_ms": 117.8, "tokens_per_sec": 5.68,
            "peak_vram_mb": 1187.83, "total_time_ms": 5701.90},
        ],
    ("float32", False): [
            {"model": "Qwen/Qwen3-0.6B", "dtype": "float32", "batch_size": 1,
            "n_tokens": 128, "ttft_ms": 152.3, "tokens_per_sec": 2.50,
            "peak_vram_mb": 2401.50, "total_time_ms": 12752.10},
            {"model": "Qwen/Qwen3-0.6B", "dtype": "float32", "batch_size": 1,
            "n_tokens": 128, "ttft_ms": 160.0, "tokens_per_sec": 2.62,
            "peak_vram_mb": 2401.50, "total_time_ms": 12150.30},
            {"model": "Qwen/Qwen3-0.6B", "dtype": "float32", "batch_size": 1,
            "n_tokens": 128, "ttft_ms": 149.5, "tokens_per_sec": 2.41,
            "peak_vram_mb": 2401.50, "total_time_ms": 13110.70},
        ],
    ("float32", True): [
            {"model": "Qwen/Qwen3-0.6B", "dtype": "float32", "batch_size": 1,
            "n_tokens": 128, "use_cache": True, "ttft_ms": 130.1, "tokens_per_sec": 14.2,
            "peak_vram_mb": 2600.55, "total_time_ms": 900.0},
            {"model": "Qwen/Qwen3-0.6B", "dtype": "float32", "batch_size": 1,
            "n_tokens": 128, "use_cache": True, "ttft_ms": 128.7, "tokens_per_sec": 15.1,
            "peak_vram_mb": 2600.55, "total_time_ms": 850.0},
        ],
}


class _FakeCompletions:
    def create(self, **kwargs):
        class _Msg:
            content = ("## Validity: HIGH\nsane\n\n"
                       "## Stability: STABLE\nconsistent\n\n"
                       "## Overall: HIGH\nderived")
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]
        return _Resp

class _FakeModelClient:
    chat = type("Chat", (), {"completions": _FakeCompletions()})()

class _TestableBenchmarkWorker(BenchmarkWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._canned = {k: list(v) for k,v in CANNED.items()}
    
    def _run_subprocess(self, cmd):
        dtype = cmd[cmd.index("--dtype") + 1]
        uc = cmd[cmd.index("--use_cache") + 1] == "true"
        return json.dumps(self._canned[(dtype, uc)].pop(0))

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
    bench_ids = [str(uuid4()) for _ in range(6)]  # 2 arms x 3 replicates
    now = datetime.now(UTC)

    def _step(sid, kind, deps, params=None):
        return Step(id=sid, task_id=task_id, kind=kind, dependencies=deps,
            input_artifacts=[], params=params or {})

    task = ResearchTask(
        id=task_id, objective="bench fp16 vs fp32", status=ResearchTaskStatus.PENDING,
        priority=Priority.MEDIUM, owner="gpu.benchmark", fingerprint_id=uuid4(),
        steps=[
            _step(fetch_id, StepKind.FETCH_WEIGHTS, [], {"repo_id": "Qwen/Qwen3-0.6B"}),
            *(_step(bid, StepKind.RUN_BENCHMARK, [fetch_id],
                    {"knob": "dtype", "dtype": dt, "n_tokens": 128})
              for bid, dt in zip(bench_ids, ["float16"] * 3 + ["float32"] * 3)),
            _step("analyze", StepKind.ANALYZE, bench_ids),
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


    # artifacts: 1 WEIGHTS + 6 RESULT + 1 ANALYSIS
    task_arts = artifacts.list_by_task(task_id)
    kinds = sorted(a.kind for a in task_arts)
    expected = [ArtifactKind.ANALYSIS, ArtifactKind.WEIGHTS] + [ArtifactKind.RESULT] * 6
    assert kinds == sorted(expected), kinds

    # the analysis computed the real deltas from the canned runs
    analysis_art = next(a for a in task_arts if a.kind == ArtifactKind.ANALYSIS)
    raw_analysis = artifacts.read(analysis_art.id)
    if raw_analysis is None:
        raise ValueError(f"Analysis artifact not found: {analysis_art.id}")
    analysis = json.loads(raw_analysis)
    # arms shape: grouped by stamped config, raw values + mean per metric
    assert [arm["config"] for arm in analysis["arms"]] == ["dtype=float16", "dtype=float32"]
    assert all(arm["n"] == 3 for arm in analysis["arms"])
    assert analysis["arms"][0]["metrics"]["tokens_per_sec"]["values"] == [5.64, 5.55, 5.68]
    assert analysis["arms"][0]["metrics"]["tokens_per_sec"]["mean"] == 5.623
    # deltas are ratio/diff of (rounded) means
    assert analysis["tokens_per_sec_ratio"] == round(5.623 / 2.51, 3)
    assert analysis["ttft_diff_ms"] == round((119.9 + 124.1 + 117.8) / 3
                                             - (152.3 + 160.0 + 149.5) / 3, 2)
    assert analysis["peak_vram_diff_mb"] == round(1187.83 - 2401.5, 2)

    # engine tag stamped on results
    for a in task_arts:
        if a.kind == ArtifactKind.RESULT:
            read_art = artifacts.read(a.id)
            assert read_art is not None, (
                f"Did not find the required task artifact: {a.id}"
            )
            assert json.loads(read_art)["engine"] == "transformers"

    # config stamp: params -> arm identity, on every RESULT
    stamped = {
        json.loads(data)["config"]
        for a in task_arts
        if a.kind == ArtifactKind.RESULT and (data := artifacts.read(a.id)) is not None
    }
    assert stamped == {"dtype=float16", "dtype=float32"}, stamped
    protocols = {
        json.loads(data)["protocol"]
        for a in task_arts
        if a.kind == ArtifactKind.RESULT and (data := artifacts.read(a.id)) is not None
    }
    assert protocols == {'{"batch_size": 1, "n_tokens": 128}'}, protocols
        
    # bookend events: 8 steps x (started + completed)
    assert len(bus.replay(EventType.STEP_COMPLETED)) == 8


def test_benchmark_writes_kg():
    conn = sqlite3.connect(":memory:")
    registry = TaskRegistry(conn)
    bus = EventBus(conn)
    artifacts = ArtifactStore(conn, base_dir=tempfile.mkdtemp())
    knowledge=KnowledgeStore(conn)

    benchmark_worker.snapshot_download = lambda repo_id: f"/fake/cache/{repo_id}"

    task_id, fetch_id, analyze_id, update_id = (str(uuid4()) for _ in range(4))
    bench_ids = [str(uuid4()) for _ in range(6)]  # 2 arms x 3 replicates
    now = datetime.now(UTC)

    def _step(sid, kind, deps, params=None):
        return Step(id=sid, task_id=task_id, kind=kind, dependencies=deps,
            input_artifacts=[], params=params or {})

    task = ResearchTask(
        id=task_id, objective="bench -> kg", status=ResearchTaskStatus.PENDING,
        priority=Priority.MEDIUM, owner="gpu.benchmark", fingerprint_id=uuid4(),
        steps=[
            _step(fetch_id, StepKind.FETCH_WEIGHTS, [], {"repo_id": "Qwen/Qwen3-0.6B"}),
            *(_step(bid, StepKind.RUN_BENCHMARK, [fetch_id],
                    {"dtype": dt, "n_tokens": 128})
              for bid, dt in zip(bench_ids, ["float16"] * 3 + ["float32"] * 3)),
            _step(analyze_id, StepKind.ANALYZE, bench_ids),
            _step(update_id, StepKind.UPDATE_GRAPH, bench_ids, {"card": "GTX-1650-Ti"}),
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
    workers = [worker]
    while pending := registry.get_pending_steps():
        progressed = False
        for step in pending:
            owner = next((w for w in workers if step.kind in w.kinds), None)
            if owner is not None:
                owner._execute(step)
                progressed = True
        if not progressed:
            raise RuntimeError(f"no worker claims: {sorted({s.kind.value for s in pending})}")

    done = registry.get_task(task_id)
    statuses = {s.kind: s.status for s in done.steps} if done else {}
    assert statuses[StepKind.UPDATE_GRAPH] == StepStatus.COMPLETED

    found = knowledge.find_findings(
        metric="tokens_per_sec", model="Qwen/Qwen3-0.6B",
        engine="transformers", card="GTX-1650-Ti",
    )
    assert len(found) == 6, f"expected 6 per-sample findings, got {len(found)}"
    per_config: dict[str, set] = {}
    for f in found:
        per_config.setdefault(f.config, set()).add(f.value)
    assert per_config["dtype=float16"] == {5.64, 5.55, 5.68}
    assert per_config["dtype=float32"] == {2.50, 2.62, 2.41}

    for metric in ("ttft_ms", "peak_vram_mb"):
        assert len(knowledge.find_findings(
            metric=metric, model="Qwen/Qwen3-0.6B",
            engine="transformers", card="GTX-1650-Ti"
        )) == 6

    kinds = [a.kind for a in artifacts.list_by_task(task_id=task_id)]
    assert ArtifactKind.GRAPH_DELTA in kinds
    # the delta counts what actually landed: 3 metrics x 3 replicates x 2 arms
    delta_art = next(a for a in artifacts.list_by_task(task_id)
                     if a.kind == ArtifactKind.GRAPH_DELTA)
    data = artifacts.read(delta_art.id) 
    assert data is not None, f"Artifact {delta_art.id} returned None"
    delta = json.loads(data)
    assert delta["findings_added"] == 18, delta


def test_cache_task_writes_kg():
    conn = sqlite3.connect(":memory:")
    registry = TaskRegistry(conn)
    bus = EventBus(conn)
    artifacts = ArtifactStore(conn, base_dir=tempfile.mkdtemp())
    knowledge = KnowledgeStore(conn)
    lessons = LessonStore(conn)
    benchmark_worker.snapshot_download = lambda repo_id: f"/fake/cache/{repo_id}"

    task = build_cache_task(n_replicates=1)
    registry.create_task(task)
    worker = _TestableBenchmarkWorker(
        registry=registry, bus=bus, artifacts=artifacts,
        kinds=[StepKind.FETCH_WEIGHTS, StepKind.RUN_BENCHMARK,
            StepKind.ANALYZE, StepKind.UPDATE_GRAPH],
        worker_id="bench-1", knowledge_store=knowledge)
    critic = CriticWorker(
        registry=registry, bus=bus, artifacts=artifacts,
        kinds=[StepKind.CRITIQUE], worker_id="critic-1",
        model_client=_FakeModelClient(), lesson_store=lessons,
        knowledge_store=knowledge,
    )
    workers = [worker, critic]
    while pending := registry.get_pending_steps():
        progressed = False
        for step in pending:
            owner = next((w for w in workers if step.kind in w.kinds), None)
            if owner is not None:
                owner._execute(step); progressed = True
        if not progressed:
            raise RuntimeError(f"no worker claims : {sorted({s.kind.value for s in pending})}")

    # configs split by the cache knob; protocol carries frozen dtype
    results = [a for a in artifacts.list_by_task(task.id)
        if a.kind == ArtifactKind.RESULT]
    assert {
        json.loads(data)["config"] 
        for a in results if (data := artifacts.read(a.id)) is not None
    } == {"use_cache=False", "use_cache=True"}

    assert {
        json.loads(data)["protocol"]
        for a in results if (data := artifacts.read(a.id)) is not None
    } == {'{"batch_size": 1, "dtype": "float32", "n_tokens": 128}'}

    found = knowledge.find_findings(
        metric="tokens_per_sec", model="Qwen/Qwen3-0.6B",
        engine="transformers", card="GTX-1650-Ti",
        protocol='{"batch_size": 1, "dtype": "float32", "n_tokens": 128}')
    assert {f.config for f in found } == {"use_cache=False", "use_cache=True"}

_TESTS = [
    test_benchmark_fanout_dag,
    test_benchmark_writes_kg,
    test_cache_task_writes_kg,
]

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