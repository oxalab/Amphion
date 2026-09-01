"""Amphion CLI — the composition root + task runner.

This is the single place that wires the real system together (DB connection,
stores, LLM client) and drives a task end-to-end. Everything else receives
dependencies; this file constructs them.

Usage:
    uv run python -m src.cli <url>
    uv run python -m src.cli https://arxiv.org/abs/2407.08755
"""

import argparse
import json
import os
import sqlite3
from datetime import UTC, datetime
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI

from src.models.research_task import Priority, ResearchTask, ResearchTaskStatus
from src.models.step import Step, StepKind
from src.registry.artifact_store import ArtifactStore
from src.registry.event_bus import EventBus
from src.registry.knowledge_store import KnowledgeStore
from src.registry.lesson_store import LessonStore
from src.registry.registry import TaskRegistry
from src.workers import critic_worker
from src.workers.benchmark_worker import BenchmarkWorker
from src.workers.critic_worker import CriticWorker
from src.workers.paper_worker import PaperWorker
from src.workers.reflect_worker import ReflectWorker

DB_PATH = "amphion.db"
ARTIFACT_DIR = "./data/artifacts"


def load_config() -> tuple[str, str]:
    """Load LLM credentials from .env. Returns (api_key, base_url).

    NOTE: verify these names match your .env. If you get a RuntimeError on
    startup, your .env uses different variable names -- rename here or there.
    """
    load_dotenv()
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("API_KEY and BASE_URL must be set in .env")
    return api_key, base_url


def wire() -> tuple[TaskRegistry, EventBus, ArtifactStore, LessonStore, KnowledgeStore, OpenAI]:
    """Composition root: build the real stores + LLM client. The ONE place that constructs."""
    api_key, base_url = load_config()

    conn = sqlite3.connect(DB_PATH)
    registry = TaskRegistry(conn)
    bus = EventBus(conn)
    artifacts = ArtifactStore(conn, base_dir=ARTIFACT_DIR)
    lessons = LessonStore(conn)
    knowledge = KnowledgeStore(conn)

    model_client = OpenAI(base_url=base_url, api_key=api_key)
    return registry, bus, artifacts, lessons, knowledge, model_client


def _comparison_task(model_repo:str, arm_params: list[dict], objective:str, n_replicates: int = 3) -> ResearchTask:
    """Shared 2-arm skeleton: FetchWeights -> RunBenchmark x (arms x replicates)
    -> Analyze / UpdateGraph / Critique. Arm params replicate identically — no 
    replicate index, so the config stamp groups them in the KG."""
    if n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")
    task_id = str(uuid4())
    fetch_id = str(uuid4())
    expanded = [dict(p) for p in arm_params for _ in range(n_replicates)]
    bench_ids = [str(uuid4()) for _ in expanded]
    analyze_id, update_id, critique_id = str(uuid4()), str(uuid4()), str(uuid4())

    update_graph = Step(
        id=update_id, task_id=task_id, kind=StepKind.UPDATE_GRAPH,
        dependencies=[*bench_ids], input_artifacts=[],
        params={"card": "GTX-1650-Ti"},
    )
    now = datetime.now(UTC)
    fetch = Step(
        id=fetch_id, task_id=task_id, kind=StepKind.FETCH_WEIGHTS,
        dependencies=[], input_artifacts=[], params={"repo_id": model_repo},
    )
    bench_steps = [
        Step(id=bid, task_id=task_id, kind=StepKind.RUN_BENCHMARK,
            dependencies=[fetch_id], input_artifacts=[], params=params)
        for bid, params in zip(bench_ids, expanded)
    ]
    analyze = Step(
        id=analyze_id, task_id=task_id, kind=StepKind.ANALYZE,
        dependencies=bench_ids, input_artifacts=[],
    )
    critique = Step(
        id=critique_id, task_id=task_id, kind=StepKind.CRITIQUE,
        dependencies=[*bench_ids, analyze_id, update_id], input_artifacts=[]
    )
    return ResearchTask(
        id=task_id, objective=objective, status=ResearchTaskStatus.PENDING,
        priority=Priority.MEDIUM, owner="gpu.benchmark", fingerprint_id=uuid4(),
        steps=[fetch, *bench_steps, analyze, update_graph, critique],
        artifacts=[], retry_count=0, retry_budget=3, created_at=now, updated_at=now,
    )
    

def build_paper_task(url: str) -> ResearchTask:
    """A 3-step task: FetchPaper -> ExtractSummary -> Critique.

    Critique depends on BOTH fetch and extract (needs the source to judge the
    summary against). Multi-dependency -> input resolution hands it PAPER + SUMMARY.
    """
    task_id = str(uuid4())
    fetch_id = str(uuid4())
    extract_id = str(uuid4())
    critique_id = str(uuid4())
    reflect_id = str(uuid4())
    now = datetime.now(UTC)

    fetch = Step(
        id=fetch_id, task_id=task_id, kind=StepKind.FETCH_PAPER,
        dependencies=[], input_artifacts=[], params={"url": url},
    )
    extract = Step(
        id=extract_id, task_id=task_id, kind=StepKind.EXTRACT_SUMMARY,
        dependencies=[fetch_id], input_artifacts=[],
    )
    critique = Step(
        id=critique_id, task_id=task_id, kind=StepKind.CRITIQUE,
        dependencies=[fetch_id, extract_id], input_artifacts=[],
    )
    reflect = Step(
        id=reflect_id, task_id=task_id, kind=StepKind.REFLECT,
        dependencies=[extract_id, critique_id],
        input_artifacts=[],
        params={
            "model": "glm-4.5-air",
            "platform": "arxiv",
            "reflects_on": "ExtractSummary"
        },
    )

    return ResearchTask(
        id=task_id,
        objective=f"Summarize + critique paper: {url}",
        status=ResearchTaskStatus.PENDING,
        priority=Priority.MEDIUM,
        owner="llm.reason",
        fingerprint_id=uuid4(),
        steps=[fetch, extract, critique, reflect],
        artifacts=[],
        retry_count=0,
        retry_budget=3,
        created_at=now,
        updated_at=now,
    )


def build_benchmark_task(
    model_repo: str = "Qwen/Qwen3-0.6B",
    dtypes: list[str] | None = None,
    n_replicates: int = 3,
) -> ResearchTask:
    """Dtype knob: fp16 vs fp32 arms."""
    dtypes = dtypes or ["float16", "float32"]
    arm_params = [
        {"knob": "dtype", "dtype": dt, "n_tokens": 128, "batch_size": 1}
        for dt in dtypes
    ]
    return _comparison_task(
        model_repo, arm_params,
        f"Benchmark {model_repo}: {' vs '.join(dtypes)}", n_replicates)


def build_cache_task(
    model_repo: str = "Qwen/Qwen3-0.6B",
    dtype: str = "float32",
    n_replicates: int = 3
) -> ResearchTask:
    """Cache knob, dtype FROZEN: one cause, one effect. (The dtype x cache
    grid is a later decision, on evidence.)"""
    arm_params = [
        {"knob": "use_cache", "use_cache": uc, "dtype": dtype,
            "n_tokens": 128, "batch_size": 1}
        for uc in (False, True)
    ]
    return _comparison_task(
        model_repo, arm_params,
        f"Benchmark {model_repo}: cache off vs on ({dtype})", n_replicates)

def main(url: str) -> None:
    registry, bus, artifacts, lessons, knowledge, client = wire()
    task = build_paper_task(url)
    registry.create_task(task)

    paper_worker = PaperWorker(
        registry=registry, bus=bus, artifacts=artifacts,
        kinds=[StepKind.FETCH_PAPER, StepKind.EXTRACT_SUMMARY],
        worker_id="paper-1", model_client=client,
    )
    critic_worker = CriticWorker(
        registry=registry, bus=bus, artifacts=artifacts,
        kinds=[StepKind.CRITIQUE], worker_id="critic-1",
        model_client=client, lesson_store=lessons, knowledge_store=knowledge
    )
    reflect_worker = ReflectWorker(
        registry=registry, bus=bus, artifacts=artifacts,
        kinds=[StepKind.REFLECT], worker_id="reflect-1",
        model_client=client, lesson_store=lessons
    )
    workers = [paper_worker, critic_worker, reflect_worker]

    # Multi-worker driver: route each pending step to whichever worker handles its kind.
    # get_pending_steps is DAG-guarded, so Critique won't appear until fetch + extract
    # are both Completed (its dependencies). Loop exits when no steps remain.
    run_started = datetime.now(UTC)
    while pending := registry.get_pending_steps():
        for step in pending:
            owner = next((w for w in workers if step.kind in w.kinds), None)
            if owner is not None:
                owner._execute(step)

    # Report: the summary + the critique.
    extract_id = next(s.id for s in task.steps if s.kind == StepKind.EXTRACT_SUMMARY)
    critique_id = next(s.id for s in task.steps if s.kind == StepKind.CRITIQUE)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    extract_step = registry.get_step(extract_id)
    if extract_step and extract_step.output_artifact:
        s = artifacts.read(extract_step.output_artifact)
        print(s.decode("utf-8", errors="replace") if s else "<empty>")
    else:
        print("<ExtractSummary did not produce an output>")

    print("\n" + "=" * 60)
    print("CRITIQUE")
    print("=" * 60)
    critique_step = registry.get_step(critique_id)
    if critique_step and critique_step.output_artifact:
        c = artifacts.read(critique_step.output_artifact)
        print(c.decode("utf-8", errors="replace") if c else "<empty>")
    else:
        print("<Critique did not produce an output>")

    print("\n" + "=" * 60)
    print("EVENT TIMELINE")
    print("=" * 60)
    for event in bus.replay():
        if event.created_at >= run_started:
            print(f"    {event.created_at.isoformat()}  {event.type.value}  ({event.producer_id})")

    print("\n" + "=" * 60)
    print("LESSONS (this run)")
    print("=" * 60)
    reflect_id = next(s.id for s in task.steps if s.kind == StepKind.REFLECT)
    reflect_step = registry.get_step(reflect_id)
    if reflect_step and reflect_step.output_artifact:
        raw = artifacts.read(reflect_step.output_artifact)
        if raw is not None:
            for lesson in json.loads(raw):
                print(f"  - {lesson['text']}")
                print(f"      tags: {', '.join(lesson['tags'])}")
        else:
            print("  <reflection artifact unreadable>")
    else:
        print("  <no lessons>")

    print("\n" + "=" * 60)
    print("ACTIVE LESSONS (promoted across runs)")
    print("=" * 60)
    for lesson in lessons.get_active([]):   # empty tags = all active lessons
        print(f"  - [{lesson.confidence}x] {lesson.text}")
        print(f"      tags: {', '.join(lesson.tags)}")


def main_benchmark(cache: bool = False) -> None:
    registry, bus, artifacts, lessons, knowledge, client = wire()
    task = build_cache_task() if cache else build_benchmark_task()
    registry.create_task(task)

    benchmark_worker = BenchmarkWorker(
        registry=registry, bus=bus, artifacts=artifacts,
        kinds=[StepKind.FETCH_WEIGHTS, StepKind.RUN_BENCHMARK,
            StepKind.ANALYZE, StepKind.UPDATE_GRAPH],
        worker_id="bench-1", knowledge_store=knowledge
    )
    critic_worker = CriticWorker(
        registry=registry, bus=bus, artifacts=artifacts,
        kinds=[StepKind.CRITIQUE], worker_id="critic-1",
        model_client=client, lesson_store=lessons, knowledge_store=knowledge
    )
    workers = [benchmark_worker, critic_worker]

    run_started = datetime.now(UTC)
    while pending := registry.get_pending_steps():
        for step in pending:
            owner = next((w for w in workers if step.kind in w.kinds), None)
            if owner is not None:
                owner._execute(step)
    analyze_id = next(s.id for s in task.steps if s.kind == StepKind.ANALYZE)
    analyze_step = registry.get_step(analyze_id)
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    if analyze_step and analyze_step.output_artifact:
        raw = artifacts.read(analyze_step.output_artifact)
        print(raw.decode("utf-8", errors="replace") if raw else "<empty>")
    else:
        print("<Analyze did not produce an output>")

    print("\n" + "=" * 60)
    print("CRITIQUE")
    print("=" * 60)
    bench_critique_id = next(s.id for s in task.steps if s.kind == StepKind.CRITIQUE)
    bench_critique = registry.get_step(bench_critique_id)
    if bench_critique and bench_critique.output_artifact:
        c = artifacts.read(bench_critique.output_artifact)
        print(c.decode("utf-8", errors="replace") if c else "<empty>")
    else:
        print("<Critique did not produce an output>")

    print("\n" + "=" * 60)
    print("EVENT TIMELINE")
    for event in bus.replay():
        if event.created_at >= run_started:
            print(f"   {event.created_at.isoformat()} {event.type.value}        ({event.producer_id})")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amphion runner")
    sub = parser.add_subparsers(dest="command", required=True)
    p_paper = sub.add_parser("paper", help="summarize + critique + reflect on a URL")
    p_paper.add_argument("url")
    p_bench = sub.add_parser("bench", help="run the dtype comparison benchmark")
    args = parser.parse_args()
    if args.command == "paper":
        main(args.url)
    else:
        main_benchmark(cache=args.cache)