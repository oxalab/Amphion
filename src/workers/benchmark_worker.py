import json
import subprocess
import sys
from pathlib import Path
from typing import override

from huggingface_hub import snapshot_download

from src.models.artifact import ArtifactConfidence, ArtifactKind
from src.models.step import Step, StepKind
from src.registry.knowledge_store import KnowledgeStore
from src.workers.worker import RunContext, Worker

BENCHMARK_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark.py"
SUBPROCESS_TIMEOUT = 600  # seconds
_METRICS = ("ttft_ms", "tokens_per_sec", "peak_vram_mb")
_PROTOCOL_DEFAULTS = {"n_tokens": 64, "batch_size": 1, "dtype": "float16"}

class BenchmarkWorker(Worker):
    def __init__(self, registry, bus, artifacts, kinds, worker_id, knowledge_store: KnowledgeStore):
        super().__init__(registry, bus, artifacts, kinds, worker_id)
        self.knowledge_store = knowledge_store
    
    @override
    def handle(self, step: Step, ctx: RunContext) -> str:
        match step.kind:
            case StepKind.FETCH_WEIGHTS:
                return self._fetch_weights(step, ctx)
            case StepKind.RUN_BENCHMARK:
                return self._run_benchmark(step, ctx)
            case StepKind.ANALYZE:
                return self._analyze(step, ctx)
            case StepKind.UPDATE_GRAPH:
                return self._update_graph(step, ctx)
            case _:
                raise ValueError(f"BenchmarkWorker cannot handle step kind {step.kind!r}")

    def _fetch_weights(self, step: Step, ctx: RunContext) -> str:
        repo_id= step.params.get("repo_id", "")
        if not repo_id:
            raise ValueError("FetchWeights step missing params['repo_id']")

        # Idempotent: populates the HF cache, returns the local snapshot path.
        # Re-running with the model already cached is a fast no-op.
        local_path = snapshot_download(repo_id=repo_id)
        record = {"repo_id": repo_id, "local_path": str(local_path)}
        art = ctx.artifacts.put(
            json.dumps(record, indent=2).encode("utf-8"),
            kind=ArtifactKind.WEIGHTS,
            task_id=step.task_id,
            produced_by=step.id,
            content_type="application/json",
        )
        return art.id

    def _run_benchmark(self, step: Step, ctx: RunContext) -> str:
        # The model comes from the FetchWeights input artifact (resolved by the
        # base class from step.dependencies) -- not from params. The DAG carries it.
        weights = next((a for a in ctx.input_artifacts if a.kind == ArtifactKind.WEIGHTS), None)
        if weights is None:
            raise ValueError("RunBenchmark received no WEIGHTS input artifact")
        weights_bytes = ctx.artifacts.read(weights.id)
        if weights_bytes is None:
            raise ValueError(f"WEIGHTS artifact {weights.id} not readable")
        record = json.loads(weights_bytes)
        repo_id = record["repo_id"]

        dtype = step.params.get("dtype", "float16")
        n_tokens = str(step.params.get("n_tokens", 64))
        batch_size = str(step.params.get("batch_size", 1))
        use_cache = str(bool(step.params.get("use_cache", False))).lower()
        cmd = [
            sys.executable, str(BENCHMARK_SCRIPT),
            "--model_path", repo_id,
            "--dtype", dtype,
            "--n_tokens", n_tokens,
            "--batch_size", batch_size,
            "--use_cache", use_cache
        ]
        stdout = self._run_subprocess(cmd)

        metrics = json.loads(stdout)
        metrics["engine"] = "transformers"
        metrics["config"], metrics["protocol"] = self._stamps_from_params(step.params)
        art = ctx.artifacts.put(
            json.dumps(metrics, indent=2).encode("utf-8"),
            kind=ArtifactKind.RESULT,
            task_id=step.task_id,
            produced_by=step.id,
            content_type="application/json",
            confidence=ArtifactConfidence.DEDICATED,
        )
        return art.id

    def _stamps_from_params(self, params: dict) -> tuple[str, str]:
        """Two identities from one params dict, split by ROLE in this task:
        config = the axis that VARIED (declared by the task builder via 'knob')
        protocol = everything frozen, defaults filled at WRITE time (an absent
        key must never silently mean 'whatever the default is later'). Roles 
        swap tasks: in a seq-length sweep n_tokens is the knob and dtype is protocol.
        Canonical JSON so the identity string is stable.
        """
        knob = params.get("knob", "dtype")
        config = f"{knob}={params[knob]}" if knob in params else f"{knob}=__missing__"
        frozen = {k: params.get(k, default)
            for k, default in _PROTOCOL_DEFAULTS.items() if k != knob}
        protocol = json.dumps(frozen, sort_keys=True)
        return config, protocol

    def _run_subprocess(self, cmd: list[str]) -> str:
        """Execute and return stdout. On failure/timeout, raise (-> _fail, with
        stderr captured in the message for attempts log). Tests override this."""
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT, check=False
        )
        if result.returncode != 0:
            stderr_tail = result.stderr.strip().splitlines()[-5:]
            raise RuntimeError(
                f"benchmark script exited {result.returncode}: {'   |  '.join(stderr_tail)}"
            )
        return result.stdout

    def _analyze(self, step: Step, ctx: RunContext) -> str:
        """Group raw runs by stamped config; per-metric raw values + mean per arm;
        deltas as ratio/diff of means. Raw values stay in -- the judge counts
        samples; the mean is the headline, the values are the evidence."""
        results = [a for a in ctx.input_artifacts if a.kind == ArtifactKind.RESULT]
        if len(results) < 2:
            raise ValueError(f"Analyze needs >=2 RESULT inputs, got {len(results)}")

        arms: dict[str, list[dict]] = {}
        for r in results:
            raw = ctx.artifacts.read(r.id)
            if raw is None:
                raise ValueError(f"RESULT artifact {r.id} not readable")
            run = json.loads(raw)
            arms.setdefault(run["config"], []).append(run)
        if len(arms) != 2:
            raise ValueError(f"Analyze compares exactly 2 configs, got {sorted(arms)}")

        arm_rows = []
        for config, runs in sorted(arms.items()):
            first = runs[0]
            arm_rows.append({
                "config": config,
                "n": len(runs),
                # Context fields, identical with an arm by construction.
                # UpdateGraph reads these; it never sees raw runs.
                "model": first["model"],
                "engine": first["engine"],
                "gpu": first.get("gpu"),
                "metrics": {
                    m: {
                        "values": [run[m] for run in runs],
                        "mean": round(sum(run[m] for run in runs) / len(runs), 3),
                    }
                    for m in _METRICS
                },
            })

        a, b = arm_rows
        analysis = {
            "comparison": f"{a['config']} vs {b['config']} ({a['model']})",
            "arms": arm_rows,
            "tokens_per_sec_ratio": (
                round(a["metrics"]["tokens_per_sec"]["mean"]
                    / b["metrics"]["tokens_per_sec"]["mean"], 3)
                if b["metrics"]["tokens_per_sec"]["mean"] else None
            ),
            "ttft_diff_ms": round(
                a["metrics"]["ttft_ms"]["mean"] - b["metrics"]["ttft_ms"]["mean"], 2
            ),
            "peak_vram_diff_mb": round(
                a["metrics"]["peak_vram_mb"]["mean"] - b["metrics"]["peak_vram_mb"]["mean"], 2
            ),
        }
        art = ctx.artifacts.put(
            json.dumps(analysis, indent=2).encode("utf-8"),
            kind=ArtifactKind.ANALYSIS,
            task_id=step.task_id,
            produced_by=step.id,
            content_type="application/json",
        )
        return art.id

    def _update_graph(self, step: Step, ctx: RunContext) -> str:
        """RESULT runs -> KG findings. Pure executor: deterministic JSON parse, zero LLM.

        Reads the raw runs directly (its deps = the RESULT artifacts), NOT the
        derived analysis -- the KG records OBSERVATIONS, never aggregates. One
        finding per (run, metric), keyed by the run's RESULT artifact id:
        identical values from different runs append (deterministic VRAM!),
        re-ingesting the same run is a no-op. 3 metrics x n runs per task.
        """
        results = [a for a in ctx.input_artifacts if a.kind == ArtifactKind.RESULT]
        if not results:
            raise ValueError("UpdateGraph received no RESULT input artifacts")

        card_fallback = step.params.get("card", "GTX-1650-Ti")
        added = 0
        for r in results:
            raw = ctx.artifacts.read(r.id)
            if raw is None:
                raise ValueError(f"RESULT artifact {r.id} not readable")
            run = json.loads(raw)
            card = run.get("gpu", card_fallback)
            for metric in _METRICS:
                appended = self.knowledge_store.add_finding(
                    metric=metric,
                    model=run["model"],
                    engine=run["engine"],
                    config=run["config"],
                    card=card,
                    task_id=step.task_id,
                    value=run[metric],
                    source=r.id,
                    protocol=run["protocol"],
                )
                if appended:
                    added += 1
        delta = {
            "findings_added": added,
            "card": card_fallback,
            "note": "findings deduped by source run -- re-ingesting a task is a no-op",
        }
        art = ctx.artifacts.put(
            json.dumps(delta, indent=2).encode("utf-8"),
            kind=ArtifactKind.GRAPH_DELTA,
            task_id=step.task_id,
            produced_by=step.id,
            content_type="application/json"
        )
        return art.id