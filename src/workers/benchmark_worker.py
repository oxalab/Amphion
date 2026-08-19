import json
import subprocess
from pathlib import Path
from typing import override

from huggingface_hub import snapshot_download
from torch import sys

from src.models.artifact import ArtifactConfidence, ArtifactKind
from src.models.step import Step, StepKind
from src.workers.worker import RunContext, Worker

BENCHMARK_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark.py"
SUBPROCESS_TIMEOUT = 600 # ms

class BenchmarkWorker(Worker):
    def __init__(self, registry, bus, artifacts, kinds, worker_id):
        super().__init__(registry, bus, artifacts, kinds, worker_id)
    
    @override
    def handle(self, step: Step, ctx: RunContext) -> str:
        match step.kind:
            case StepKind.FETCH_WEIGHTS:
                return self._fetch_weights(step, ctx)
            case StepKind.RUN_BENCHMARK:
                return self._run_benchmark(step, ctx)
            case StepKind.ANALYZE:
                return self._analyze(step, ctx)
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

        cmd = [
            sys.executable, str(BENCHMARK_SCRIPT),
            "--model_path", repo_id,
            "--dtype", dtype,
            "--n_tokens", n_tokens,
            "--batch_size", batch_size,
        ]
        stdout = self._run_subprocess(cmd)

        metrics = json.loads(stdout)
        metrics["engine"] = "transformers"
        art = ctx.artifacts.put(
            json.dumps(metrics, indent=2).encode("utf-8"),
            kind=ArtifactKind.RESULT,
            task_id=step.task_id,
            produced_by=step.id,
            content_type="application/json",
            confidence=ArtifactConfidence.DEDICATED,
        )
        return art.id

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
        results = [a for a in ctx.input_artifacts if a.kind == ArtifactKind.RESULT]
        if len(results) < 2:
            raise ValueError(f"Analyze needs >=2 RESULT inputs, got {len(results)}")
        runs = []
        for r in results:
            raw = ctx.artifacts.read(r.id)
            if raw is None:
                raise ValueError(f"RESULT artifact {r.id} not readable")
            runs.append(json.loads(raw))
        if len(runs) != 2:  
            raise ValueError(f"Analyze compares exactly 2 runs, got {len(runs)}")

        a, b = runs
        # Guard against divide-by-zero (e.g. an fp32 run that barely crawled)
        a_tps = a.get("tokens_per_sec", 0.0)
        b_tps = b.get("tokens_per_sec", 0.0)
        analysis = {
            "comparison": f"{a['dtype']} vs {b['dtype']} ({a['model']})",
            "run_a": a, "run_b": b,
            "tokens_per_sec_ratio": round(a_tps / b_tps, 3) if b_tps else None,
            "ttft_diff_ms": round(a["ttft_ms"] - b["ttft_ms"], 2),
            "peak_vram_diff_mb": round(a["peak_vram_mb"] - b["peak_vram_mb"], 2),
        }
        art = ctx.artifacts.put(
            json.dumps(analysis, indent=2).encode("utf-8"),
            kind=ArtifactKind.ANALYSIS,
            task_id=step.task_id,
            produced_by=step.id,
            content_type="application/json",
            )
        return art.id