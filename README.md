# Amphion

> An autonomous research organization for LLM inference performance — it runs its own experiments on real GPUs, critiques its own results, and gets less wrong over time.

Amphion ingests papers, benchmarks LLM inference on actual consumer hardware, judges its own measurements, and accumulates what it learns in a knowledge graph. The interesting part is not any single benchmark — it's that the system's *judgment demonstrably improves* as its memory fills.

## The proof: a judgment that changed on accumulated evidence

The system ran a dtype comparison (fp16 vs fp32, Qwen3-0.6B, transformers, GTX 1650 Ti 4GB) and found something counterintuitive: **fp32 beat fp16 ~1.6–2.1× on tokens/sec** — the opposite of folk wisdom.

Its own judge initially rejected the finding. Then its memory filled in. Same question, three stages of the build:

| Stage | What the judge could see | Stability verdict |
|---|---|---|
| No knowledge graph | Only this task's 2 runs | **UNSTABLE** — *"counter-intuitive... likely noise"* |
| KG built, evidence not wired | Still only its own runs | **MARGINAL** — *"sample of size 2... magnitude unstable"* |
| **Evidence injected from the KG** | 6 samples (3 per config) across 3 tasks — its own just-written findings included | **STABLE** — *"consistent across all 6 recorded samples (3 per config)... perfect reproducibility against the knowledge graph"* — Overall: HIGH |

The direction of the finding never changed. **The epistemics did — memory changed judgment.** The judge counted the KG's samples, split them by config, computed cross-task variance (±0.06 / ±0.41 tok/s), and cited the graph by name. That closing critique is [reproduced below](#the-closing-critique-verbatim).

## What it found (real numbers, real card)

Generic benchmarks assume H100s and tensor cores. A 4GB consumer card is a different physics regime, and Amphion exists to produce those truths systematically:

| Metric | fp16 | fp32 |
|---|---|---|
| tokens/sec (3 comparisons) | 5.55–5.68 | 8.89–11.79 (**~1.6–2.1× faster**) |
| run-to-run variance | **±2%** | **±25%** |
| peak VRAM (identical across runs, to 0.01 MB) | 1187.83 MB | 2368.71 MB (~2× the fp16 footprint) |

- **Finding 1 — fp16 folk wisdom is false on this card.** The GTX 1650 Ti (TU117) has no fp16 tensor cores; small-shape fp16 falls back to untuned paths while cuBLAS's fp32 GEMMs stay hyper-optimized. At 0.6B scale the "half the bytes" bandwidth argument doesn't dominate.
- **Finding 2 — variance is dtype-asymmetric.** The *faster* dtype is the *less reproducible* one (likely VRAM contention with the display OS). One flat ±5% tolerance is wrong in both directions: latency on fp32 needs repeats; memory needs one run.
- **Finding 3 — memory measurements are perfectly deterministic.** Peak VRAM identical to 0.01 MB across every run. Trust-one-run for memory; repeats for latency.

## How it works

**Agency map — agents reason, executors obey.** Workers split into two classes: 🤖 *agents* (LLM-driven reasoning: `ExtractSummary`, `Analyze`, `Critique`, `Reflect`) and ⚙️ *executors* (deterministic code: `FetchPaper`, `RunBenchmark`, `UpdateGraph`). Agentic because the agents reason; reliable because they're scoped. No free-roaming autonomy — every step is a checkpointed node in a declared DAG.

```
paper pipeline:  FetchPaper → ExtractSummary → Critique → Reflect
                                                   │            │
bench pipeline:  FetchWeights → RunBenchmark ×2 ─→ Analyze → UpdateGraph → Critique
                 (fan-out DAG, one knob per arm)      │                        │
                                                       ▼                        ▼
                                              knowledge graph ◀──── critique reads prior
                                              (semantic memory)     findings as evidence
                                                       ▲
                 lessons (procedural memory) ◀── Reflect harvests them from critiques
```

**Two memory loops, both closed:**

- *Procedural* (`LessonStore`): `Reflect` harvests lessons from each critique ("the summary invented specifics — demand quoted numbers"), stores them as candidates, and a recurrence engine promotes repeated lessons to ACTIVE, which are re-injected into future agent prompts.
- *Semantic* (`KnowledgeStore`): a property graph over SQLite. Context nodes (metric / model / engine / card / config) are get-or-create identities; FINDING nodes are append-only, one per measurement. `UpdateGraph` writes 6 findings per benchmark task; `Critique` retrieves them, groups by config, and weighs them as prior evidence.

**Production discipline carried from day one:**

- Every artifact is content-addressed (SHA-256, sharded); every event is idempotent by ID (INSERT OR IGNORE + replay); every store is SQLite with atomic claim/complete guards on steps.
- The benchmark script is an **untrusted-executor boundary**: zero Amphion imports, one pure-JSON stdout line, OOM → stderr + exit 1, run under a subprocess with a hard timeout.
- Fake-first tests via overridable I/O seams — 16 tests, no network, no GPU, no LLM spend.

## The design docs (the systems-thinking evidence)

Built doc-first across four phases, then implemented to milestones with divergences tracked:

| Doc | Covers |
|---|---|
| [`00-foundations`](docs/design/00-foundations.md) | mission, architectural drivers, reproducibility invariant, threat model |
| [`01-domain`](docs/design/01-domain.md) | ResearchTask as the core abstraction, 10-step vocabulary, lifecycles, events |
| [`02-runtime`](docs/design/02-runtime.md) | workers, scheduling, failure handling, idempotency, two-store model |
| [`03-intelligence`](docs/design/03-intelligence.md) | four memory kinds, lesson lifecycle, reflection engine |
| [`04-operability`](docs/design/04-operability.md) | observability floor, security floor, what's deferred and why |
| [`05-build-status`](docs/design/05-build-status.md) | **living truth doc** — what's actually built, findings, deltas from design |

## Run it

```bash
# prerequisites: uv, a CUDA GPU (bench pipeline), any OpenAI-compatible LLM endpoint
uv sync

# LLM access via .env:
#   API_KEY=...      BASE_URL=...

# the dtype-comparison experiment (2 GPU runs + analysis + graph update + critique)
uv run python -m src.cli bench

# the paper pipeline: fetch → summarize → critique → reflect
uv run python -m src.cli paper https://arxiv.org/abs/2407.08755

# tests — no network, no GPU, no LLM spend
uv run python -m tests.test_stores && uv run python -m tests.test_worker
uv run python -m tests.test_paper_worker && uv run python -m tests.test_lesson_store
uv run python -m tests.test_benchmark_worker && uv run python -m tests.test_knowledge_store
```

State persists to `amphion.db` (SQLite), `data/artifacts/` (content-addressed), and `data/hf/` (model cache — weights are a cache, never artifacts).

## The closing critique (verbatim)

> ## Stability: STABLE — current values (5.57 and 10.41 tok/s) match the prior historical observations exactly, and the 2x performance gap is consistent across all 6 recorded samples (3 per config) with minimal variance (±0.06 tok/s for float16, ±0.41 tok/s for float32)...
>
> ## Overall: HIGH — ... perfect reproducibility against the knowledge graph ...

## Known debts & deliberate deferrals

Honesty section — the stuff a sharp interviewer will find anyway:

- **Lesson dedup is exact-text** — LLMs rephrase the same lesson differently, so promotion is rarer than it should be. Semantic dedup (embeddings) is the planned swap-in; the store interface won't change.
- **N=1 per arm within a task, n_tokens=32** — flagged by the system's *own judge* as insufficient methodology. Fixing judge-flagged issues is the next milestone.
- **No container isolation** — benchmark runs as a subprocess on host (deliberate v0 choice; Docker+GPU-on-Windows is the highest stall-risk path). Docker `--network=none` lands with Phase 4 hardening.
- **No lease/heartbeat** — single-driver v0 makes GPU serialization automatic; enforcement is deferred.
- **UpdateGraph re-runs double-append findings** — append-only by design; dedup is aggregate-on-read's job.

## Roadmap

Amphion is the measurement-and-reasoning substrate, not the optimizer:

1. ✅ **Measure** — reproducible benchmarks, fingerprinted runs
2. ✅ **Compare & reason** — critique with validity/stability judgment, evidence injection
3. 🔜 **Propose** — next-experiment recommendations fall out of lessons + KG accumulation
4. ⬜ **Autotune** — systematic config sweeps (a SearchDriver worker on top of this substrate)
5. ⬜ **Kernel synthesis** — LLM-generated CUDA/Triton (a separate system that plugs into this lab)

---

*Built as a v0 on a single 4GB laptop GPU (GTX 1650 Ti), Python 3.14, torch 2.11 + cu128, transformers 5.x. Design docs in [`docs/design/`](docs/design/); living build truth in [`05-build-status.md`](docs/design/05-build-status.md).*
