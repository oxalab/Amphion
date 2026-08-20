"""CriticWorker -- judges other workers' output against its source.

Cross-cutting: critiques summaries now, benchmark results later (milestone 6).
The judging logic (prompt + validation) lives here once, not duplicated per producer.

Output format: markdown with fixed headers (## Faithfulness / ## Coverage / ## Overall),
rating tokens per section. No JSON -- LLMs produce headers reliably, and Reflect
(an LLM) reads the prose. Anti-sycophancy is baked into the prompt: a review that
says "looks good" is a FAILED review.
"""

from src.models.artifact import ArtifactKind
from src.models.step import Step, StepKind
from src.registry.lesson_store import LessonStore
from src.workers.worker import RunContext, Worker

_RATED = {"HIGH", "MEDIUM", "LOW"}
_STABILITY = {"STABLE", "MARGINAL", "UNSTABLE"}
_COVERAGE_TOKENS = {"PRESENT", "THIN", "MISSING"}
_COVERAGE_FIELDS = {"method", "dataset", "metrics", "novelty", "limitations"}


class CriticWorker(Worker):
    DEFAULT_MODEL = "glm-4.5-air"

    def __init__(self, registry, bus, artifacts, kinds, worker_id, model_client, lesson_store: LessonStore):
        super().__init__(registry, bus, artifacts, kinds, worker_id)
        self.model_client = model_client
        self.lesson_store = lesson_store

    def handle(self, step: Step, ctx: RunContext) -> str:
        match step.kind:
            case StepKind.CRITIQUE:
                return self._critique(step, ctx)
            case _:
                raise ValueError(f"CriticWorker cannot handle step kind {step.kind!r}")

    def _critique_summary(self, step: Step, ctx: RunContext) -> str:
        # Multi-dependency: needs PAPER (the source) + SUMMARY (the thing judged).
        paper = next((a for a in ctx.input_artifacts if a.kind == ArtifactKind.PAPER), None)
        summary = next((a for a in ctx.input_artifacts if a.kind == ArtifactKind.SUMMARY), None)
        if paper is None or summary is None:
            raise ValueError("Critique needs both PAPER and SUMMARY input artifacts")

        paper_bytes = ctx.artifacts.read(paper.id)
        summary_bytes = ctx.artifacts.read(summary.id)
        if paper_bytes is None or summary_bytes is None:
            raise ValueError("PAPER or SUMMARY artifact not readable")

        model = step.params.get("model", self.DEFAULT_MODEL)
        # Retrieve active lessons for this step + model -- the loop closes for Critique too.
        lessons = self.lesson_store.get_active(["step_kind:Critique", f"model:{model}"])

        prompt = self._critique_prompt(
            paper_bytes.decode("utf-8", errors="replace"),
            summary_bytes.decode("utf-8", errors="replace"),
            lessons,
        )
        response = self.model_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        critique_md = response.choices[0].message.content

        # Light validation -- never fail the step on malformed prose. Reflect can still
        # use the natural-language critique even if rating tokens are off.
        self._validate_critique(critique_md)

        art = ctx.artifacts.put(
            critique_md.encode("utf-8"),
            kind=ArtifactKind.CRITIQUE,
            task_id=step.task_id,
            produced_by=step.id,
            content_type="text/markdown",
        )
        return art.id


    def _critique(self, step: Step, ctx: RunContext) -> str:
        """Dispatch by input kinds: summary-mode (PAPER+SUMMARY) or
        benchmark-node (ANALYSIS + raw RESULTs). Two prompts, one judge."""
        kinds = {a.kind for a in ctx.input_artifacts}

        if {ArtifactKind.PAPER, ArtifactKind.SUMMARY} <= kinds:
            return self._critique_summary(step, ctx)

        if ArtifactKind.ANALYSIS in kinds and ArtifactKind.RESULT in kinds:
            return self._critique_benchmark(step, ctx)
        raise ValueError(
            f"Critique cannot dipatch on input kinds: {sorted(k.value for k in kinds)}"
        )


    def _critique_benchmark(self, step: Step, ctx: RunContext)-> str:
        """Reads the ANALYSIS as well as raw RESULTs, judge checks analysis
        against its own underlying data, not just attenuate to the claims.
        """
        analysis_art = next((a for a in ctx.input_artifacts if a.kind == ArtifactKind.ANALYSIS), None)
        results = [a for a in ctx.input_artifacts if a.kind == ArtifactKind.RESULT]
        if analysis_art is None or len(results) < 2:
            raise ValueError("Benchmark crtitique needs analysis + >= 2 RESULT inputs")

        analysis_bytes = ctx.artifacts.read(analysis_art.id)
        if analysis_bytes is None:
            raise ValueError(f"ANALYSIS artifact {analysis_art.id} not readable")

        runs_json = []
        for r in results:
            raw = ctx.artifacts.read(r.id)
            if raw is None:
                raise ValueError(f"RESULT artifact {r.id} not readable")
            runs_json.append(raw.decode("utf-8", errors="replace"))

        model = step.params.get("model", self.DEFAULT_MODEL)
        lessons = self.lesson_store.get_active(["step_kind:Critique", f"model:{model}"])
        prompt = self._benchmark_prompt(
            analysis_bytes.decode("utf-8", errors="replace"), runs_json, lessons
        )
        response = self.model_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        critique_md = response.choices[0].message.content
        self._validate_benchmark_critique(critique_md)
        art = ctx.artifacts.put(
            critique_md.encode("utf-8"),
            kind=ArtifactKind.CRITIQUE,
            task_id=step.task_id,
            produced_by=step.id,
            content_type="text/markdown",
        )
        return art.id


    def _benchmark_prompt(self, analysis_text: str, runs_json: list[str], lessons) -> str:
        lessons_block = ""
        if lessons:
            bullets = "\n".join(f"- {lesson.text}" for lesson in lessons)
            lessons_block = "Lessons learned from past critiques (apply where relevant):\n"f"{bullets}\n\n"
        runs_blocks = "\n\n".join(f"## RAW RUN {i + 1}\n{r}" for i, r in enumerate(runs_json))
        return (
            "You are a STRICT, SKEPTICAL benchmark reviewer. Your job is to judge whether "
            "these measurements are VALID and whether the comparison's conclusion is "
            "JUSTIFIED -- not to praise. 'Both runs completed, looks fine' is a FAILED "
            "review. Hunt for the failure modes: noise mistaken for signal, invalid runs, "
            "unstable effect sizes.\n\n"
            f"{lessons_block}"
            "## ANALYSIS (the claims being judged)\n"
            f"{analysis_text}\n\n"
            f"{runs_blocks}\n\n"
            "Judge the benchmark. Output EXACTLY the markdown below -- no preamble:\n\n"
            "## Validity: <HIGH | MEDIUM | LOW>\n"
            "Were both measurements trustworthy? Checklist, name any violation concretely: "
            "sane values (no 0 tok/s, no negative latency); and the runs differ ONLY in the "
            "declared knob -- same model, engine, n_tokens, batch_size (verify against the "
            "raw runs, not the analysis's claim). If fully valid, say so in one line.\n\n"
            "## Stability: <STABLE | MARGINAL | UNSTABLE>\n"
            "Is the delta real -- outside the noise floor? The raw runs are a sample of "
            "size 2 per config: reason about run-to-run variance from what you can see "
            "(e.g. a 25% effect on a metric that swings 25% between runs is MARGINAL at "
            "best). Would more runs change the conclusion? Which specific number is the "
            "shakiest?\n\n"
            "## Overall: <HIGH | MEDIUM | LOW>\n"
            "One line, DERIVED from the above -- not a new judgment.\n"
        )


    def _validate_benchmark_critique(self, text: str) -> dict:
        """Same lenient per-line parse as summary mode, with the Stability token set."""
        ratings: dict[str, str | None] = {"validity": None, "stability": None, "overall": None}
        for line in text.splitlines():
            s = line.strip()
            if "|" in s:
                continue
            if s.startswith("## Validity:"):
                ratings["validity"] = self._first_token_after_colon(s, _RATED)
            if s.startswith("## Stability:"):
                ratings["stability"] = self._first_token_after_colon(s, _STABILITY)
            if s.startswith("## Overall:"):
                ratings["overall"] = self._first_token_after_colon(s, _RATED)
        return ratings

    def _critique_prompt(self, paper_text: str, summary_text: str, lessons) -> str:
        lessons_block = ""
        if lessons:
            bullets = "\n".join(f"- {lesson.text}" for lesson in lessons)
            lessons_block = (
                "Lessons learned from past critiques (apply where relevant):\n"
                f"{bullets}\n\n"
            )
        return (
            "You are a STRICT, SKEPTICAL reviewer evaluating a paper summary against "
            "its source. Your job is to find concrete weaknesses -- not to praise. "
            'A review that says "looks good" is a FAILED review; even strong summaries '
            "have gaps. Be specific: name the exact field missing, quote the unsupported "
            "claim, point to where the summary diverges from the source.\n\n"
            f"{lessons_block}"
            "## SOURCE (the paper)\n"
            f"{paper_text}\n\n"
            "## SUMMARY (being evaluated)\n"
            f"{summary_text}\n\n"
            "Evaluate the summary. Output EXACTLY the markdown below -- no preamble, "
            "no commentary. Replace each 'HIGH | MEDIUM | LOW' with your single choice:\n\n"
            "## Faithfulness: <HIGH | MEDIUM | LOW>\n"
            "Name any unsupported claims or hallucinations. Quote the claim and point to "
            "where it diverges from the source. If genuinely faithful, say so in one line.\n\n"
            "## Coverage\n"
            "- Method: <PRESENT | THIN | MISSING>\n"
            "- Dataset: <PRESENT | THIN | MISSING>\n"
            "- Metrics: <PRESENT | THIN | MISSING>\n"
            "- Novelty: <PRESENT | THIN | MISSING>\n"
            "- Limitations: <PRESENT | THIN | MISSING>\n"
            "For anything THIN or MISSING, name exactly what's absent or shallow. For "
            'Metrics, note whether real numbers are quoted (e.g. "2.3x speedup") or '
            'hand-waved ("faster").\n\n'
            "## Overall: <HIGH | MEDIUM | LOW>\n"
            "One line, DERIVED from the above -- not a new judgment.\n"
        )

    def _validate_critique(self, text: str) -> dict:
        """Parse rating tokens per-line. Lenient: missing/malformed -> None, never raises.

        Skips any line containing '|' (the echoed template, where no pick was made).
        Returns a dict so the validator is unit-testable with canned critique text.
        """
        ratings = {"faithfulness": None, "coverage": {}, "overall": None}

        for line in text.splitlines():
            s = line.strip()
            if "|" in s:
                continue  # echoed "HIGH | MEDIUM | LOW" template -- no pick made
            if s.startswith("## Faithfulness:"):
                ratings["faithfulness"] = self._first_token_after_colon(s, _RATED)
            elif s.startswith("## Overall:"):
                ratings["overall"] = self._first_token_after_colon(s, _RATED)
            elif s.startswith("- ") and ":" in s:
                field, _, val = s[2:].partition(":")
                field = field.strip().lower()
                if field in _COVERAGE_FIELDS:
                    tok = val.strip().split()[0].rstrip(",.") if val.strip() else ""
                    ratings["coverage"][field] = tok if tok in _COVERAGE_TOKENS else None
        return ratings

    @staticmethod
    def _first_token_after_colon(line: str, allowed: set[str]) -> str | None:
        rest = line.split(":", 1)[1].strip()
        tok = rest.split()[0].rstrip(",.") if rest else ""
        return tok if tok in allowed else None
