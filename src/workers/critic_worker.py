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

    def _critique(self, step: Step, ctx: RunContext) -> str:
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
