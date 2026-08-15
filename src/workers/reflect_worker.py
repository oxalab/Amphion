import json

from src.models.artifact import ArtifactKind
from src.models.step import Step, StepKind
from src.registry.lesson_store import LessonStore
from src.workers.worker import RunContext, Worker

_VALID_DIMS = {"step_kind", "model", "platform", "engine", "technique"}

class ReflectWorker(Worker):
    DEFAULT_MODEL = "glm-4.5-air"
    def __init__(self, registry, bus, artifacts, kinds, worker_id, model_client, lesson_store: LessonStore):
        super().__init__(registry, bus, artifacts, kinds, worker_id)
        self.model_client = model_client
        self.lesson_store = lesson_store

    def handle(self, step: Step, ctx: RunContext) -> str:
        match step.kind:
            case StepKind.REFLECT: 
                return self._reflect(step, ctx)
            case _:
                return "Nothing to Reflect!"

    def _reflect(self, step: Step, ctx:RunContext) -> str:
        summary = next((a for a in ctx.input_artifacts if a.kind == ArtifactKind.SUMMARY), None)
        critique = next((a for a in ctx.input_artifacts if a.kind == ArtifactKind.CRITIQUE), None)

        if summary is None or critique is None:
            raise ValueError("No summary or critique found for the corresponding!")

        summary_bytes = ctx.artifacts.read(summary.id)
        critique_bytes = ctx.artifacts.read(critique.id)
        if summary_bytes is None or critique_bytes is None:
            raise ValueError("No summary_bytes or critique_bytes to read from")

        run_context = {
            "model": step.params.get("model", self.DEFAULT_MODEL),
            "platform": step.params.get("platform"),
            "reflects_on": step.params.get("reflects_on"),
        }
        model = step.params.get("model", self.DEFAULT_MODEL)
        prompt = self._reflect_prompt(
            summary_text=summary_bytes.decode("utf-8", errors="replace"), 
            critique_text=critique_bytes.decode("utf-8", errors="replace"), 
            run_context=run_context)

        response = self.model_client.chat.completions.create(model=model,
            messages=[{"role":"user", "content":prompt}]
        )
        md = response.choices[0].message.content
        lessons = self._parse_lessons(md)
        for text, tags in lessons:
            self.lesson_store.put(text, tags)

        payload = [{"text":t, "tags": tg} for t, tg in lessons]
        art = ctx.artifacts.put(
            json.dumps(payload, indent=2).encode("utf-8"),
            kind=ArtifactKind.REFLECTION, task_id=step.task_id,
            produced_by=step.id, content_type="application/json"
        )
        return art.id

    def _reflect_prompt(self, summary_text:str, critique_text:str, run_context) -> str:
        context_lines = []
        if run_context.get("model"):
            context_lines.append(f"model: {run_context['model']}")
        if run_context.get("platform"):
            context_lines.append(f"platform: {run_context['platform']}")
        if run_context.get("reflects_on"):
            context_lines.append(f"reflects_on: {run_context['reflects_on']}")
        context_block = "\n".join(context_lines) if context_lines else "(none)"

        return(
            "You harvest reusable lessons from a critique. Each lesson must be a "
            "concrete PRESCRIPTION derived from a specific finding in the critique "
            "-- not a restatement of the problem. 'Be more accurate' is useless. "
            "'Model X fabricates the word 'simulations' when the source only says "
            "'calculate' -- verify method verbs against the source' is useful. "
            "If the critique found nothing actionable, output NO lessons -- empty "
            "reflection is correct; padded reflection is noise.\n\n"
            f"This run:\n{context_block}\n\n"
            "Tags MUST use only these dimensions: step_kind, model, platform, "
            "engine, technique. Use this run's values where relevant. Do not "
            "invent dimensions or values -- omit a tag rather than guess.\n\n"
            "## SUMMARY\n"
            f"{summary_text}\n\n"
            "## CRITIQUE\n"
            f"{critique_text}\n\n"
            "Output each lesson as exactly this block (repeat the block per lesson, "
            "no preamble, no closing summary):\n"
            "### Lesson\n"
            "text: <one actionable rule>\n"
            "tags: step_kind:ExtractSummary, model:glm-4.5-air\n"
        )

    def _parse_lessons(self, md) -> list[tuple[str, list[str]]]:
        blocks = md.split("### Lesson")[1:]
        lessons: list[tuple[str, list[str]]] = []
        for block in blocks:
            text = None
            tags_line = None
            for line in block.splitlines():
                s = line.strip()
                if s.startswith("text:"):
                    text = s[len("text:"):].strip()
                elif s.startswith("tags:"):
                    tags_line = s[len("tags:"):].strip()
            if not text or not tags_line:
                continue
            raw_tags = [t.strip() for t in tags_line.split(",")]
            valid = [t for t in raw_tags if self._valid_tag(t)]
            if not valid:
                continue
            lessons.append((text, valid))
        return lessons
                
    def _valid_tag(self, tag: str) -> bool:
        if ":" not in tag:
            return False
        dim, _, val = tag.partition(":")
        return dim.strip().lower() in _VALID_DIMS and bool(val.strip())