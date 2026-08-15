import httpx
from bs4 import BeautifulSoup

from src.models.artifact import ArtifactKind
from src.models.step import Step, StepKind
from src.workers.worker import RunContext, Worker


class PaperWorker(Worker):
    """Handles FetchPaper (HTTP fetch -> PAPER artifact) and ExtractSummary
    (LLM call on the paper -> SUMMARY artifact).

    ``model_client`` must be OpenAI-compatible: ``client.chat.completions.create(...)``.
    Inject a fake in tests; the real client in production.
    """

    DEFAULT_MODEL = "glm-4.5-air"

    def __init__(self, *args, model_client, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_client = model_client

    def handle(self, step: Step, ctx: RunContext) -> str:
        match step.kind:
            case StepKind.FETCH_PAPER:
                return self._fetch_paper(step, ctx)
            case StepKind.EXTRACT_SUMMARY:
                return self._extract_summary(step, ctx)
            case _:
                raise ValueError(f"PaperWorker cannot handle step kind {step.kind!r}")

    def _fetch_paper(self, step: Step, ctx: RunContext) -> str:
        url = step.params.get("url")
        if not url:
            raise ValueError("FetchPaper step missing params['url']")
        content = self._fetch_url_content(url)
        is_html = content.lstrip()[:200].lower().startswith((b"<!doctype html", b"<html")) or b"html" in content[:500].lower()
        if is_html:
            text = BeautifulSoup(content, "html.parser").get_text(separator=" ", strip=True)
        else:
            text = content
        bytes_ = text.encode("utf-8") if isinstance(text, str) else text
        artifact = ctx.artifacts.put(
            bytes_,
            kind=ArtifactKind.PAPER,
            task_id=step.task_id,          # was: step.id (that's the step id, not the task)
            produced_by=step.id,           # was: self.worker_id (convention = producing step)
            content_type="text/plain",     # was: "plain/text" (not a real MIME type)
        )
        return artifact.id

    def _extract_summary(self, step: Step, ctx: RunContext) -> str:
        # next() with a default, else StopIteration bypasses the None check below.
        paper = next((a for a in ctx.input_artifacts if a.kind == ArtifactKind.PAPER), None)
        if paper is None:
            raise ValueError("ExtractSummary received no PAPER input artifact")
        paper_bytes = ctx.artifacts.read(paper.id)
        if paper_bytes is None:
            raise ValueError(f"PAPER artifact {paper.id} not readable")

        # v0 simplification: decode as text. Real PDF parsing deferred.
        paper_text = paper_bytes.decode("utf-8", errors="replace")
        model = step.params.get("model", self.DEFAULT_MODEL)
        response = self.model_client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    "Summarize the following paper. Extract: method, dataset, "
                    "key metrics, novelty, and limitations.\n\n" + paper_text
                ),
            }],
        )
        summary = response.choices[0].message.content
        artifact = ctx.artifacts.put(
            summary.encode("utf-8"),
            kind=ArtifactKind.SUMMARY,
            task_id=step.task_id,
            produced_by=step.id,
            content_type="text/plain",
        )
        return artifact.id

    def _fetch_url_content(self, url: str) -> bytes:
        """Fetch raw bytes for a URL. Separate method so tests can override it
        (httpx doesn't open file:// URLs, and we want tests network-free)."""
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()   # 4xx/5xx -> HTTPError -> propagates to _fail
        return response.content
