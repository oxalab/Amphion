from enum import StrEnum

from pydantic import BaseModel, Field, computed_field


class StepKind(StrEnum):
    FETCH_PAPER = "FetchPaper"
    EXTRACT_SUMMARY = "ExtractSummary"
    FETCH_REPO = "FetchRepo"
    BUILD_IMAGE = "BuildImage"
    FETCH_WEIGHTS = "FetchWeights"
    RUN_BENCHMARK = "RunBenchmark"
    ANALYZE = "Analyze"
    CRITIQUE = "Critique"
    REFLECT = "Reflect"
    UPDATE_GRAPH = "UpdateGraph"

class StepStatus(StrEnum):
    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    SKIPPED = "Skipped"
    
class Step(BaseModel):
    id: str
    task_id: str 
    kind: StepKind
    status: StepStatus = StepStatus.PENDING
   
    @computed_field        # (True for agent steps: ExtractSummary, Analyze, Critique, Reflect)
    @property
    def requires_reasoning(self) -> bool:
        return self.kind in {
            StepKind.EXTRACT_SUMMARY,
            StepKind.ANALYZE,
            StepKind.CRITIQUE,
            StepKind.REFLECT
        }
    
    dependencies: list[str]
    input_artifacts: list[str]
    output_artifact: str | None = None
    checkpoint_ref: str | None = None
    attempts: list[str] = Field(default_factory=list)     # Stub: Fixed when Attempt is defined
    worker_id: str | None = None
    