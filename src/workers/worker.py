import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.models.artifact import Artifact
from src.models.event import EventType
from src.models.step import Step, StepKind
from src.registry.artifact_store import ArtifactStore
from src.registry.event_bus import EventBus
from src.registry.registry import TaskRegistry


@dataclass
class RunContext:
    artifacts: ArtifactStore
    events: EventBus
    input_artifacts: list[Artifact]
    
class Worker(ABC):
    def __init__(self, registry: TaskRegistry, bus: EventBus, artifacts: ArtifactStore, kinds: list[StepKind], worker_id: str):
        self.registry = registry
        self.bus = bus
        self.artifacts = artifacts
        self.kinds = kinds
        self.worker_id = worker_id

    @abstractmethod
    def handle(self, step: Step, ctx: RunContext) -> str:
        raise NotImplementedError

    def _execute(self, step: Step):
        # Claim the step with a worker
        execution = self.registry.claim_step(step_id=step.id, worker_id=self.worker_id)
        if not execution:
            return
        # Event bus emits the step started
        self.bus.emit(type=EventType.STEP_STARTED, producer_id=self.worker_id)
        # Try condition to verify if execution fails or succeeds
        try:
            input_artifacts = []
            for dep_id in step.dependencies:
                dep = self.registry.get_step(dep_id)
                if dep is not None and dep.output_artifact is not None:
                    art = self.artifacts.get(dep.output_artifact)
                    if art is not None:
                        input_artifacts.append(art)
            context = RunContext(artifacts=self.artifacts, events=self.bus, input_artifacts=input_artifacts)
            value = self.handle(step, ctx=context)
            self._succeed(step, value)
        except RuntimeError as e:
            self._fail(step, e)
       

    def _succeed(self, step: Step, artifact_id: str) -> None:
        self.registry.complete_step(step.id, artifact_id)
        self.bus.emit(EventType.STEP_COMPLETED, producer_id=self.worker_id,
            payload={"artifact_id":artifact_id})

    def _fail(self, step: Step, error: Exception) -> None:
        self.registry.fail_step(step.id, str(error))
        self.bus.emit(EventType.STEP_FAILED, producer_id=self.worker_id,
            payload={"error": str(error)})

    def run(self):
        while True:
            ran = False
            steps = self.registry.get_pending_steps()
            for step in steps:
                if step.kind in self.kinds:
                    self._execute(step)
                    ran = True
            if not ran:
                time.sleep(0.5)
        
                