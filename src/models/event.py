from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """Event vocabulary — past-tense facts on the bus (per docs/design/01-domain.md).

    Named <entity>.<past_event>. Consumers react; they don't reply.
    """

    # Task lifecycle
    TASK_CREATED = "task.created"
    TASK_READY = "task.ready"
    TASK_STARTED = "task.started"
    TASK_PAUSED = "task.paused"
    TASK_RESUMED = "task.resumed"
    TASK_AWAITING_APPROVAL = "task.awaiting_approval"
    TASK_APPROVED = "task.approved"
    TASK_REJECTED = "task.rejected"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_STALED = "task.staled"

    # Step lifecycle
    STEP_STARTED = "step.started"
    STEP_CHECKPOINTED = "step.checkpointed"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    STEP_RETRIED = "step.retried"
    STEP_SKIPPED = "step.skipped"

    # Artifact
    ARTIFACT_PRODUCED = "artifact.produced"
    ARTIFACT_VERSIONED = "artifact.versioned"
    ARTIFACT_INVALIDATED = "artifact.invalidated"

    # Knowledge graph
    GRAPH_UPDATED = "graph.updated"
    GRAPH_CONTRADICTION_DETECTED = "graph.contradiction_detected"

    # Resource / system
    GPU_ACQUIRED = "gpu.acquired"
    GPU_RELEASED = "gpu.released"
    WORKER_BOOTED = "worker.booted"
    WORKER_DIED = "worker.died"


class Event(BaseModel):
    """A past-tense fact. Idempotent by `id` — the bus dedups on it (Phase 2).

    For content-derived dedup (same logical event emitted twice on retry),
    pass an explicit `id` derived from the producer. Otherwise a fresh UUID.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: EventType
    producer_id: str
    payload: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
