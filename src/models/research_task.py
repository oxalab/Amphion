from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

from .step import Step


class ResearchTaskStatus(StrEnum):
   PENDING = "pending"
   READY = "ready"
   RUNNING = "running"
   PAUSED = "paused"
   AWAITING_APPROVAL = "awaiting_approval"
   COMPLETED = "completed"
   FAILED = "failed"
   STALE = "stale"


class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ResearchTask(BaseModel):
    id: str
    objective: str
    status: ResearchTaskStatus
    priority: Priority
    owner: str
    fingerprint_id: UUID
    steps: list[Step]
    artifacts: list[str]
    retry_count: int
    retry_budget: int
    parent_task_id: UUID | None = None
    created_at: datetime
    updated_at: datetime
    error: str | None = None