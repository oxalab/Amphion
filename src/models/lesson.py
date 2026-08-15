from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class LessonStatus(StrEnum):
    CANDIDATE = "candidate"   # seen once, not yet retrievable
    ACTIVE = "active"         # promoted (seen ≥ M), workers retrieve this
    RETIRED = "retired"       # superseded / stale

class Lesson(BaseModel):
    id: str                                                  # content-hash of text (dedup key) 
    text: str                                                # tge lesson 
    tags: list[str]                                          # ["step_kind:ExtractSummary", "model:glm-4.5-air", ...]
    status: LessonStatus = LessonStatus.CANDIDATE            
    confidence: int = 1                                      # incremented on recurrence
    created_at: datetime
    superseded_by: str | None = None