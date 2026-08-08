from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ArtifactKind(StrEnum):
    PAPER = "paper"
    SUMMARY = "summary"
    REPO = "repo"
    IMAGE = "image"
    WEIGHTS = "weights"
    RESULT = "result"
    CRITIQUE = "critique"
    REFLECTION = "reflection"
    GRAPH_DELTA = "graph_delta"

class ArtifactConfidence(StrEnum):
    DEDICATED = "dedicated"
    SHARED = "shared"
    
class Artifact(BaseModel):
    id: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[a-fA-F0-9]{64}$"
    )
    kind: ArtifactKind
    task_id: str
    produced_by: str
    payload_uri: str
    content_type: str
    size_bytes: int
    created_at: datetime
    supersedes: str | None = None
    confidence: ArtifactConfidence | None = None