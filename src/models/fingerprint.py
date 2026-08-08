from uuid import UUID

from pydantic import BaseModel


class Fingerprint(BaseModel):
    # TODO: Phase 0 fields
    id: UUID