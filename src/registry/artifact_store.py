import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from src.models.artifact import Artifact, ArtifactConfidence, ArtifactKind


class ArtifactStore:
    """Content-addressed file storage + metadata in SQLite.

    Per docs/design/02-runtime.md + 01-domain.md:
    - The SHA-256 hash of the content IS the artifact id (and the dedup key).
    - Files are sharded by the first 2 hex chars to avoid huge directories.
    - Idempotent: identical content always maps to the same id; a second `put`
      returns the existing artifact without rewriting the file or metadata.
    """

    def __init__(self, conn: sqlite3.Connection, base_dir: str | Path):
        self.conn = conn
        self.base_dir = Path(base_dir)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    produced_by TEXT NOT NULL,
                    payload_uri TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    supersedes TEXT,
                    confidence TEXT
                )
                """
            )

    def put(
        self,
        content: bytes,
        *,
        kind: ArtifactKind,
        task_id: str,
        produced_by: str,
        content_type: str,
        supersedes: str | None = None,
        confidence: ArtifactConfidence | None = None,
    ) -> Artifact:
        """Store content-addressed bytes. Idempotent on content hash.

        First writer wins on metadata (provenance: task_id / produced_by).
        Duplicate puts with identical content return the existing artifact.
        """
        artifact_id = hashlib.sha256(content).hexdigest()

        # Idempotent fast path: identical content already stored.
        existing = self.get(artifact_id)
        if existing is not None:
            return existing

        # Write the file, sharded by first 2 hex chars. Skip if a concurrent
        # put already wrote it (same bytes, harmless).
        payload_uri = f"{artifact_id[:2]}/{artifact_id}"
        path = self.base_dir / payload_uri
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)

        artifact = Artifact(
            id=artifact_id,
            kind=kind,
            task_id=task_id,
            produced_by=produced_by,
            payload_uri=payload_uri,
            content_type=content_type,
            size_bytes=len(content),
            created_at=datetime.now(UTC),
            supersedes=supersedes,
            confidence=confidence,
        )

        # INSERT OR IGNORE guards the rare concurrent-put race on metadata.
        with self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO artifacts (
                    id, kind, task_id, produced_by, payload_uri, content_type,
                    size_bytes, created_at, supersedes, confidence
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    artifact.id,
                    artifact.kind.value,
                    artifact.task_id,
                    artifact.produced_by,
                    artifact.payload_uri,
                    artifact.content_type,
                    artifact.size_bytes,
                    artifact.created_at.isoformat(),
                    artifact.supersedes,
                    artifact.confidence.value if artifact.confidence else None,
                ),
            )
        return artifact

    def get(self, artifact_id: str) -> Artifact | None:
        """Fetch metadata only (no bytes)."""
        row = self.conn.execute(
            "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_artifact(row)

    def exists(self, artifact_id: str) -> bool:
        return self.get(artifact_id) is not None

    def read(self, artifact_id: str) -> bytes | None:
        """Read artifact content. Returns None if missing or file vanished."""
        artifact = self.get(artifact_id)
        if artifact is None:
            return None
        path = self.base_dir / artifact.payload_uri
        return path.read_bytes() if path.exists() else None

    def list_by_task(self, task_id: str) -> list[Artifact]:
        rows = self.conn.execute(
            "SELECT * FROM artifacts WHERE task_id = ?", (task_id,)
        ).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    def _row_to_artifact(self, row: sqlite3.Row) -> Artifact:
        return Artifact(
            id=row["id"],
            kind=ArtifactKind(row["kind"]),
            task_id=row["task_id"],
            produced_by=row["produced_by"],
            payload_uri=row["payload_uri"],
            content_type=row["content_type"],
            size_bytes=row["size_bytes"],
            created_at=datetime.fromisoformat(row["created_at"]),
            supersedes=row["supersedes"],
            confidence=ArtifactConfidence(row["confidence"]) if row["confidence"] else None,
        )
