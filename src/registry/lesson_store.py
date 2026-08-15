import json
import sqlite3
from datetime import UTC, datetime
from hashlib import sha256

from src.models.lesson import Lesson, LessonStatus


class LessonStore:
    """Procedural-memory store: the recurrence engine + retrieval.

    Per docs/design/03-intelligence.md:
    - Lessons are deduped by content-hash of `text` (same text -> same id).
    - `put` is the recurrence engine: first sighting = CANDIDATE (confidence 1);
      a repeat bumps confidence and promotes to ACTIVE at the promotion threshold.
    - Only ACTIVE lessons are retrieved by `get_active` (workers inject these
      into prompts). A one-off hallucination stays CANDIDATE and is never seen.
    """

    def __init__(self, conn: sqlite3.Connection, promotion_threshold: int = 2):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row   # required for row["col"] access in _row_to_lesson
        self.promotion_threshold = promotion_threshold
        self._create_table()

    def _create_table(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lessons (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    confidence INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _row_to_lesson(self, row: sqlite3.Row) -> Lesson:
        """Map a lessons-table row to a Lesson. Single source of truth."""
        return Lesson(
            id=row["id"],
            text=row["text"],
            tags=json.loads(row["tags"]),
            confidence=row["confidence"],
            status=LessonStatus(row["status"]),   # rehydrate: "candidate" -> LessonStatus.CANDIDATE
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def put(self, text: str, tags: list[str]) -> Lesson:
        """Insert or recur a lesson. Returns the resulting Lesson from DB truth.

        Same text -> same content-hash -> same id. First sighting inserts as
        CANDIDATE; a repeat increments confidence and promotes to ACTIVE when
        confidence >= promotion_threshold.
        """
        lesson_id = sha256(text.encode("utf-8")).hexdigest()
        row = self.conn.execute(
            "SELECT status, confidence FROM lessons WHERE id = ?", (lesson_id,)
        ).fetchone()

        if row is None:
            # First sighting: insert as CANDIDATE, confidence 1.
            with self.conn:
                self.conn.execute(
                    "INSERT INTO lessons (id, text, tags, confidence, status, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (lesson_id, text, json.dumps(tags), 1,
                     LessonStatus.CANDIDATE.value, datetime.now(UTC).isoformat()),
                )
        else:
            status = LessonStatus(row["status"])          # enum-to-enum from here on
            confidence = row["confidence"]
            if status == LessonStatus.CANDIDATE:
                confidence += 1
                if confidence >= self.promotion_threshold:
                    status = LessonStatus.ACTIVE
                with self.conn:
                    self.conn.execute(
                        "UPDATE lessons SET confidence=?, status=? WHERE id=?",
                        (confidence, status.value, lesson_id),
                    )
            elif status == LessonStatus.ACTIVE:
                confidence += 1
                with self.conn:
                    self.conn.execute(
                        "UPDATE lessons SET confidence=? WHERE id=?",
                        (confidence, lesson_id),
                    )
            # RETIRED: no-op for now (retire is a later milestone)

        # Single source of truth: re-fetch the committed row, build the Lesson once.
        fresh = self.conn.execute(
            "SELECT * FROM lessons WHERE id = ?", (lesson_id,)
        ).fetchone()
        return self._row_to_lesson(fresh)

    def get_active(self, matching_tags: list[str]) -> list[Lesson]:
        """Return ACTIVE lessons whose tags are a superset of matching_tags.

        Tag-subset match: a lesson surfaces only if every wanted tag is present.
        Instant at hundreds of lessons; semantic (pgvector) retrieval comes later.
        """
        rows = self.conn.execute(
            "SELECT * FROM lessons WHERE status = ?", (LessonStatus.ACTIVE.value,)
        ).fetchall()
        wanted = set(matching_tags)
        return [
            self._row_to_lesson(row)
            for row in rows
            if wanted.issubset(set(json.loads(row["tags"])))
        ]

    def close(self) -> None:
        self.conn.close()
