import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import Callable

from src.models.event import Event, EventType

Handler = Callable[[Event], None]


class EventBus:
    """In-process event bus: sync dispatch + durable log for replay.

    Per docs/design/02-runtime.md:
    - Events are past-tense facts, logged by `id` (idempotent).
    - Subscribers react; they don't reply.
    - v0 dispatch is synchronous — subscribers run inline in the emitter's call
      stack. Fine while subscribers are fast (DB writes, logging).
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._init_db()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    producer_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Register a handler for an event type. Multiple handlers per type allowed."""
        self._subscribers[event_type.value].append(handler)

    def emit(
        self,
        type: EventType,
        *,
        producer_id: str,
        payload: dict | None = None,
        event_id: str | None = None,
    ) -> tuple[Event, bool]:
        """Log + dispatch an event.

        Returns (event, was_new). `was_new` is False when the event was a
        duplicate (same `id` already logged) — per Phase 2 idempotency,
        duplicates are silently absorbed and NOT re-dispatched.

        Pass an explicit `event_id` for content-derived dedup (e.g. derived
        from the step run). Omit it for a fresh one-off event.
        """
        from uuid import uuid4

        event = Event(
            id=event_id or str(uuid4()),
            type=type,
            producer_id=producer_id,
            payload=payload or {},
        )

        # Idempotent log: INSERT OR IGNORE — duplicate id is a no-op.
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT OR IGNORE INTO events (id, type, producer_id, payload, created_at)
                VALUES (?,?,?,?,?)
                """,
                (
                    event.id,
                    event.type.value,
                    event.producer_id,
                    json.dumps(event.payload),
                    event.created_at.isoformat(),
                ),
            )
            was_new = cursor.rowcount == 1

        # Only dispatch newly-logged events. Duplicates don't re-trigger subscribers.
        # NOTE: handlers run inline (sync). A raising handler aborts dispatch to
        # later subscribers — v0 surfaces bugs fast. Wrap in try/except if a flaky
        # subscriber must be isolated.
        if was_new:
            for handler in self._subscribers.get(event.type.value, []):
                handler(event)

        return event, was_new

    def replay(self, event_type: EventType | None = None) -> list[Event]:
        """Replay logged events in creation order — for recovery / debugging.

        Reads only; does NOT re-dispatch. The caller decides what to do with them.
        """
        if event_type is not None:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE type = ? ORDER BY created_at",
                (event_type.value,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM events ORDER BY created_at"
            ).fetchall()

        return [
            Event(
                id=row["id"],
                type=EventType(row["type"]),
                producer_id=row["producer_id"],
                payload=json.loads(row["payload"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]
