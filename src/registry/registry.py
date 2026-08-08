import json
import sqlite3
from datetime import UTC, datetime
from uuid import UUID

from src.models.research_task import Priority, ResearchTask, ResearchTaskStatus
from src.models.step import Step, StepKind, StepStatus


class TaskRegistry:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist (idempotent)."""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_tasks (
                    id TEXT PRIMARY KEY,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    fingerprint_id TEXT NOT NULL,
                    artifacts TEXT NOT NULL,
                    retry_count INTEGER NOT NULL,
                    retry_budget INTEGER NOT NULL,
                    parent_task_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS steps (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dependencies TEXT NOT NULL,
                    input_artifacts TEXT NOT NULL,
                    output_artifact TEXT,
                    checkpoint_ref TEXT,
                    attempts TEXT NOT NULL,
                    worker_id TEXT,
                    FOREIGN KEY (task_id) REFERENCES research_tasks(id)
                )
                """
            )

    # --- Task operations ---

    def create_task(self, task: ResearchTask) -> ResearchTask:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO research_tasks (
                    id, objective, status, priority, owner, fingerprint_id,
                    artifacts, retry_count, retry_budget, parent_task_id,
                    created_at, updated_at, error
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    task.id,
                    task.objective,
                    task.status.value,
                    task.priority.value,
                    task.owner,
                    str(task.fingerprint_id),
                    json.dumps(task.artifacts),
                    task.retry_count,
                    task.retry_budget,
                    str(task.parent_task_id) if task.parent_task_id else None,
                    task.created_at.isoformat(),
                    task.updated_at.isoformat(),
                    task.error,
                ),
            )
            for step in task.steps:
                self.conn.execute(
                    """
                    INSERT INTO steps (
                        id, task_id, kind, status, dependencies, input_artifacts,
                        output_artifact, checkpoint_ref, attempts, worker_id
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        step.id,
                        step.task_id,
                        step.kind.value,
                        step.status.value,
                        json.dumps(step.dependencies),
                        json.dumps(step.input_artifacts),
                        step.output_artifact,
                        step.checkpoint_ref,
                        json.dumps(step.attempts),
                        step.worker_id,
                    ),
                )
        return task

    def get_task(self, task_id: str) -> ResearchTask | None:
        task_row = self.conn.execute(
            "SELECT * FROM research_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if task_row is None:
            return None

        step_rows = self.conn.execute(
            "SELECT * FROM steps WHERE task_id = ?", (task_id,)
        ).fetchall()
        steps = [self._row_to_step(row) for row in step_rows]

        return ResearchTask(
            id=task_row["id"],
            objective=task_row["objective"],
            status=ResearchTaskStatus(task_row["status"]),
            priority=Priority(task_row["priority"]),
            owner=task_row["owner"],
            fingerprint_id=UUID(task_row["fingerprint_id"]),
            steps=steps,
            artifacts=json.loads(task_row["artifacts"]),
            retry_count=task_row["retry_count"],
            retry_budget=task_row["retry_budget"],
            parent_task_id=UUID(task_row["parent_task_id"]) if task_row["parent_task_id"] else None,
            created_at=datetime.fromisoformat(task_row["created_at"]),
            updated_at=datetime.fromisoformat(task_row["updated_at"]),
            error=task_row["error"],
        )

    def list_tasks(self, status: ResearchTaskStatus | None = None) -> list[ResearchTask]:
        if status is not None:
            rows = self.conn.execute(
                "SELECT id FROM research_tasks WHERE status = ?", (status.value,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT id FROM research_tasks").fetchall()
        return [t for row in rows if (t := self.get_task(row["id"])) is not None]

    def update_task_status(
        self, task_id: str, status: ResearchTaskStatus, error: str | None = None
    ) -> bool:
        with self.conn:
            cursor = self.conn.execute(
                """
                UPDATE research_tasks
                SET status = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status.value, error, datetime.now(UTC).isoformat(), task_id),
            )
            return cursor.rowcount == 1

    # --- Step operations ---

    def get_pending_steps(self, kind: StepKind | None = None) -> list[Step]:
        """Return claimable steps: Pending, with all dependencies Completed.

        A step is only claimable once every step in its `dependencies` list
        has reached Completed status — this is what enforces DAG ordering.
        Optional `kind` filter narrows to a specific step type.
        """
        pending_rows = self.conn.execute(
            "SELECT * FROM steps WHERE status = ?", (StepStatus.PENDING.value,)
        ).fetchall()

        completed_rows = self.conn.execute(
            "SELECT id FROM steps WHERE status = ?", (StepStatus.COMPLETED.value,)
        ).fetchall()
        completed_ids = {row["id"] for row in completed_rows}

        claimable: list[Step] = []
        for row in pending_rows:
            step = self._row_to_step(row)
            # DAG guard: every dependency must be Completed before this step is claimable.
            if not set(step.dependencies).issubset(completed_ids):
                continue
            if kind is not None and step.kind != kind:
                continue
            claimable.append(step)
        return claimable

    def claim_step(self, step_id: str, worker_id: str) -> bool:
        """Atomic claim: only a Pending step can be claimed. Lock-free via rowcount."""
        with self.conn:
            cursor = self.conn.execute(
                """
                UPDATE steps
                SET status = ?, worker_id = ?
                WHERE id = ? AND status = ?
                """,
                (
                    StepStatus.RUNNING.value,
                    worker_id,
                    step_id,
                    StepStatus.PENDING.value,
                ),
            )
            return cursor.rowcount == 1

    def complete_step(self, step_id: str, output_artifact: str) -> bool:
        """Only the worker currently Running the step may complete it."""
        with self.conn:
            cursor = self.conn.execute(
                """
                UPDATE steps
                SET status = ?, output_artifact = ?
                WHERE id = ? AND status = ?
                """,
                (
                    StepStatus.COMPLETED.value,
                    output_artifact,
                    step_id,
                    StepStatus.RUNNING.value,
                ),
            )
            return cursor.rowcount == 1

    def fail_step(self, step_id: str, error: str) -> bool:
        """Mark a Running step Failed; append the error to its attempts log.

        Read-modify-write of attempts happens inside one transaction so the
        append is atomic (no SQL JSON1 gymnastics — clearer to debug).
        """
        with self.conn:
            row = self.conn.execute(
                "SELECT attempts FROM steps WHERE id = ? AND status = ?",
                (step_id, StepStatus.RUNNING.value),
            ).fetchone()
            if row is None:
                return False

            attempts: list[str] = json.loads(row["attempts"])
            attempts.append(error)

            cursor = self.conn.execute(
                """
                UPDATE steps
                SET status = ?, attempts = ?
                WHERE id = ? AND status = ?
                """,
                (
                    StepStatus.FAILED.value,
                    json.dumps(attempts),
                    step_id,
                    StepStatus.RUNNING.value,
                ),
            )
            return cursor.rowcount == 1

    # --- helpers ---

    def _row_to_step(self, row: sqlite3.Row) -> Step:
        """Map a steps-table row to a Step model (shared by get_task / get_pending_steps)."""
        return Step(
            id=row["id"],
            task_id=row["task_id"],
            kind=StepKind(row["kind"]),
            status=StepStatus(row["status"]),
            dependencies=json.loads(row["dependencies"]),
            input_artifacts=json.loads(row["input_artifacts"]),
            output_artifact=row["output_artifact"],
            checkpoint_ref=row["checkpoint_ref"],
            attempts=json.loads(row["attempts"]),
            worker_id=row["worker_id"],
        )
