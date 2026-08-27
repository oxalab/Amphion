"""KnowledgeStore → Semantic Memory: A property graph on SQLite

Model: typed nodes + types edges, stored in two tables. Context nodes
(MODEL/ENGINE/CARD/METRIC/CONFIG/TASK) are IDENTITIES — get-or-created
by (type, key), shared across all tasks. FINDING nodes are OBSERVATIONS
— append only, one per measurement, never deduped (Shapre I: aggregated-on-read,
so the judge sees the raw evidence and does the aggregating).

The retrieval query (find_findings) is the 1-hop traversal in SQL: FINDINGs
whose context edges point at the four given nodes. Deliberately NOT filtered 
by config — the critique needs both arms of a comparison to judge direction.
"""

import json
import sqlite3
from dataclasses import dataclass

_FINDING = "FINDING"
_VALID_TYPES = {"MODEL", "ENGINE", "CARD", "METRIC", "CONFIG", "TASK", _FINDING}

@dataclass
class Finding:
    """One measurement observation, as returned to readers. (e.g the critique)"""
    value: float
    task_id: str
    config: str


class KnowledgeStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kg_nodes(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    UNIQUE(type, key)
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kg_edges(
                    src_id INTEGER NOT NULL REFERENCES kg_nodes(id),
                    rel    TEXT NOT NULL,
                    dst_id INTEGER NOT NULL REFERENCES kg_nodes(id),
                    UNIQUE(src_id, rel, dst_id)
                )
                """
            )

    # --- Nodes ---
    def get_or_create_node(self, type: str, key: str) -> int:
        """Identity nodes: one per (type, key), shared across tasks. Race-free
        via ON CONFLICT DO NOTHING + re-SELECT"""
        if type not in _VALID_TYPES:
            raise ValueError(f"unknown node type {type!r}")
        with self.conn:
            self.conn.execute(
                "INSERT INTO kg_nodes (type, key) VALUES (?, ?) "
                "ON CONFLICT(type, key) DO NOTHING",
                (type, key)
            )
        row = self.conn.execute(
            "SELECT id FROM kg_nodes WHERE type = ? AND key = ?", (type, key)
        ).fetchone()
        return row["id"]

    def _add_edge(self, src_id: int, rel: str, dst_id: int) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO kg_edges (src_id, rel, dst_id) VALUES (?, ?, ?) "
                "ON CONFLICT(src_id, rel, dst_id) DO NOTHING",
                (src_id, rel, dst_id),
            )

    # --- Writing Findings---
    def add_finding(self, *, metric: str, model: str, engine: str, card: str, config: str,
        task_id: str, value: float) -> int:
        """Append one measurement observation. Takes DOMAIN args, not node ids ---
        callers never touch graph intenals. Context nodes are get-or-created (identities);
        the FINDING node is a fresh INSERT (append-only)."""
        metric_id = self.get_or_create_node("METRIC", metric)
        model_id = self.get_or_create_node("MODEL", model)
        engine_id = self.get_or_create_node("ENGINE", engine)
        card_id = self.get_or_create_node("CARD", card)
        config_id = self.get_or_create_node("CONFIG", config)
        task_node_id = self.get_or_create_node("TASK", f"task-{task_id}")

        # FINDING key carries the payload: value + provenance (Phase 0 discipline)
        finding_key = json.dumps({"value": value, "task": task_id})
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO kg_nodes (type, key) VALUES (?, ?)", (_FINDING, finding_key)
            )
        finding_id = cur.lastrowid
        if finding_id is None:
            raise ValueError("No Finding's ID found!")
        self._add_edge(finding_id, "MEASURED", metric_id)
        self._add_edge(finding_id, "ON_MODEL", model_id)
        self._add_edge(finding_id, "ON_ENGINE", engine_id)
        self._add_edge(finding_id, "ON_CARD", card_id)
        self._add_edge(finding_id, "WITH_CONFIG", config_id)
        self._add_edge(finding_id, "FROM_TASK", task_node_id)
        return finding_id

    # --- reading (the 1-hop traversal, in SQL) ---
    def find_findings(self, *, metric: str, model: str, engine: str, card: str) -> list[Finding]:
        """All FINDING observations in this four-part context — every dtype/config, every task.
        Returns raw evidence; aggregation is the reader's judgement."""
        metric_id = self._find_node("METRIC", metric)
        model_id = self._find_node("MODEL", model)
        engine_id = self._find_node("ENGINE", engine)
        card_id = self._find_node("CARD", card)
        if None in (metric_id, model_id, engine_id, card_id):
            return []

        rows = self.conn.execute(
            """
            SELECT f.key AS finding_key, c.key AS config_key
            FROM kg_nodes f
            JOIN kg_edges e1 ON e1.src_id = f.id AND e1.rel = 'MEASURED'  AND e1.dst_id = :metric_id
            JOIN kg_edges e2 ON e2.src_id = f.id AND e2.rel = 'ON_MODEL'  AND e2.dst_id = :model_id
            JOIN kg_edges e3 ON e3.src_id = f.id AND e3.rel = 'ON_ENGINE' AND e3.dst_id = :engine_id
            JOIN kg_edges e4 ON e4.src_id = f.id AND e4.rel = 'ON_CARD'   AND e4.dst_id = :card_id
            JOIN kg_edges ec ON ec.src_id = f.id AND ec.rel = 'WITH_CONFIG'
            JOIN kg_nodes c  ON c.id = ec.dst_id
            WHERE f.type = 'FINDING'
            """,
            {"metric_id": metric_id, "model_id": model_id,
                "engine_id": engine_id, "card_id": card_id},
        ).fetchall()

        findings: list[Finding] = []
        for row in rows:
            payload = json.loads(row["finding_key"])
            findings.append(Finding(
                value=payload["value"],
                task_id=payload["task"],
                config=row["config_key"],
            ))
        return findings

    def _find_node(self, type: str, key: str) -> int | None:
        row = self.conn.execute(
            "SELECT id FROM kg_nodes WHERE type = ? AND key = ?", (type, key)
        ).fetchone()
        return row["id"] if row else None
