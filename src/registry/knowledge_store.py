"""KnowledgeStore → Semantic Memory: A property graph on SQLite

Model: typed nodes + types edges, stored in two tables. Context nodes
(MODEL/ENGINE/CARD/METRIC/CONFIG/TASK) are IDENTITIES — get-or-created
by (type, key), shared across all tasks. FINDING nodes are OBSERVATIONS
— append-only, one per measurement, deduped by SOURCE RUN (provenance),
never by value (Shape I: aggregated-on-read, so the judge sees the raw
evidence and does the aggregating).

The retrieval query (find_findings) is the 1-hop traversal in SQL: FINDINGs
whose context edges point at the four given nodes. Deliberately NOT filtered 
by config — the critique needs both arms of a comparison to judge direction.
"""

import json
import sqlite3
from dataclasses import dataclass

_FINDING = "FINDING"
_VALID_TYPES = {"MODEL", "ENGINE", "CARD", "METRIC", "CONFIG", "PROTOCOL", "TASK", _FINDING}

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
        task_id: str, value: float, source: str, protocol: str) -> bool:
        """Append one measurement observation, identified by its SOURCE run (the
        RESULT artifact id). Takes DOMAIN args, not node ids --- callers never
        touch graph internals. Context nodes are get-or-created (identities).

        Returns True if a new finding was appended; False if this exact (source,
        value) was already recorded --- re-ingesting a run is an idempotent no-op.
        Identical VALUES from different runs are distinct observations: deterministic
        metrics (peak_vram repeats to 0.01 MB) must append, not collide.
        protocol: canonical JSON of the frozen setup (what did NOT vary).
        """
        metric_id = self.get_or_create_node("METRIC", metric)
        model_id = self.get_or_create_node("MODEL", model)
        engine_id = self.get_or_create_node("ENGINE", engine)
        card_id = self.get_or_create_node("CARD", card)
        config_id = self.get_or_create_node("CONFIG", config)
        protocol_id = self.get_or_create_node("PROTOCOL", protocol)
        task_node_id = self.get_or_create_node("TASK", f"task-{task_id}")

        # FINDING key carries the payload + provenance (Phase 0 discipline).
        # Identity is the RUN it came from, NOT the value -- identity-by-value
        # made deterministic metrics crash on replicate 2.
        finding_key = json.dumps({"value": value, "task": task_id, "source": source})
        existing = self.conn.execute(
            "SELECT id FROM kg_nodes WHERE type = ? AND key = ?", (_FINDING, finding_key)
        ).fetchone()
        if existing is not None:
            return False
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
        self._add_edge(finding_id, "WITH_PROTOCOL", protocol_id)
        return True

    # --- reading (the 1-hop traversal, in SQL) ---
    def find_findings(self, *, metric: str, model: str, engine: str, card: str, protocol: str | None = None) -> list[Finding]:
        """All FINDING observations in this four-part context — every dtype/config, every task.
        Returns raw evidence; aggregation is the reader's judgement."""
        metric_id = self._find_node("METRIC", metric)
        model_id = self._find_node("MODEL", model)
        engine_id = self._find_node("ENGINE", engine)
        card_id = self._find_node("CARD", card)
        protocol_id = self._find_node("PROTOCOL", protocol) if protocol else None
        if None in (metric_id, model_id, engine_id, card_id):
            return []
        if protocol is not None and protocol_id is None:
            return []
        
        rows = self.conn.execute(
            """
            SELECT f.key AS finding_key, c.key AS config_key
            FROM kg_nodes f
            JOIN kg_edges e1 ON e1.src_id = f.id AND e1.rel = 'MEASURED'  AND e1.dst_id = :metric_id
            JOIN kg_edges e2 ON e2.src_id = f.id AND e2.rel = 'ON_MODEL'  AND e2.dst_id = :model_id
            JOIN kg_edges e3 ON e3.src_id = f.id AND e3.rel = 'ON_ENGINE' AND e3.dst_id = :engine_id
            JOIN kg_edges e4 ON e4.src_id = f.id AND e4.rel = 'ON_CARD'   AND e4.dst_id = :card_id
            JOIN kg_edges ep ON ep.src_id = f.id AND ep.rel = 'WITH_PROTOCOL'
            JOIN kg_nodes pn ON pn.id = ep.dst_id
            JOIN kg_edges ec ON ec.src_id = f.id AND ec.rel = 'WITH_CONFIG'
            JOIN kg_nodes c  ON c.id = ec.dst_id
            WHERE f.type = 'FINDING' AND (:protocol IS NULL OR pn.key = :protocol)
            """,
            {"metric_id": metric_id, "model_id": model_id,
                "engine_id": engine_id, "card_id": card_id, "protocol": protocol},
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

    def browse(self) -> list[dict]:
        """Every finding, grouped for tree display: metric -> card -> config,
        leaves = (value, task, source, protocol). Read-only, additive — the
        TUI Knowledge screen; no graph logic lives in the UI.

        This is find_findings' sibling: that one answers "what do I know in
        THIS context", this one answers "what do I know at all".
        """
        rows = self.conn.execute(
            """
            SELECT f.key AS finding_key, m.key AS metric, cd.key AS card,
                   c.key AS config, p.key AS protocol
            FROM kg_nodes f
            JOIN kg_edges e1 ON e1.src_id = f.id AND e1.rel = 'MEASURED'
            JOIN kg_nodes m  ON m.id = e1.dst_id AND m.type = 'METRIC'
            JOIN kg_edges e4 ON e4.src_id = f.id AND e4.rel = 'ON_CARD'
            JOIN kg_nodes cd ON cd.id = e4.dst_id
            JOIN kg_edges ec ON ec.src_id = f.id AND ec.rel = 'WITH_CONFIG'
            JOIN kg_nodes c  ON c.id = ec.dst_id
            LEFT JOIN kg_edges ep ON ep.src_id = f.id AND ep.rel = 'WITH_PROTOCOL'
            LEFT JOIN kg_nodes p  ON p.id = ep.dst_id
            WHERE f.type = 'FINDING'
            ORDER BY m.key, cd.key, c.key
            """
        ).fetchall()

        tree: dict[str, dict[str, dict[str, list[dict]]]] = {}
        for row in rows:
            payload = json.loads(row["finding_key"])
            leaf = {
                "value": payload.get("value"),
                "task": payload.get("task"),
                "source": payload.get("source"),
                "protocol": row["protocol"],
            }
            tree.setdefault(row["metric"], {}).setdefault(
                row["card"], {}).setdefault(row["config"], []).append(leaf)
        # flatten to ordered dicts of lists — the shape a Textual Tree eats
        return [
            {"metric": metric, "cards": [
                {"card": card, "configs": [
                    {"config": config, "findings": findings}
                    for config, findings in sorted(configs.items())
                ]}
                for card, configs in sorted(cards.items())
            ]}
            for metric, cards in sorted(tree.items())
        ]
