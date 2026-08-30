"""One-off migration: stamp WITH_PROTOCOL edges onto findings recorded before
the config/protocol split. Derives each finding's frozen setup from the params
of the RunBenchmark step that produced its source RESULT. Idempotent: re-running
re-derives the same edges and ON CONFLICT makes it a no-op.

Two provenance eras:
 - NEW era findings carry "source" (the RESULT artifact id) -> exact step match.
 - OLD era findings (pre-provenance) have no "source" in the key -> fall back
   to their TASK: every RunBenchmark step in a task ran the same protocol.
"""

import json
import sqlite3

conn = sqlite3.connect("amphion.db")
conn.row_factory = sqlite3.Row

# Protocol defaults of the era that wrote each finding (absent key -> this)
ERA_DEFAULTS = {"n_tokens": 32, "batch_size": 1}


def get_or_create(conn: sqlite3.Connection, type_: str, key: str) -> int:
    conn.execute(
        "INSERT INTO kg_nodes (type, key) VALUES (?, ?) "
        "ON CONFLICT(type, key) DO NOTHING",
        (type_, key),
    )
    row = conn.execute(
        "SELECT id FROM kg_nodes WHERE type = ? AND key = ?", (type_, key)
    ).fetchone()
    return row["id"]


# 1) protocol per RESULT artifact, and per TASK (old-era fallback).
#    Every RunBenchmark step in a task shares one protocol by construction
#    (the builder stamps identical params on all of them).
rows = conn.execute(
    "SELECT s.params AS params, s.output_artifact AS result_id, s.task_id AS task_id "
    "FROM steps s "
    "WHERE s.kind = 'RunBenchmark' AND s.output_artifact IS NOT NULL"
).fetchall()
protocol_of_result, protocol_of_task = {}, {}
for r in rows:
    params = json.loads(r["params"])
    frozen = {k: params.get(k, d) for k, d in ERA_DEFAULTS.items() if k != "dtype"}
    protocol = json.dumps(frozen, sort_keys=True)
    protocol_of_result[r["result_id"]] = protocol
    if r["task_id"] in protocol_of_task and protocol_of_task[r["task_id"]] != protocol:
        # Would mean two protocols inside one task -- impossible for the builder
        # shapes that have existed; refuse rather than silently mis-stamp.
        raise SystemExit(
            f"task {r['task_id']} has conflicting protocols -- aborting, inspect manually"
        )
    protocol_of_task[r["task_id"]] = protocol


# 2) stamp every finding lacking a WITH_PROTOCOL edge, from its provenance
findings = conn.execute(
    "SELECT f.id AS fid, json_extract(f.key, '$.source') AS src, "
    "json_extract(f.key, '$.task') AS task "
    "FROM kg_nodes f WHERE f.type = 'FINDING'"
).fetchall()
stamped = skipped = 0
for f in findings:
    if conn.execute(
        "SELECT 1 FROM kg_edges WHERE src_id = ? AND rel = 'WITH_PROTOCOL'",
        (f["fid"],),
    ).fetchone():
        continue
    protocol = protocol_of_result.get(f["src"]) or protocol_of_task.get(f["task"])
    if protocol is None:
        print(f"  SKIP finding {f['fid']}: no provenance matched to any step")
        skipped += 1
        continue
    pid = get_or_create(conn, "PROTOCOL", protocol)
    conn.execute(
        "INSERT INTO kg_edges (src_id, rel, dst_id) VALUES (?, 'WITH_PROTOCOL', ?) "
        "ON CONFLICT(src_id, rel, dst_id) DO NOTHING",
        (f["fid"], pid),
    )
    stamped += 1

conn.commit()
protocols = [
    r["key"]
    for r in conn.execute(
        "SELECT DISTINCT key FROM kg_nodes WHERE type = ? ORDER BY key", ("PROTOCOL",)
    ).fetchall()
]
print(f"stamped {stamped} findings (skipped {skipped})")
print(f"protocol nodes: {protocols}")
