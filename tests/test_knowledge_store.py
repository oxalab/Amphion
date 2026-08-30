"""KnowledgeStore tests — identity nodes, append-only findings, the traversal.

Three tests prove the load bearing properties:
 - context nodes are IDENTITIES (same model = one node across tasks)
 - findings are APPEND-ONLY OBSERVATIONS (both runs retrievable; no in-place merge)
 - the traversal filters correctly and carries config so callers can split arms
"""

import sqlite3
import sys
import traceback

from src.registry.knowledge_store import KnowledgeStore

CTX = {"metric": "token_per_sec", "model": "Qwen/Qwen3-0.6B",
    "engine": "transformers", "card": "GTX-1650-Ti"}

# canonical protocol stamp: defaults filled at write time, sort_keys=True
P128 = '{"batch_size": 1, "n_tokens": 128}'

def _fresh() -> KnowledgeStore:
    return KnowledgeStore(sqlite3.connect(":memory:"))


def test_findings_append_across_tasks():
    store = _fresh()
    store.add_finding(**CTX, config="dtype=float32", task_id="t1", value=11.79,
                      source="run-1", protocol=P128)
    store.add_finding(**CTX, config="dtype=float32", task_id="t2", value=8.89,
                      source="run-2", protocol=P128)
    store.add_finding(**CTX, config="dtype=float32", task_id="t3", value=8.46,
                      source="run-3", protocol=P128)

    found = store.find_findings(**CTX)
    assert len(found) == 3, f"expected 3 observations, got {len(found)}"
    assert {f.value for f in found} == {11.79, 8.89, 8.46}
    assert all(f.config == "dtype=float32" for f in found)
    print("  append-only: 3 tasks -> 3 observatiosn retrievable OK")


def test_context_nodes_are_identities():
    store = _fresh()
    store.add_finding(**CTX, config="dtype=float16", task_id="t1", value=5.68,
                      source="run-1", protocol=P128)
    store.add_finding(**CTX, config="dtype=float16", task_id="t1", value=8.46,
                      source="run-2", protocol=P128)

    # the model node exists exactly once despite two add_finding calls
    n = store.conn.execute(
        "SELECT COUNT(*) AS n FROM kg_nodes WHERE type='MODEL' AND key=?",
        (CTX["model"],),
    ).fetchone()["n"]
    assert n == 1, f"model node duplicated: {n}"

    # get_or_create returns the SAME id on repeat
    a = store.get_or_create_node("MODEL", CTX["model"])
    b = store.get_or_create_node("MODEL", CTX["model"])
    assert a == b
    print(" identity: context nodes shared across tasks OK")


def test_traversal_filters_and_splits_arms():
    store = _fresh()
    # 3 tasks x 2 dtypes
    for task, fp16, fp32 in [("t1", 5.68, 11.79 ), ("t2", 5.55, 8.89), ("t3", 5.07, 8.46)]:
        store.add_finding(**CTX, config="dtype=float16", task_id=task, value=fp16,
                          source=f"{task}-fp16", protocol=P128)
        store.add_finding(**CTX, config="dtype=float32", task_id=task, value=fp32,
                          source=f"{task}-fp32", protocol=P128)

    found = store.find_findings(**CTX)
    assert len(found) == 6, f"expected both arms, got {len(found)}"

    fp32_vals = [f.value for f in found if f.config == "dtype=float32"]
    fp16_vals = [f.value for f in found if f.config == "dtype=float16"] 
    assert sorted(fp32_vals) == [8.46, 8.89, 11.79]
    assert sorted(fp16_vals) == [5.07, 5.55, 5.68]

    # a different card's findings must NOT leak in.
    store.add_finding(**{**CTX, "card": "RTX:4090"},
        config="dtype=float32", task_id="t9", value=95.0, source="run-9", protocol=P128)
    assert len(store.find_findings(**CTX)) == 6
    # unknown context -> empty evidence, no error
    assert store.find_findings(**{**CTX, "engine": "llama.cpp"}) == []
    print("  traversal: arm-splitting + context isolation OK")


def test_identical_values_distinct_sources_append():
    """The Phase A regression: peak VRAM is deterministic to 0.01 MB, so replicate
    2 and 3 carry the SAME value in the SAME task. Identity-by-value crashed here;
    identity-by-provenance must append all three, and dedup re-ingestion."""
    store = _fresh()
    for run in ("run-1", "run-2", "run-3"):
        store.add_finding(**CTX, config="dtype=float16", task_id="t1",
                          value=1187.83, source=run, protocol=P128)

    found = store.find_findings(**CTX)
    assert len(found) == 3, f"3 replicates of identical VRAM -> 3 findings, got {len(found)}"

    # re-ingesting the same run is a no-op (returns False, appends nothing)
    again = store.add_finding(**CTX, config="dtype=float16", task_id="t1",
                              value=1187.83, source="run-2", protocol=P128)
    assert again is False, "re-ingest of a known (source, value) must return False"
    assert len(store.find_findings(**CTX)) == 3, "re-ingest must not append"


def test_protocol_strict_filter():
    """Config/protocol split: same context, same config, two protocols (eras).
    Strict filter returns only the same-protocol evidence; unknown -> empty, not error
    """
    store = _fresh()
    # plain strings, NOT f-strings: f'{"a": 1}' parses the braces as a
    # replacement field + format spec and blows up
    p32 = '{"batch_size": 1, "n_tokens": 32}'
    p128 = P128
    store.add_finding(**CTX, config="dtype=float32", task_id="t1",
        value=10.41, source="run-1", protocol=p32)
    store.add_finding(**CTX, config="dtype=float32", task_id="t2",
        value=3.99, source="run-2", protocol=p128)
    new = store.find_findings(**CTX, protocol=p128)
    assert len(new) == 1 and new[0].value == 3.99, new
    old = store.find_findings(**CTX, protocol=p32)
    assert len(old) == 1 and old[0].value == 10.41, old

    # no filter -> both eras (legacy reader path)
    assert len(store.find_findings(**CTX)) == 2
    # a protocol never observed -> no evidence, not an error
    assert store.find_findings(**CTX, protocol='{"batch_size": 1, "n_tokens": 8}') == []

_TESTS = [test_findings_append_across_tasks,
    test_context_nodes_are_identities,
    test_traversal_filters_and_splits_arms,
    test_identical_values_distinct_sources_append,
    test_protocol_strict_filter]

if __name__ == "__main__":
    passed = 0
    for t in _TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
        except Exception as e:  # TypeError etc. must fail ONE test, not the module
            print(f"ERROR  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(_TESTS)} passed")
    sys.exit(0 if passed == len(_TESTS) else 1)