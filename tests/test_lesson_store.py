"""LessonStore tests -- the recurrence engine + retrieval.

Two tests prove the load-bearing properties:
  - recurrence promotes (the engine that separates signal from hallucination)
  - candidates are never retrieved (only ACTIVE lessons surface to workers)
"""

import sqlite3
import sys
import traceback

from src.models.lesson import LessonStatus
from src.registry.lesson_store import LessonStore


def _fresh() -> LessonStore:
    return LessonStore(sqlite3.connect(":memory:"), promotion_threshold=2)


def test_recurrence_promotes():
    store = _fresh()
    text = "pin thread affinity on windows"

    l1 = store.put(text, ["step_kind:RunBenchmark", "platform:windows"])
    assert l1.status == LessonStatus.CANDIDATE, f"first sighting should be candidate, got {l1.status}"
    assert l1.confidence == 1

    l2 = store.put(text, ["step_kind:RunBenchmark", "platform:windows"])  # recurrence
    assert l2.status == LessonStatus.ACTIVE, f"second sighting should promote, got {l2.status}"
    assert l2.confidence == 2

    l3 = store.put(text, ["step_kind:RunBenchmark", "platform:windows"])
    assert l3.status == LessonStatus.ACTIVE   # already active, stays active
    assert l3.confidence == 3                 # just bumps

    # different text -> different lesson, independent candidate
    other = store.put("use flash attention", ["step_kind:BuildImage"])
    assert other.status == LessonStatus.CANDIDATE
    assert other.id != l1.id


def test_only_active_retrieved_and_tag_filtered():
    store = _fresh()
    # a lone candidate -- must NOT surface
    store.put("never promoted one-off", ["step_kind:ExtractSummary", "model:glm-4.5-air"])
    # a promoted lesson (put twice)
    store.put("real pattern", ["step_kind:ExtractSummary", "model:glm-4.5-air"])
    store.put("real pattern", ["step_kind:ExtractSummary", "model:glm-4.5-air"])

    # retrieve by a tag subset
    got = store.get_active(["step_kind:ExtractSummary"])
    assert len(got) == 1, f"expected only the promoted lesson, got {len(got)}"
    assert got[0].text == "real pattern"
    assert got[0].status == LessonStatus.ACTIVE

    # tag that doesn't match -> nothing
    assert store.get_active(["step_kind:RunBenchmark"]) == []

    # narrower tag subset still matches (superset on the lesson side)
    assert len(store.get_active(["step_kind:ExtractSummary", "model:glm-4.5-air"])) == 1


_TESTS = [test_recurrence_promotes, test_only_active_retrieved_and_tag_filtered]

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
        except Exception as e:
            print(f"ERROR  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(_TESTS)} passed")
    sys.exit(0 if passed == len(_TESTS) else 1)
