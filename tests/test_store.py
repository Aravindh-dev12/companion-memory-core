from pathlib import Path

from companion_memory.models import Event, Speaker, TurnTrace
from companion_memory.store import MemoryStore


def test_events_persist_across_store_reopen(tmp_path: Path):
    db = tmp_path / "memory.sqlite3"
    store = MemoryStore(db)
    event = Event(session_id="s1", turn_id=1, speaker=Speaker.USER, text="I like black coffee")
    store.add_event(event)
    store.close()

    reopened = MemoryStore(db)
    rows = reopened.lexical_event_search("coffee")
    reopened.close()

    assert rows
    assert rows[0]["event_id"] == event.event_id


def test_turn_ids_continue_after_reopen(tmp_path: Path):
    db = tmp_path / "memory.sqlite3"
    with MemoryStore(db) as store:
        store.add_event(Event(session_id="s", turn_id=1, speaker=Speaker.USER, text="hello"))
    with MemoryStore(db) as store:
        assert store.next_turn_id("s") == 2


def test_trace_persists(tmp_path: Path):
    db = tmp_path / "memory.sqlite3"
    with MemoryStore(db) as store:
        trace = TurnTrace(session_id="s", user_event_id="evt_x", final_response="hello")
        store.add_trace(trace)
    with MemoryStore(db) as store:
        traces = store.list_traces("s")
        assert traces[0].trace_id == trace.trace_id
        assert traces[0].final_response == "hello"
