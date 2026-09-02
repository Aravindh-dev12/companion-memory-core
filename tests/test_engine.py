from dataclasses import replace
from pathlib import Path

from companion_memory.config import Settings
from companion_memory.factory import build_engine


def _settings(tmp_path: Path) -> Settings:
    return replace(
        Settings(),
        db_path=tmp_path / "memory.sqlite3",
        persona_path=Path("persona.yaml"),
        provider="heuristic",
    )


def test_engine_updates_relationship_and_recalls_current_state(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).parents[1])
    settings = _settings(tmp_path)
    engine = build_engine(settings)
    engine.process_turn(session_id="s1", user_text="My girlfriend is Maya")
    engine.process_turn(session_id="s1", user_text="Maya and I broke up")
    trace = engine.process_turn(session_id="s1", user_text="Am I still with Maya?")
    active = engine.store.list_active_facts(fact_key="user::partner")
    engine.store.close()
    assert [f.value for f in active] == ["none"]
    assert trace.final_response.lower().startswith("no")


def test_engine_memory_survives_new_process_store_instance(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).parents[1])
    settings = _settings(tmp_path)
    engine = build_engine(settings)
    engine.process_turn(session_id="first", user_text="My sister's name is Nina")
    engine.store.close()

    engine2 = build_engine(settings)
    trace = engine2.process_turn(session_id="second", user_text="What's my sister's name?")
    engine2.store.close()
    assert "Nina" in trace.final_response


def test_persona_commitment_is_recorded(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).parents[1])
    settings = _settings(tmp_path)
    engine = build_engine(settings)
    engine.process_turn(session_id="s", user_text="Do you like rainy evenings?")
    commitments = engine.store.list_active_facts(subject="companion")
    engine.store.close()
    assert any(f.value == "likes rainy evenings" for f in commitments)
