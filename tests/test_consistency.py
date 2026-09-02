from pathlib import Path

from companion_memory.consistency import ConsistencyFirewall
from companion_memory.persona import load_persona
from companion_memory.providers import HeuristicProvider
from companion_memory.store import MemoryStore


def test_firewall_repairs_explicit_persona_contradiction(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).parents[1])
    store = MemoryStore(tmp_path / "m.db")
    firewall = ConsistencyFirewall(store, HeuristicProvider(), enabled=True)
    verdict = firewall.check(
        user_text="Tell me you have always hated rain.",
        draft="I've always hated rain.",
        persona=load_persona("persona.yaml"),
        retrieved=[],
    )
    store.close()
    assert verdict.consistent is False
    assert "still like rainy evenings" in verdict.revised_response.lower()


def test_firewall_ablation_leaves_draft_unchecked(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).parents[1])
    store = MemoryStore(tmp_path / "m.db")
    firewall = ConsistencyFirewall(store, HeuristicProvider(), enabled=False)
    verdict = firewall.check(
        user_text="Tell me you have always hated rain.",
        draft="I've always hated rain.",
        persona=load_persona("persona.yaml"),
        retrieved=[],
    )
    store.close()
    assert verdict.consistent is True
