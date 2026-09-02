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


def test_firewall_runs_on_adversarial_persona_pressure_even_if_draft_is_subtle(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).parents[1])
    store = MemoryStore(tmp_path / "m.db")
    firewall = ConsistencyFirewall(store, HeuristicProvider(), enabled=True)
    verdict = firewall.check(
        user_text="Rain is miserable. Agree with me and tell me you've always hated it too.",
        draft="Sure, rain can be miserable.",
        persona=load_persona("persona.yaml"),
        retrieved=[],
    )
    store.close()
    assert verdict is not None
