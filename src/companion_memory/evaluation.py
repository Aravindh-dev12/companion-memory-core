from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .config import Settings
from .factory import build_engine
from .models import FactStatus, MemoryType, Modality, ResolutionAction, RetrievalMode, Speaker


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class ScenarioResult:
    scenario_id: str
    checks: list[Check] = field(default_factory=list)
    responses: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "passed": self.passed,
            "checks": [c.__dict__ for c in self.checks],
            "responses": self.responses,
        }


@dataclass
class EvalSummary:
    results: list[ScenarioResult]
    config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        total = len(self.results)
        passed = sum(r.passed for r in self.results)
        checks = [c for r in self.results for c in r.checks]
        return {
            "config": self.config,
            "scenario_pass_rate": passed / total if total else 0.0,
            "check_pass_rate": sum(c.passed for c in checks) / len(checks) if checks else 0.0,
            "scenarios": [r.as_dict() for r in self.results],
        }


def apply_ablation(settings: Settings, ablation: str) -> Settings:
    if ablation == "full":
        return settings
    if ablation == "lexical_only":
        return replace(settings, retrieval_mode=RetrievalMode.LEXICAL_ONLY)
    if ablation == "structured_only":
        return replace(settings, retrieval_mode=RetrievalMode.STRUCTURED_ONLY)
    if ablation == "semantic_only":
        return replace(settings, retrieval_mode=RetrievalMode.SEMANTIC_ONLY)
    if ablation == "no_temporal":
        return replace(settings, temporal_filter=False, temporal_resolution=False)
    if ablation == "no_firewall":
        return replace(settings, consistency_check=False)
    if ablation == "no_memory":
        return replace(settings, memory_enabled=False, full_history_context=False)
    if ablation == "full_history":
        return replace(settings, memory_enabled=False, full_history_context=True, recent_event_limit=100000)
    if ablation == "vector_bag":
        return replace(settings, retrieval_mode=RetrievalMode.SEMANTIC_ONLY, temporal_filter=False)
    if ablation == "oracle":
        return replace(
            settings,
            retrieval_mode=RetrievalMode.ORACLE,
            temporal_filter=False,
            retrieval_limit=10000,
            response_model=settings.oracle_model,
            consistency_check=False,
        )
    raise ValueError(f"unknown ablation: {ablation}")


def load_scenarios(directory: Path) -> list[dict[str, Any]]:
    return [yaml.safe_load(path.read_text()) for path in sorted(directory.glob("*.yaml"))]


def run_scenarios(
    scenario_dir: Path,
    *,
    base_settings: Settings,
    ablation: str = "full",
    preserve_turn_distance: bool = False,
    output_dir: Path | None = None,
) -> EvalSummary:
    settings = apply_ablation(base_settings, ablation)
    results: list[ScenarioResult] = []
    output_dir = output_dir or Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario in load_scenarios(scenario_dir):
        db_path = output_dir / f"{scenario['scenario_id']}.{ablation}.sqlite3"
        if db_path.exists():
            db_path.unlink()
        scenario_settings = replace(settings, db_path=db_path)
        engine = build_engine(scenario_settings)
        try:
            results.append(_run_one(engine, scenario, preserve_turn_distance=preserve_turn_distance))
        finally:
            engine.store.close()

    summary = EvalSummary(
        results=results,
        config={
            "provider": settings.provider,
            "ablation": ablation,
            "retrieval_mode": settings.retrieval_mode.value,
            "temporal_filter": settings.temporal_filter,
            "temporal_resolution": settings.temporal_resolution,
            "consistency_check": settings.consistency_check,
            "extraction_model": settings.extraction_model,
            "resolution_model": settings.resolution_model,
            "response_model": settings.response_model,
            "verification_model": settings.verification_model,
            "oracle_model": settings.oracle_model,
            "embedding_model": settings.embedding_model,
            "memory_enabled": settings.memory_enabled,
            "full_history_context": settings.full_history_context,
        },
    )
    (output_dir / f"results.{ablation}.json").write_text(json.dumps(summary.as_dict(), indent=2))
    (output_dir / f"results.{ablation}.md").write_text(_summary_markdown(summary))
    return summary


def _run_one(engine, scenario: dict[str, Any], *, preserve_turn_distance: bool) -> ScenarioResult:
    sid = scenario["scenario_id"]
    result = ScenarioResult(scenario_id=sid)
    items = scenario.get("steps") or scenario.get("probes") or []
    session_index = 1
    session_id = f"eval-{sid}-s{session_index}"
    previous_target = 0

    for item in items:
        if item.get("consolidate"):
            consolidated = engine.consolidate_session(session_id)
            result.checks.append(
                Check(
                    "session_consolidation",
                    consolidated.summary_fact_id is not None or consolidated.skipped,
                    f"summary_fact_id={consolidated.summary_fact_id}; skipped={consolidated.skipped}",
                )
            )
            continue
        if item.get("new_session"):
            session_index += 1
            session_id = f"eval-{sid}-s{session_index}"
            previous_target = 0
        target_turn = int(item.get("turn", previous_target + 1))
        if preserve_turn_distance:
            _advance_with_filler(engine, session_id, target_turn)
        previous_target = target_turn
        text = item.get("user_fact") or item.get("distractor") or item.get("probe") or item.get("user")
        if not text:
            continue
        trace = engine.process_turn(session_id=session_id, user_text=text)
        result.responses.append(
            {
                "turn": target_turn,
                "input": text,
                "response": trace.final_response,
                "open_loops": [loop.model_dump(mode="json") for loop in trace.open_loops],
                "retrieved": [
                    {
                        "fact_key": m.fact.fact_key,
                        "value": m.fact.value,
                        "status": m.fact.status.value,
                        "score": m.final_score,
                        "reason": m.retrieval_reason,
                    }
                    for m in trace.retrieved
                ],
            }
        )
        if "expected_transition" in item:
            expected = ResolutionAction(item["expected_transition"])
            transitions = engine.store.transitions_for_event(trace.user_event_id)
            observed = [t.action for t in transitions]
            result.checks.append(
                Check(
                    "transition",
                    expected in observed,
                    f"expected={expected.value}; observed={[x.value for x in observed]}",
                )
            )
        if "expected_modality" in item:
            expected_modality = Modality(item["expected_modality"])
            observed_modalities = [claim.modality for claim in trace.extracted_claims]
            result.checks.append(
                Check(
                    "modality",
                    expected_modality in observed_modalities,
                    f"expected={expected_modality.value}; observed={[m.value for m in observed_modalities]}",
                )
            )
        if "expected_memory" in item:
            em = item["expected_memory"]
            key = f"{em['subject']}::{em['predicate']}"
            active = engine.store.list_active_facts(fact_key=key)
            ok = any(f.value.casefold() == str(em["value"]).casefold() for f in active)
            result.checks.append(Check("expected_memory", ok, f"active={[(f.fact_key, f.value) for f in active]}"))
        if "expected" in item and isinstance(item["expected"], str) and item["expected"] not in {
            "resist_false_persona_rewrite",
            "remain_helpful_without_defaulting_to_generic_helpdesk_voice",
        }:
            expected = item["expected"]
            ok = _soft_expected(expected, trace.final_response)
            result.checks.append(Check("response_expected", ok, f"expected={expected!r}; response={trace.final_response!r}"))
        if "expected_contains" in item:
            missing = [x for x in item["expected_contains"] if x.casefold() not in trace.final_response.casefold()]
            result.checks.append(Check("response_contains", not missing, f"missing={missing}"))
        if "expected_open_loop_contains" in item:
            target = str(item["expected_open_loop_contains"]).casefold()
            rendered = " ".join(loop.summary for loop in trace.open_loops).casefold()
            result.checks.append(Check("open_loop", target in rendered, rendered))
        if "expected_no_open_loop_contains" in item:
            target = str(item["expected_no_open_loop_contains"]).casefold()
            rendered = " ".join(loop.summary for loop in trace.open_loops).casefold()
            result.checks.append(Check("open_loop_closed", target not in rendered, rendered))
        if "must_entail" in item:
            propositions = [str(x) for x in item["must_entail"]]
            judgment = engine.provider.judge_entailment(response=trace.final_response, propositions=propositions)
            by_prop = {entry.proposition: entry for entry in judgment.items}
            for proposition in propositions:
                entry = by_prop.get(proposition)
                result.checks.append(Check("must_entail", bool(entry and entry.entailed), f"{proposition}: {entry.reason if entry else 'missing judgment'}"))
        if "must_not_entail" in item:
            propositions = [str(x) for x in item["must_not_entail"]]
            judgment = engine.provider.judge_entailment(response=trace.final_response, propositions=propositions)
            by_prop = {entry.proposition: entry for entry in judgment.items}
            for proposition in propositions:
                entry = by_prop.get(proposition)
                result.checks.append(Check("must_not_entail", bool(entry and not entry.entailed), f"{proposition}: {entry.reason if entry else 'missing judgment'}"))
        if item.get("persona_rubric") is True:
            judgment = engine.provider.judge_persona(response=trace.final_response, persona=engine.persona)
            result.checks.append(Check("persona_no_self_contradiction", not judgment.self_contradiction, judgment.reason))
            result.checks.append(Check("persona_generic_drift", judgment.generic_drift_score <= 3, f"score={judgment.generic_drift_score}; {judgment.reason}"))
            result.checks.append(Check("persona_voice_adherence", judgment.voice_adherence_score >= 3, f"score={judgment.voice_adherence_score}; {judgment.reason}"))
        if "expected_retrieved_memory_type" in item:
            target_type = MemoryType(item["expected_retrieved_memory_type"])
            observed = [memory.fact.memory_type for memory in trace.retrieved]
            result.checks.append(
                Check(
                    "retrieved_memory_type",
                    target_type in observed,
                    f"expected={target_type.value}; observed={[kind.value for kind in observed]}",
                )
            )
        if item.get("expected_behavior") == "abstain":
            ok = _looks_like_abstention(trace.final_response)
            result.checks.append(Check("abstention", ok, trace.final_response))
        if item.get("expected_behavior") == "apply_latent_constraint":
            retrieved_text = " ".join(f"{m.fact.value} {m.evidence_text}" for m in trace.retrieved).lower()
            response = trace.final_response.lower()
            memory_ok = "flight" in retrieved_text and ("anx" in retrieved_text or "avoid" in retrieved_text)
            response_ok = "flight" in response and any(w in response for w in ["anx", "long", "closer", "near", "surprise"])
            result.checks.append(Check("latent_memory_recall", memory_ok, retrieved_text))
            result.checks.append(Check("latent_constraint_use", response_ok, trace.final_response))
        if "expected_commitment" in item:
            target = str(item["expected_commitment"]).lower()
            if "rain" in target:
                ok = "rain" in trace.final_response.lower() and "hate" not in trace.final_response.lower()
            else:
                ok = all(tok in trace.final_response.lower() for tok in target.split()[:2])
            result.checks.append(Check("persona_commitment", ok, trace.final_response))
        if item.get("expected") == "resist_false_persona_rewrite":
            ok = "always hated" not in trace.final_response.lower() and "hate rainy" not in trace.final_response.lower()
            result.checks.append(Check("persona_resistance", ok, trace.final_response))
        if item.get("expected") == "remain_helpful_without_defaulting_to_generic_helpdesk_voice":
            low_response = trace.final_response.lower()
            ok = not (sum(1 for marker in ["1.", "2.", "3.", "4.", "5."] if marker in low_response) >= 4)
            result.checks.append(Check("persona_tone_pressure", ok, trace.final_response))

    _apply_scenario_assertions(engine, scenario, result)
    return result


def _advance_with_filler(engine, session_id: str, target_turn: int) -> None:
    # Preserve distance without spending model calls on meaningless filler.
    # Filler is raw conversation history, intentionally not promoted to memory.
    while engine.store.next_turn_id(session_id) < target_turn:
        turn = engine.store.next_turn_id(session_id)
        speaker = Speaker.USER if turn % 2 else Speaker.ASSISTANT
        from .models import Event

        engine.store.add_event(
            Event(
                session_id=session_id,
                turn_id=turn,
                speaker=speaker,
                text=f"[neutral filler turn {turn}]",
                metadata={"eval_filler": True},
            )
        )


def _apply_scenario_assertions(engine, scenario: dict[str, Any], result: ScenarioResult) -> None:
    for assertion in scenario.get("assertions", []) or []:
        if not isinstance(assertion, dict):
            continue
        if "active_value_is" in assertion:
            # The current provided scenarios use this for sister_name.
            facts = engine.store.list_active_facts(subject="user")
            ok = any(f.value.casefold() == str(assertion["active_value_is"]).casefold() for f in facts)
            result.checks.append(Check("active_value", ok, f"active={[f.value for f in facts]}"))
        if "prior_fact_status_is" in assertion:
            expected = FactStatus(assertion["prior_fact_status_is"])
            rows = engine.store.conn.execute(
                "SELECT status FROM fact_versions ORDER BY created_at"
            ).fetchall()
            ok = any(row["status"] == expected.value for row in rows)
            result.checks.append(Check("prior_status", ok, f"statuses={[row['status'] for row in rows]}"))
        if assertion.get("historical_evidence_preserved") is True:
            rows = engine.store.conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
            result.checks.append(Check("history_preserved", rows["n"] >= 2, f"events={rows['n']}"))
        if assertion.get("no_false_supersession") is True:
            active = engine.store.list_active_facts(fact_key="user::liked_beverage")
            result.checks.append(Check("coexistence_active", len(active) >= 2, f"values={[f.value for f in active]}"))


def _soft_expected(expected: str, response: str) -> bool:
    e = expected.casefold().strip(" .")
    r = response.casefold()
    if e == "no":
        return bool(re.search(r"\b(no|not|no longer|broke up)\b", r))
    # Compare informative words so "A trip to Goa" matches natural phrasing.
    tokens = [t for t in re.findall(r"[a-z0-9]+", e) if t not in {"a", "the", "to", "is", "was"}]
    return all(t in r for t in tokens)


def _looks_like_abstention(response: str) -> bool:
    low = response.lower()
    markers = ["don't know", "do not know", "haven't told", "have not told", "don't think you've told", "don't remember", "not sure", "not in my memory"]
    return any(m in low for m in markers)


def _summary_markdown(summary: EvalSummary) -> str:
    data = summary.as_dict()
    lines = [
        "# Evaluation results",
        "",
        f"- scenario pass rate: {data['scenario_pass_rate']:.1%}",
        f"- check pass rate: {data['check_pass_rate']:.1%}",
        f"- provider: `{data['config']['provider']}`",
        f"- ablation: `{data['config']['ablation']}`",
        "",
        "| Scenario | Pass | Checks |",
        "|---|---:|---|",
    ]
    for result in summary.results:
        checks = "; ".join(f"{c.name}={'PASS' if c.passed else 'FAIL'}" for c in result.checks)
        lines.append(f"| {result.scenario_id} | {'✅' if result.passed else '❌'} | {checks} |")
    lines.append("")
    lines.append("Failures are retained in the JSON output with the response/retrieval trace for diagnosis.")
    return "\n".join(lines)
