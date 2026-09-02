from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import Settings
from .evaluation import run_scenarios
from .factory import build_engine
from .models import RetrievalMode
from .store import MemoryStore

app = typer.Typer(help="Companion Memory Core research CLI", no_args_is_help=True)
console = Console()


@app.command()
def init(db: Path = Path("data/companion.sqlite3")) -> None:
    """Initialize the persistent SQLite memory store."""
    store = MemoryStore(db)
    store.close()
    console.print(f"Initialized memory store: [bold]{db}[/bold]")


@app.command()
def state(db: Path = Path("data/companion.sqlite3"), subject: str = "user") -> None:
    """Show active living state for a subject."""
    store = MemoryStore(db)
    facts = store.list_active_facts(subject=subject)
    table = Table(title=f"Active state: {subject}")
    for column in ("entity", "slot/predicate", "value", "modality", "type", "fact_id"):
        table.add_column(column)
    for fact in facts:
        table.add_row(
            fact.entity_mention,
            fact.slot or fact.predicate_text,
            fact.value_text,
            fact.modality.value,
            fact.memory_type.value,
            fact.fact_id,
        )
    console.print(table)
    store.close()


@app.command("inspect")
def inspect_state(
    fact_key: str,
    db: Path = Path("data/companion.sqlite3"),
) -> None:
    """Inspect all historical versions of a fact key."""
    store = MemoryStore(db)
    history = store.list_fact_history(fact_key.strip().lower())
    table = Table(title=f"Fact history: {fact_key}")
    for column in ("fact_id", "value", "modality", "status", "valid_from", "valid_to", "recorded_at", "retired_at", "supersedes"):
        table.add_column(column)
    for fact in history:
        table.add_row(
            fact.fact_id,
            fact.value_text,
            fact.modality.value,
            fact.status.value,
            str(fact.valid_from or ""),
            str(fact.valid_to or ""),
            str(fact.recorded_at or ""),
            str(fact.retired_at or ""),
            str(fact.supersedes_fact_id or ""),
        )
    console.print(table)
    store.close()


@app.command()
def entities(db: Path = Path("data/companion.sqlite3")) -> None:
    """Show the durable entity registry and aliases."""
    store = MemoryStore(db)
    table = Table(title="Entity registry")
    for column in ("entity_id", "canonical_name", "type", "relation_to_user", "aliases"):
        table.add_column(column)
    for entity in store.list_entities():
        table.add_row(
            entity.entity_id,
            entity.canonical_name,
            entity.entity_type.value,
            str(entity.relation_to_user or ""),
            ", ".join(entity.aliases),
        )
    console.print(table)
    store.close()


@app.command()
def traces(
    db: Path = Path("data/companion.sqlite3"),
    session: str | None = None,
    limit: int = 10,
) -> None:
    """Inspect persisted extraction/retrieval/verification traces."""
    store = MemoryStore(db)
    for trace in store.list_traces(session_id=session, limit=limit):
        console.rule(f"trace {trace.trace_id}")
        console.print(f"user_event={trace.user_event_id} assistant_event={trace.assistant_event_id}")
        if trace.extracted_claims:
            console.print("[bold]claims[/bold]")
            for claim, decision in zip(trace.extracted_claims, trace.decisions):
                console.print(
                    f"  {claim.fact_key}={claim.value!r} [{claim.memory_type.value}] -> {decision.action.value}"
                )
        if trace.open_loops:
            console.print("[bold]open loops[/bold]")
            for loop in trace.open_loops:
                console.print(f"  {loop.kind.value}: {loop.summary} ({loop.reason})")
        if trace.retrieved:
            console.print("[bold]retrieved[/bold]")
            for item in trace.retrieved:
                console.print(
                    f"  {item.fact.fact_key}={item.fact.value!r} {item.fact.status.value} "
                    f"rrf={item.rrf_score:.5f} final={item.final_score:.5f} ({item.retrieval_reason})"
                )
        if trace.consistency:
            console.print(f"consistency={trace.consistency.consistent}")
    store.close()


@app.command("loops")
def show_loops(
    session: str = typer.Option("demo"),
    db: Path = typer.Option(Path("data/companion.sqlite3")),
    query: str = typer.Option("hi", help="Current user text used for relevance."),
) -> None:
    """Show unresolved plan/goal/promise follow-up candidates."""
    from .open_loops import OpenLoopManager

    store = MemoryStore(db)
    manager = OpenLoopManager(store)
    loops = manager.candidates(
        current_session_id=session,
        user_text=query,
        first_turn_in_session=store.next_turn_id(session) == 1,
    )
    table = Table(title=f"Open loops for {session}")
    for column in ("kind", "summary", "due", "priority", "reason"):
        table.add_column(column)
    for loop in loops:
        table.add_row(loop.kind.value, loop.summary, str(loop.due_at or ""), f"{loop.priority:.2f}", loop.reason)
    console.print(table)
    store.close()


@app.command("consolidate")
def consolidate_session(
    session: str = typer.Option(..., help="Session id to consolidate."),
    provider: str = typer.Option("openai"),
    db: Path = typer.Option(Path("data/companion.sqlite3")),
    persona: Path = typer.Option(Path("persona.yaml")),
) -> None:
    """Create an idempotent shared-history session episode and hedged inferences."""
    settings = replace(Settings.from_env(), db_path=db, persona_path=persona, provider=provider)
    engine = build_engine(settings)
    try:
        result = engine.consolidate_session(session)
        if result.skipped:
            console.print(f"Session {session!r} was empty or already consolidated.")
            return
        console.print(f"summary_fact_id={result.summary_fact_id}")
        console.print(f"inference_fact_ids={list(result.inference_fact_ids)}")
    finally:
        engine.store.close()


@app.command()
def chat(
    session: str = typer.Option("demo", help="Conversation session id."),
    provider: str = typer.Option("openai", help="openai or heuristic"),
    db: Path = typer.Option(Path("data/companion.sqlite3")),
    persona: Path = typer.Option(Path("persona.yaml")),
    trace: bool = typer.Option(False, "--trace/--no-trace"),
    consistency: bool = typer.Option(True, "--consistency/--no-consistency"),
    retrieval_mode: RetrievalMode = typer.Option(RetrievalMode.HYBRID),
    auto_consolidate: bool = typer.Option(False, "--auto-consolidate/--no-auto-consolidate"),
) -> None:
    """Run the persistent command-line companion loop."""
    settings = replace(
        Settings.from_env(),
        db_path=db,
        persona_path=persona,
        provider=provider,
        consistency_check=consistency,
        retrieval_mode=retrieval_mode,
        auto_consolidate_previous_sessions=auto_consolidate,
    )
    try:
        engine = build_engine(settings)
    except Exception as exc:
        console.print(f"[red]Could not start companion:[/red] {exc}")
        if provider == "openai":
            console.print("Set OPENAI_API_KEY and install `pip install -e '.[all]'`.")
        raise typer.Exit(code=2)

    console.print(
        f"[bold]{engine.persona.name}[/bold] — session={session}, provider={provider}. "
        "Commands: /state, /loops, /consolidate, /quit"
    )
    try:
        while True:
            try:
                text = console.input("[bold cyan]you>[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not text:
                continue
            if text in {"/quit", "/exit"}:
                break
            if text == "/state":
                facts = engine.store.list_active_facts(subject="user")
                for fact in facts:
                    console.print(f"{fact.fact_key} = {fact.value!r}")
                continue
            if text == "/loops":
                loops = engine.open_loop_manager.candidates(
                    current_session_id=session, user_text="hi", first_turn_in_session=True
                )
                for loop in loops:
                    console.print(f"{loop.kind.value}: {loop.summary} ({loop.reason})")
                if not loops:
                    console.print("No open loops.")
                continue
            if text == "/consolidate":
                result = engine.consolidate_session(session)
                console.print(f"consolidated summary_fact_id={result.summary_fact_id} inferences={list(result.inference_fact_ids)}")
                continue
            turn = engine.process_turn(session_id=session, user_text=text)
            console.print(f"[bold green]{engine.persona.name}>[/bold green] {turn.final_response}")
            if trace:
                for claim, decision in zip(turn.extracted_claims, turn.decisions):
                    console.print(
                        f"  [dim]memory {claim.fact_key}={claim.value!r} -> {decision.action.value}[/dim]"
                    )
                for loop in turn.open_loops:
                    console.print(f"  [dim]open-loop {loop.kind.value}: {loop.summary} ({loop.reason})[/dim]")
                for item in turn.retrieved:
                    console.print(
                        f"  [dim]retrieve {item.fact.fact_key}={item.fact.value!r} "
                        f"status={item.fact.status.value} modality={item.fact.modality.value} rrf={item.rrf_score:.5f}[/dim]"
                    )
                if turn.consistency:
                    console.print(f"  [dim]firewall consistent={turn.consistency.consistent}[/dim]")
    finally:
        engine.store.close()


@app.command("eval")
def evaluate(
    provider: str = typer.Option("heuristic"),
    scenarios: Path = typer.Option(Path("eval/scenarios")),
    output: Path = typer.Option(Path("eval/results")),
    ablation: str = typer.Option("full"),
    preserve_turn_distance: bool = typer.Option(False, "--preserve-turn-distance/--compact-turns"),
) -> None:
    """Run the adversarial scenario suite and emit JSON/Markdown results."""
    settings = replace(Settings.from_env(), provider=provider)
    summary = run_scenarios(
        scenarios,
        base_settings=settings,
        ablation=ablation,
        preserve_turn_distance=preserve_turn_distance,
        output_dir=output,
    )
    data = summary.as_dict()
    console.print(f"scenario_pass_rate={data['scenario_pass_rate']:.1%}")
    console.print(f"check_pass_rate={data['check_pass_rate']:.1%}")


if __name__ == "__main__":
    app()
