from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from companion_memory.config import Settings
from companion_memory.evaluation import run_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Companion Memory Core evaluation scenarios")
    parser.add_argument("--provider", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument(
        "--ablation",
        choices=["full", "lexical_only", "structured_only", "semantic_only", "no_temporal", "no_firewall", "no_memory", "full_history", "vector_bag", "oracle"],
        default="full",
    )
    parser.add_argument("--scenarios", type=Path, default=Path("eval/scenarios"))
    parser.add_argument("--output", type=Path, default=Path("eval/results"))
    parser.add_argument("--preserve-turn-distance", action="store_true")
    args = parser.parse_args()

    settings = replace(Settings.from_env(), provider=args.provider)
    summary = run_scenarios(
        args.scenarios,
        base_settings=settings,
        ablation=args.ablation,
        preserve_turn_distance=args.preserve_turn_distance,
        output_dir=args.output,
    )
    data = summary.as_dict()
    print(f"scenario_pass_rate={data['scenario_pass_rate']:.1%}")
    print(f"check_pass_rate={data['check_pass_rate']:.1%}")
    for scenario in data["scenarios"]:
        print(f"{'PASS' if scenario['passed'] else 'FAIL'} {scenario['scenario_id']}")


if __name__ == "__main__":
    main()
