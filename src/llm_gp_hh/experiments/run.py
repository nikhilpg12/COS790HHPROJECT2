from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import platform
from pathlib import Path
import statistics
import sys
import time

from llm_gp_hh.config import RunConfig, development_config, paper_config
from llm_gp_hh.gp.evolution import EvolutionEngine, EvolutionResult, GenerationState
from llm_gp_hh.gp.individual import EvaluatedIndividual
from llm_gp_hh.gp.tree import node_count, tree_depth, tree_to_prefix
from llm_gp_hh.llm.ollama_client import OllamaClient
from llm_gp_hh.llm.operators import QwenTreeOperators
from llm_gp_hh.rng import resolve_seed
from llm_gp_hh.toronto.parser import load_toronto

from .logging import ExperimentLogger
from .reference_results import PUBLISHED_TORONTO_AHH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the LLM-assisted vanilla-GP Toronto hyper-heuristic baseline."
    )
    parser.add_argument("--crs", type=Path, required=True, help="Toronto .crs file")
    parser.add_argument("--stu", type=Path, required=True, help="Toronto .stu file")
    parser.add_argument("--periods", type=int, required=True, help="Number of timetable periods")
    parser.add_argument("--profile", choices=("dev", "paper"), default="dev")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", default="qwen3-coder:30b")
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--tournament-size", type=int)
    parser.add_argument("--crossover-rate", type=float)
    parser.add_argument("--mutation-rate", type=float)
    parser.add_argument("--max-initial-depth", type=int)
    parser.add_argument("--initial-batch-size", type=int)
    parser.add_argument("--retry-limit", type=int)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser


def _config_from_args(args: argparse.Namespace, seed: int) -> RunConfig:
    base = development_config() if args.profile == "dev" else paper_config()
    changes: dict[str, object] = {
        "seed": seed,
        "model": args.model,
        "results_dir": args.results_dir,
    }
    optional = {
        "population_size": args.population_size,
        "generations": args.generations,
        "tournament_size": args.tournament_size,
        "crossover_rate": args.crossover_rate,
        "mutation_rate": args.mutation_rate,
        "max_initial_depth": args.max_initial_depth,
        "initial_batch_size": args.initial_batch_size,
        "llm_retry_limit": args.retry_limit,
    }
    changes.update({key: value for key, value in optional.items() if value is not None})
    return replace(base, **changes)


def candidate_record(item: EvaluatedIndividual) -> dict[str, object]:
    return {
        "id": item.id,
        "generation": item.generation,
        "operation": item.operation,
        "parent_ids": list(item.parent_ids),
        "seed": item.seed,
        "tree": tree_to_prefix(item.tree),
        "tree_depth": tree_depth(item.tree),
        "node_count": node_count(item.tree),
        "hcv": item.hcv,
        "scv": item.scv,
        "fitness": item.fitness,
        "evaluation_seconds": item.evaluation_seconds,
    }


def generation_record(state: GenerationState, result: EvolutionResult) -> dict[str, object]:
    population = state.population
    best = min(population, key=lambda item: (item.fitness, item.hcv, item.scv, item.id))
    invalid_calls = sum(
        1 for call in result.llm_calls if call.generation == state.index and not call.valid
    )
    valid_calls = sum(
        1 for call in result.llm_calls if call.generation == state.index and call.valid
    )
    return {
        "generation": state.index,
        "population_size": len(population),
        "best_id": best.id,
        "best_fitness": best.fitness,
        "best_hcv": best.hcv,
        "best_scv": best.scv,
        "mean_fitness": statistics.fmean(item.fitness for item in population),
        "mean_hcv": statistics.fmean(item.hcv for item in population),
        "mean_scv": statistics.fmean(item.scv for item in population),
        "crossover_calls": state.operator_counts["crossover"],
        "mutation_calls": state.operator_counts["mutation"],
        "valid_llm_calls": valid_calls,
        "invalid_llm_calls": invalid_calls,
    }


def _model_names_from_ollama_response(response: object) -> set[str]:
    if isinstance(response, dict):
        models = response.get("models", [])
    else:
        models = getattr(response, "models", [])
    names: set[str] = set()
    for model in models:
        if isinstance(model, dict):
            name = model.get("model") or model.get("name")
        else:
            name = getattr(model, "model", None) or getattr(model, "name", None)
        if name:
            names.add(str(name))
    return names


def require_ollama_model(model: str) -> None:
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("Python package 'ollama' is not installed. Run: pip install -e .") from exc
    try:
        names = _model_names_from_ollama_response(ollama.list())
    except Exception as exc:
        raise RuntimeError("Could not contact local Ollama. Start Ollama and try again.") from exc
    if model not in names:
        raise RuntimeError(f"Ollama model {model!r} is not installed. Run: ollama pull {model}")


def _safe_instance_key(name: str) -> str:
    key = name.lower().strip()
    for suffix in (".crs", ".stu"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key.replace("_", "-")


def persist_result(
    *,
    logger: ExperimentLogger,
    result: EvolutionResult,
    config: RunConfig,
    instance_name: str,
    periods: int,
    seed: int,
    started_at: datetime,
    finished_at: datetime,
    wall_seconds: float,
) -> None:
    logger.write_run(
        {
            "benchmark": "Toronto examination timetabling",
            "instance": instance_name,
            "periods": periods,
            "seed": seed,
            "model": config.model,
            "config": asdict(config),
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "wall_seconds": wall_seconds,
            "python": sys.version,
            "platform": platform.platform(),
        }
    )
    for state in result.generations:
        for item in state.population:
            logger.append_candidate(candidate_record(item))
    for call in result.llm_calls:
        logger.append_llm_call(asdict(call))
    logger.write_generation_rows(generation_record(state, result) for state in result.generations)
    logger.write_best(candidate_record(result.best))

    reference = PUBLISHED_TORONTO_AHH.get(_safe_instance_key(instance_name))
    total_llm_seconds = sum(call.latency_seconds for call in result.llm_calls)
    summary: dict[str, object] = {
        "best": candidate_record(result.best),
        "total_candidates_evaluated": sum(len(state.population) for state in result.generations),
        "total_llm_calls": len(result.llm_calls),
        "invalid_llm_calls": sum(1 for call in result.llm_calls if not call.valid),
        "total_llm_seconds": total_llm_seconds,
        "wall_seconds": wall_seconds,
        "published_reference": asdict(reference) if reference is not None else None,
        "published_reference_is_local_reimplementation": False,
    }
    if reference is not None and result.best.hcv == 0 and reference.hcv == 0:
        summary["soft_cost_difference_vs_published_ahh"] = result.best.scv - reference.scv
    logger.write_summary(summary)


def run_experiment(args: argparse.Namespace) -> Path:
    seed = resolve_seed(args.seed)
    config = _config_from_args(args, seed)
    require_ollama_model(config.model)
    instance = load_toronto(args.crs, args.stu, periods=args.periods)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.results_dir / f"{instance.name}-{timestamp}-seed{seed}"
    logger = ExperimentLogger(run_dir)
    operators = QwenTreeOperators(
        client=OllamaClient(model=config.model),
        retry_limit=config.llm_retry_limit,
    )

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    result = EvolutionEngine().run(instance, operators, config, seed=seed)
    wall_seconds = time.perf_counter() - started
    finished_at = datetime.now(timezone.utc)
    persist_result(
        logger=logger,
        result=result,
        config=config,
        instance_name=instance.name,
        periods=instance.periods,
        seed=seed,
        started_at=started_at,
        finished_at=finished_at,
        wall_seconds=wall_seconds,
    )
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_dir = run_experiment(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
