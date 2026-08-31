from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunConfig:
    model: str = "qwen3-coder:30b"
    population_size: int = 8
    generations: int = 2
    tournament_size: int = 4
    crossover_rate: float = 0.5
    mutation_rate: float = 0.5
    max_initial_depth: int = 4
    initial_batch_size: int = 4
    llm_retry_limit: int = 2
    llm_failure_policy: str = "abort"
    seed: int | None = 42
    temperature: float = 0.4
    results_dir: Path = Path("results")

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.generations <= 0:
            raise ValueError("generations must be positive")
        if self.tournament_size <= 0 or self.tournament_size > self.population_size:
            raise ValueError("tournament_size must be between 1 and population_size")
        if self.max_initial_depth < 1:
            raise ValueError("max_initial_depth must be at least 1")
        if self.initial_batch_size < 1:
            raise ValueError("initial_batch_size must be at least 1")
        if self.llm_retry_limit < 1:
            raise ValueError("llm_retry_limit must be at least 1")
        if self.llm_failure_policy != "abort":
            raise ValueError("llm_failure_policy must be 'abort' for the baseline")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.crossover_rate < 0 or self.mutation_rate < 0:
            raise ValueError("operator rates must be non-negative")
        if abs((self.crossover_rate + self.mutation_rate) - 1.0) > 1e-9:
            raise ValueError("crossover_rate and mutation_rate must sum to 1.0")


def development_config(**overrides: object) -> RunConfig:
    values: dict[str, object] = {
        "population_size": 8,
        "generations": 2,
        "tournament_size": 4,
        "crossover_rate": 0.5,
        "mutation_rate": 0.5,
        "max_initial_depth": 4,
        "initial_batch_size": 4,
        "llm_retry_limit": 2,
    }
    values.update(overrides)
    return RunConfig(**values)


def paper_config(**overrides: object) -> RunConfig:
    values: dict[str, object] = {
        "population_size": 500,
        "generations": 50,
        "tournament_size": 4,
        "crossover_rate": 0.5,
        "mutation_rate": 0.5,
        "max_initial_depth": 4,
        "initial_batch_size": 8,
        "llm_retry_limit": 2,
    }
    values.update(overrides)
    return RunConfig(**values)
