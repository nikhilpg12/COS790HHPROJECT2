from llm_gp_hh.config import RunConfig, development_config, paper_config
from llm_gp_hh.rng import make_rng, resolve_seed


def test_paper_profile_matches_reported_ahh_parameters():
    cfg = paper_config()
    assert cfg.population_size == 500
    assert cfg.generations == 50
    assert cfg.tournament_size == 4
    assert cfg.crossover_rate == 0.50
    assert cfg.mutation_rate == 0.50
    assert cfg.max_initial_depth == 4


def test_rates_must_sum_to_one():
    try:
        RunConfig(crossover_rate=0.8, mutation_rate=0.3)
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
    else:
        raise AssertionError("invalid rates should fail")


def test_explicit_seed_is_reproducible():
    seed = resolve_seed(12345)
    assert seed == 12345
    assert [make_rng(seed).random() for _ in range(3)] == [make_rng(seed).random() for _ in range(3)]


def test_development_profile_is_small():
    cfg = development_config()
    assert cfg.population_size <= 12
    assert cfg.generations <= 5


def test_baseline_failure_policy_is_explicit_abort():
    assert RunConfig().llm_failure_policy == "abort"
    try:
        RunConfig(llm_failure_policy="random-tree")
    except ValueError as exc:
        assert "llm_failure_policy" in str(exc)
    else:
        raise AssertionError("unsupported fallback policy should fail")
