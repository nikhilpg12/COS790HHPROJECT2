import json

from llm_gp_hh.experiments.logging import ExperimentLogger
from llm_gp_hh.experiments.reference_results import PUBLISHED_TORONTO_AHH


def test_reference_value_is_tagged_as_published_not_reproduced():
    ref = PUBLISHED_TORONTO_AHH["car-f-92"]
    assert ref.hcv == 0
    assert ref.scv == 4.32
    assert ref.source == "Pillay & Özcan (2019), Table 18"
    assert ref.reproduced is False


def test_all_thirteen_toronto_reference_instances_are_present():
    assert len(PUBLISHED_TORONTO_AHH) == 13
    assert PUBLISHED_TORONTO_AHH["sta-f-83"].scv == 157.64
    assert PUBLISHED_TORONTO_AHH["uta-s-92"].scv == 3.35


def test_logger_writes_required_files(tmp_path):
    logger = ExperimentLogger(tmp_path / "run")
    logger.write_run({"seed": 123, "model": "qwen3-coder:30b"})
    logger.append_candidate({"id": "g0-i0", "fitness": 4.2})
    logger.append_llm_call({"operation": "initial", "latency_seconds": 1.0})
    logger.write_generation_rows([{"generation": 0, "best_fitness": 4.2}])
    logger.write_best({"id": "g0-i0"})
    logger.write_summary({"best_fitness": 4.2})
    for name in [
        "run.json",
        "candidates.jsonl",
        "llm_calls.jsonl",
        "generations.csv",
        "best_heuristic.json",
        "summary.json",
    ]:
        assert (tmp_path / "run" / name).exists()
    assert json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))["seed"] == 123
