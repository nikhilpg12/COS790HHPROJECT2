from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


class ExperimentLogger:
    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(self, name: str, payload: Any) -> None:
        path = self.run_dir / name
        path.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _append_jsonl(self, name: str, payload: Any) -> None:
        path = self.run_dir / name
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=False) + "\n")

    def write_run(self, payload: Any) -> None:
        self._write_json("run.json", payload)

    def append_candidate(self, payload: Any) -> None:
        self._append_jsonl("candidates.jsonl", payload)

    def append_llm_call(self, payload: Any) -> None:
        self._append_jsonl("llm_calls.jsonl", payload)

    def write_generation_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        materialized = [dict(_jsonable(row)) for row in rows]
        path = self.run_dir / "generations.csv"
        if not materialized:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(materialized[0].keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(materialized)

    def write_best(self, payload: Any) -> None:
        self._write_json("best_heuristic.json", payload)

    def write_summary(self, payload: Any) -> None:
        self._write_json("summary.json", payload)
