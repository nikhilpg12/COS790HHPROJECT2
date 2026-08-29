from __future__ import annotations

from pathlib import Path
import tempfile

from llm_gp_hh.experiments.run import build_parser, run_experiment


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="llm_gp_hh_smoke_") as tmp:
        root = Path(tmp)
        crs = root / "smoke.crs"
        stu = root / "smoke.stu"
        crs.write_text("1 2\n2 2\n3 2\n4 1\n", encoding="utf-8")
        stu.write_text("1 2\n1 3\n2 4\n3\n", encoding="utf-8")
        args = build_parser().parse_args(
            [
                "--crs", str(crs),
                "--stu", str(stu),
                "--periods", "4",
                "--profile", "dev",
                "--population-size", "4",
                "--generations", "2",
                "--tournament-size", "2",
                "--initial-batch-size", "4",
                "--results-dir", "results/live_smoke",
            ]
        )
        run_dir = run_experiment(args)
        print(f"Live smoke test complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
