"""Log an existing SeqRec JSON result file to MLflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.mlflow_logging import log_experiment_result


def load_result(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("result JSON must contain an object at the top level")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Log a SeqRec result JSON to MLflow.")
    parser.add_argument("result_json", help="Path to a result JSON file.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-name", default="seqrec")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--param", action="append", default=[], help="Extra param as key=value; may be repeated.")
    args = parser.parse_args()

    params = {"result_json": args.result_json}
    for raw in args.param:
        if "=" not in raw:
            raise ValueError(f"--param must be key=value, got {raw!r}")
        key, value = raw.split("=", 1)
        params[key] = value

    metrics = load_result(args.result_json)
    flattened = log_experiment_result(
        run_name=args.run_name,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        params=params,
        metrics=metrics,
        artifacts=[args.result_json],
    )
    print(json.dumps({"logged_metrics": flattened}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
