"""Optional MLflow logging helpers for SeqRec experiments."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def flatten_metrics(metrics: Mapping[str, Any], *, prefix: str = "") -> dict[str, float]:
    """Flatten nested metric dictionaries into MLflow-friendly scalar keys."""

    flattened: dict[str, float] = {}
    for key, value in metrics.items():
        metric_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_metrics(value, prefix=metric_key))
        elif isinstance(value, bool):
            flattened[metric_key] = float(value)
        elif isinstance(value, int | float):
            flattened[metric_key] = float(value)
    return flattened


def log_experiment_result(
    *,
    run_name: str,
    metrics: Mapping[str, Any],
    params: Mapping[str, Any] | None = None,
    artifacts: list[str | Path] | None = None,
    tracking_uri: str | None = None,
    experiment_name: str = "seqrec",
) -> dict[str, float]:
    """Log params, scalar metrics, and optional artifacts to MLflow.

    The import is intentionally lazy so tests and lightweight local workflows can
    use the rest of the evaluation package without starting MLflow.
    """

    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError("mlflow is required to log SeqRec experiments") from exc

    flattened = flatten_metrics(metrics)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params({key: str(value) for key, value in params.items()})
        if flattened:
            mlflow.log_metrics(flattened)
        for artifact in artifacts or []:
            artifact_path = Path(artifact)
            if artifact_path.is_dir():
                mlflow.log_artifacts(str(artifact_path))
            else:
                mlflow.log_artifact(str(artifact_path))
    return flattened
