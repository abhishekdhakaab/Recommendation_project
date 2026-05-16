import sys
import types

import pytest

from evaluation.mlflow_logging import flatten_metrics, log_experiment_result


def test_flatten_metrics_keeps_only_scalars() -> None:
    flattened = flatten_metrics(
        {
            "at_10": {"hit": 0.4, "ndcg": 0.2, "label": "skip"},
            "enabled": True,
            "notes": ["skip"],
        }
    )

    assert flattened == {"at_10.hit": 0.4, "at_10.ndcg": 0.2, "enabled": 1.0}


def test_log_experiment_result_uses_lazy_mlflow(monkeypatch, tmp_path) -> None:
    calls = {"params": None, "metrics": None, "artifacts": []}

    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    fake_mlflow = types.SimpleNamespace(
        set_tracking_uri=lambda uri: calls.setdefault("tracking_uri", uri),
        set_experiment=lambda name: calls.setdefault("experiment", name),
        start_run=lambda run_name: FakeRun(),
        log_params=lambda params: calls.__setitem__("params", params),
        log_metrics=lambda metrics: calls.__setitem__("metrics", metrics),
        log_artifact=lambda path: calls["artifacts"].append(("file", path)),
        log_artifacts=lambda path: calls["artifacts"].append(("dir", path)),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}")

    flattened = log_experiment_result(
        run_name="smoke",
        metrics={"at_10": {"hit": 0.5}},
        params={"top_items": 5000},
        artifacts=[artifact],
        tracking_uri="file:///tmp/mlruns",
    )

    assert flattened == {"at_10.hit": 0.5}
    assert calls["params"] == {"top_items": "5000"}
    assert calls["metrics"] == {"at_10.hit": 0.5}
    assert calls["artifacts"] == [("file", str(artifact))]


def test_log_experiment_result_raises_when_mlflow_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)

    with pytest.raises(ImportError, match="mlflow is required"):
        log_experiment_result(run_name="missing", metrics={"hit": 1.0})
