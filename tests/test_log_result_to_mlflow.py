import json

import pytest

from evaluation.log_result_to_mlflow import load_result


def test_load_result_requires_object(tmp_path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(["not", "an", "object"]))

    with pytest.raises(ValueError, match="top level"):
        load_result(path)


def test_load_result_reads_json_object(tmp_path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"at_10": {"hit": 0.4}}))

    assert load_result(path) == {"at_10": {"hit": 0.4}}
