import json

from impact_oil_project.pipeline import run_pipeline


def test_pipeline_runs_and_writes_metrics(tmp_path):
    output_file = tmp_path / "metrics.json"
    result = run_pipeline(output_path=str(output_file))

    assert result["rows_analyzed"] >= 36
    assert "var" in result and "random_forest" in result
    assert output_file.exists()

    written = json.loads(output_file.read_text(encoding="utf-8"))
    assert written["test_rows"] == result["test_rows"]
    assert written["var"]["inflation_mae"] >= 0
