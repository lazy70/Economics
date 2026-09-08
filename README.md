# Impact of Crude Oil Price Shocks on Inflation and Economic Growth

Deployment-ready Python project for analyzing how crude oil price shocks affect inflation and GDP growth with a combined **time-series econometrics (VAR)** and **machine-learning (Random Forest)** workflow.

## Project structure

- `/src/impact_oil_project/pipeline.py` – end-to-end data preparation, VAR modeling, ML modeling, and evaluation
- `/src/impact_oil_project/cli.py` – CLI entry point
- `/tests/test_pipeline.py` – focused pipeline smoke test
- `/.github/workflows/ci.yml` – GitHub Actions CI for tests

## Input data format

Provide CSV data with these columns:

- `date`
- `crude_oil_price`
- `inflation`
- `gdp_growth`

If no dataset is provided, the pipeline runs with generated sample monthly macroeconomic data.

## Local run

```bash
pip install -e .
impact-oil-analysis --output-path output/analysis_metrics.json
```

## Run tests

```bash
pytest
```

## GitHub deployment readiness

This repository includes:

- installable Python package (`pyproject.toml`)
- repeatable CLI execution
- automated CI test workflow on push and pull requests
- reproducible outputs saved as JSON artifacts
