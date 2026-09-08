from __future__ import annotations

import argparse
import json

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run crude oil shock impact analysis for inflation and growth."
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional CSV path containing date, crude_oil_price, inflation, gdp_growth columns.",
    )
    parser.add_argument(
        "--output-path",
        default="output/analysis_metrics.json",
        help="Output path for computed metrics JSON.",
    )
    args = parser.parse_args()

    result = run_pipeline(data_path=args.data_path, output_path=args.output_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
