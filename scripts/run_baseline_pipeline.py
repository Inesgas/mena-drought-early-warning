from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mena_drought_early_warning.config import ProjectConfig
from mena_drought_early_warning.pipeline import load_monthly_grid_table, run_baseline_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MENA drought early-warning baseline workflow from a local monthly grid CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "monthly_grid.csv",
        help="Path to a monthly grid CSV exported from Earth Engine.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Forecast horizons in months. Defaults to ProjectConfig.forecast_horizons.",
    )
    parser.add_argument(
        "--selected-horizon",
        type=int,
        default=None,
        help="Horizon used for the exported forecast issue table and map.",
    )
    parser.add_argument(
        "--issue-date",
        default=None,
        help="Issue month to export, formatted as YYYY-MM-DD. Falls back to the latest test month if unavailable.",
    )
    parser.add_argument(
        "--include-xgboost",
        action="store_true",
        help="Run XGBoost as an optional benchmark when the package is installed.",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Skip Folium map export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {args.input}. "
            "Create it from the notebook or Earth Engine extraction step first."
        )

    raw_df = load_monthly_grid_table(args.input)
    result = run_baseline_pipeline(
        raw_df,
        config,
        horizons=args.horizons,
        include_xgboost=args.include_xgboost,
        selected_horizon=args.selected_horizon,
        issue_date=args.issue_date,
        export_map=not args.no_map,
    )

    print("Baseline pipeline complete.")
    print(f"Metrics: {config.tables_dir / 'forecast_model_metrics.csv'}")
    print(f"Spatial validation metrics: {result.spatial_metrics_path}")
    print(f"Forecast issue table: {result.forecast_table_path}")
    if result.forecast_map_path is not None:
        print(f"Forecast map: {result.forecast_map_path}")
    if result.forecast_risk_map_path is not None:
        print(f"Forecast risk map: {result.forecast_risk_map_path}")
    print(f"Run summary: {result.summary_path}")


if __name__ == "__main__":
    main()
