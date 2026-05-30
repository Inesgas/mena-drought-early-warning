from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mena_drought_early_warning.config import ProjectConfig
from mena_drought_early_warning.utils import ensure_project_directories


def drought_pressure(date: pd.Timestamp, lon: float, lat: float, cell_id: int) -> float:
    month_index = (date.year - 2001) * 12 + date.month - 1
    phase = (cell_id % 11) / 11 * 2 * np.pi
    cycle = max(0.0, np.sin((month_index / 8.5) + phase)) * 0.07

    summer = 1.0 if date.month in {6, 7, 8, 9} else 0.0
    heat_season = 0.02 * summer * (lat < 34)

    training_shock = 0.0
    if pd.Timestamp("2007-03-01") <= date <= pd.Timestamp("2008-11-01"):
        west_weight = np.clip((25 - lon) / 45, 0, 1)
        dry_belt_weight = np.clip((35 - abs(lat - 25)) / 20, 0, 1)
        training_shock = 0.055 * west_weight * dry_belt_weight
    if pd.Timestamp("2015-04-01") <= date <= pd.Timestamp("2016-10-01"):
        east_weight = np.clip((lon + 5) / 70, 0, 1)
        dry_belt_weight = np.clip((36 - abs(lat - 27)) / 20, 0, 1)
        training_shock = max(training_shock, 0.065 * east_weight * dry_belt_weight)

    issue_shock = 0.0
    if pd.Timestamp("2022-04-01") <= date <= pd.Timestamp("2024-10-01"):
        east_weight = np.clip((lon + 5) / 70, 0, 1)
        dry_belt_weight = np.clip((34 - abs(lat - 26)) / 18, 0, 1)
        pulse = 0.5 + 0.5 * np.sin((month_index - 255) / 5)
        issue_shock = 0.14 * east_weight * dry_belt_weight * pulse

    return cycle + heat_season + training_shock + issue_shock


def build_sample_monthly_grid(config: ProjectConfig) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    xmin, ymin, xmax, ymax = config.bbox
    dates = pd.date_range(config.start_date, config.end_date, freq="MS")
    lon_centers = np.linspace(xmin + 3.0, xmax - 3.0, 12)
    lat_centers = np.linspace(ymin + 2.0, ymax - 2.0, 6)

    rows = []
    cell_id = 0
    for lon in lon_centers:
        for lat in lat_centers:
            aridity = np.clip(0.45 + (32 - lat) * 0.018 + (lon - 10) * 0.004, 0.15, 0.9)
            base_ndvi = 0.52 - 0.26 * aridity + 0.03 * np.cos(np.radians(lon * 2))
            rain_base = 95 - 70 * aridity + 16 * np.cos(np.radians(lon + 10))

            for date in dates:
                month_angle = 2 * np.pi * (date.month - 1) / 12
                green_season = 0.045 * np.sin(month_angle - 0.6)
                wet_season = max(0.0, np.cos(month_angle - 2.5))
                dry = drought_pressure(date, lon, lat, cell_id)

                rainfall = max(
                    0.0,
                    rain_base * (0.32 + 0.85 * wet_season) * (1 - 3.1 * dry)
                    + rng.normal(0, 4.0),
                )
                lst_c = 18 + 16 * aridity + 7 * max(0.0, np.sin(month_angle - 1.1)) + 42 * dry
                lst_c += rng.normal(0, 0.9)
                ndvi = base_ndvi + green_season - dry + 0.00035 * rainfall + rng.normal(0, 0.012)
                ndvi = float(np.clip(ndvi, 0.05, 0.85))
                pdsi = 1.4 - 2.7 * aridity - 8.5 * dry + 0.01 * rainfall + rng.normal(0, 0.25)
                vpd = 0.55 + 1.65 * aridity + 6.8 * dry + 0.25 * max(0.0, np.sin(month_angle - 1.1))
                vpd += rng.normal(0, 0.08)
                aet = max(0.0, 38 * (1 - aridity) + 0.28 * rainfall - 75 * dry + rng.normal(0, 2.0))
                deficit = max(0.0, 18 + 78 * aridity + 95 * dry - 0.18 * rainfall + rng.normal(0, 2.5))

                rows.append(
                    {
                        "cell_id": cell_id,
                        "lon_min": lon - 0.5,
                        "lat_min": lat - 0.5,
                        "lon_max": lon + 0.5,
                        "lat_max": lat + 0.5,
                        "date": date.strftime("%Y-%m-%d"),
                        "year": date.year,
                        "month": date.month,
                        "ndvi": round(ndvi, 5),
                        "rainfall": round(float(rainfall), 3),
                        "lst_c": round(float(lst_c), 3),
                        "pdsi": round(float(pdsi), 3),
                        "vpd": round(float(vpd), 3),
                        "aet": round(float(aet), 3),
                        "def": round(float(deficit), 3),
                        "lon": round(float(lon), 4),
                        "lat": round(float(lat), 4),
                    }
                )

            cell_id += 1

    return pd.DataFrame(rows).sort_values(["cell_id", "date"]).reset_index(drop=True)


def main() -> None:
    config = ProjectConfig()
    ensure_project_directories(config)
    output_path = config.project_root / "data" / "processed" / "monthly_grid.csv"
    df = build_sample_monthly_grid(config)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df):,} rows to {output_path}")
    print(f"Cells: {df['cell_id'].nunique():,}")
    print(f"Period: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()
