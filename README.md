# MENA Drought Early Warning

This project forecasts drought severity 1, 3, and 6 months ahead across the MENA region using Earth Engine satellite and climate data.

The upgraded version works as a small drought-forecast benchmark: it keeps fixed temporal splits, adds spatial holdout validation, exports class probabilities, and creates map-ready drought-risk outputs.

It extends the drought-risk mapping project:

- [MENA Drought Risk Mapping](https://github.com/Inesgas/mena-drought-risk-mapping)

The mapping project describes current drought conditions. This project uses the same regional framing and moves the workflow into early warning.

## Project Question

Given satellite, rainfall, temperature, and climatic water-balance information available at month `t`, can vegetation-stress-based drought severity be forecast at month `t+1`, `t+3`, and `t+6`?

## Data Sources

The workflow uses these Google Earth Engine datasets:

- MODIS NDVI: [`MODIS/061/MOD13A3`](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13A3)
- CHIRPS rainfall: [`UCSB-CHG/CHIRPS/DAILY`](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY)
- MODIS land surface temperature: [`MODIS/061/MOD11A2`](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2)
- TerraClimate water-balance variables: [`IDAHO_EPSCOR/TERRACLIMATE`](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE)

The analysis window is `2001-01-01` to `2024-12-31`.

## Forecast Design

The pipeline builds monthly grid-cell tables and trains forecasting models using information available at the issue month.

Targets:

- drought class at `t+1`
- drought class at `t+3`
- drought class at `t+6`

Models:

- persistence baseline
- Random Forest classifier
- optional XGBoost classifier
- LSTM sequence dataset preparation

Evaluation uses temporal train, validation, and test periods rather than random splits. The benchmark run also checks spatial transfer by holding out west, central, and east longitude bands.

## Repository Structure

```text
mena-drought-early-warning/
├── README.md
├── METHODOLOGY.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── data/
├── outputs/
│   ├── figures/
│   ├── maps/
│   ├── reports/
│   └── tables/
├── assets/
├── notebooks/
├── scripts/
├── src/
└── tests/
```

## Workflow

1. Define the MENA study area and build a regular analysis grid.
2. Pull monthly remote-sensing and climate data from Earth Engine.
3. Reduce monthly values to grid cells.
4. Build drought features, anomalies, and future targets.
5. Split the data by time into train, validation, and test periods.
6. Compare persistence and Random Forest baselines.
7. Run spatial holdout validation by region.
8. Export forecast tables, reports, figures, class probabilities, and interactive maps.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/project_setup.py
python scripts/environment_check.py
python -m unittest discover -s tests
```

Authenticate Earth Engine before running the extraction steps.

## Notebook

Open:

```text
notebooks/mena_drought_early_warning_forecasting.ipynb
```

The notebook runs the full workflow from Earth Engine extraction through feature engineering, model evaluation, and output export.

## Scripted Baseline Run

After exporting a monthly grid table to `data/processed/monthly_grid.csv`, the baseline workflow can be run without the notebook.

For a local reproducible run, build the sample monthly grid first:

```bash
python scripts/build_sample_monthly_grid.py
```

The sample grid is for checking the full workflow locally. Replace it with an Earth Engine export before treating model scores as real drought-forecast performance.

Then run the baseline workflow:

```bash
python scripts/run_baseline_pipeline.py --input data/processed/monthly_grid.csv
```

Useful options:

- `--horizons 1 3 6`
- `--selected-horizon 1`
- `--issue-date 2024-09-01`
- `--include-xgboost`
- `--no-map`

## Outputs

Generated outputs are written under `outputs/`:

- `outputs/tables/forecast_model_metrics.csv`
- `outputs/tables/spatial_validation_metrics.csv`
- `outputs/tables/forecast_issue_table_*.csv`
- `outputs/reports/*classification_report*.txt`
- `outputs/reports/baseline_run_summary.md`
- `outputs/figures/*.png`
- `outputs/maps/*.html`

The short finished project report is:

- `PROJECT_REPORT.md`

## Method Notes

- Monthly climatologies are estimated from the training period, so validation and test anomalies do not use future data.
- The drought class is a vegetation-stress proxy derived from NDVI anomaly.
- Forecast targets are created by shifting drought class forward within each grid-cell time series.
- Persistence predicts the future drought class from the current drought class.
- Random Forest uses the engineered hydroclimate and location features.
- Random Forest class probabilities are exported as drought-risk and uncertainty fields.
- Spatial validation holds out one longitude band at a time to test transfer to unseen regions.
