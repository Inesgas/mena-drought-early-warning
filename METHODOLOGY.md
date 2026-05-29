# Methodology

## Objective

This project forecasts drought severity across the MENA region at 1-month and 3-month horizons.

The workflow uses monthly satellite and climate variables, aggregates them to grid cells, builds lagged and anomaly-based predictors, and evaluates model performance on later time periods.

## Relationship to the Mapping Project

The companion project, [MENA Drought Risk Mapping](https://github.com/Inesgas/mena-drought-risk-mapping), maps current drought risk.

This project keeps the same regional focus and shifts the task from current-condition mapping to future-condition forecasting.

## Data

The workflow uses Earth Engine datasets with broad temporal coverage:

- MODIS NDVI for vegetation condition
- CHIRPS rainfall for precipitation
- MODIS land surface temperature for heat stress
- TerraClimate variables for water-balance conditions

The working period is `2001-01-01` to `2024-12-31`.

## Spatial Unit

The study area is represented with a regular grid over the MENA bounding box.

Each monthly image is reduced over each grid cell, producing one row per grid cell per month. The resulting table includes the cell location, date, vegetation, rainfall, temperature, and water-balance variables.

## Feature Engineering

The feature table includes:

- current monthly values
- calendar-month climatologies
- anomalies from climatology
- rolling 3-month and 6-month summaries
- drought class derived from NDVI anomaly

Climatologies are estimated from the training period by default. This keeps validation and test periods separate from the baseline used to compute anomalies.

## Forecast Targets

The target is the future drought class within the same grid cell:

- `drought_class_t_plus_1`
- `drought_class_t_plus_3`

Targets are created by sorting each grid-cell time series by date and shifting drought class forward by the forecast horizon.

## Validation

The split is temporal:

- training data through `2018-12-01`
- validation data through `2021-12-01`
- test data after `2021-12-01`

This preserves the order of time and evaluates the models on later months.

## Models

The main baselines are:

- persistence: use the current drought class as the future forecast
- Random Forest: learn from the engineered monthly predictors

The project also includes optional XGBoost support and LSTM sequence dataset preparation.

## Metrics

Model comparison uses:

- accuracy
- balanced accuracy
- macro F1
- classification reports
- confusion matrices

## Exports

The workflow writes:

- model metrics
- classification reports
- confusion matrices
- Random Forest feature importance
- forecast issue tables
- interactive forecast maps

These outputs are stored under `outputs/`.
