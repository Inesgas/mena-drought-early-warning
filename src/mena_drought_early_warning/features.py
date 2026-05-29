from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProjectConfig


def classify_drought_from_ndvi(anomaly: float) -> float:
    if pd.isna(anomaly):
        return np.nan
    if anomaly <= -0.14:
        return 3
    if anomaly <= -0.08:
        return 2
    if anomaly <= -0.04:
        return 1
    return 0


def _add_monthly_climatology_and_anomaly(
    df: pd.DataFrame,
    column: str,
    climatology_reference_end_date: str,
) -> pd.DataFrame:
    climatology_name = f"{column}_climatology"
    anomaly_name = f"{column}_anomaly"
    reference_end = pd.Timestamp(climatology_reference_end_date)
    reference_df = df[df["date"] <= reference_end]

    if reference_df.empty:
        raise ValueError(
            "No rows are available for climatology calculation through "
            f"{climatology_reference_end_date}."
        )

    climatology = (
        reference_df.groupby(["cell_id", "month"])[column]
        .mean()
        .rename(climatology_name)
        .reset_index()
    )

    df = df.merge(climatology, on=["cell_id", "month"], how="left", validate="many_to_one")
    df[anomaly_name] = df[column] - df[climatology_name]
    return df


def _add_grouped_rolling_mean(
    df: pd.DataFrame, source_column: str, window: int, new_column: str
) -> pd.DataFrame:
    df[new_column] = (
        df.groupby("cell_id")[source_column]
        .transform(lambda series: series.rolling(window, min_periods=1).mean())
    )
    return df


def add_forecast_features(
    df: pd.DataFrame,
    config: ProjectConfig,
    climatology_reference_end_date: str | None = None,
) -> pd.DataFrame:
    """Create forecast features without letting validation/test rows set normals.

    Monthly climatology is estimated through ``config.train_end_date`` by default
    so temporal backtests do not use future observations to define anomalies.
    Pass an explicit date to use a different historical baseline.
    """

    if climatology_reference_end_date is None:
        climatology_reference_end_date = config.train_end_date

    feature_df = df.dropna(
        subset=["ndvi", "rainfall", "lst_c", "pdsi", "vpd", "aet", "def"]
    ).copy()

    feature_df["date"] = pd.to_datetime(feature_df["date"])
    feature_df["month"] = feature_df["date"].dt.month
    feature_df["year"] = feature_df["date"].dt.year
    feature_df = feature_df.sort_values(["cell_id", "date"]).reset_index(drop=True)

    for base_column in ["ndvi", "rainfall", "lst_c"]:
        feature_df = _add_monthly_climatology_and_anomaly(
            feature_df,
            base_column,
            climatology_reference_end_date,
        )

    feature_df = _add_grouped_rolling_mean(feature_df, "ndvi_anomaly", 3, "ndvi_anom_3m")
    feature_df = _add_grouped_rolling_mean(feature_df, "ndvi_anomaly", 6, "ndvi_anom_6m")
    feature_df = _add_grouped_rolling_mean(feature_df, "rainfall", 3, "rainfall_3m")
    feature_df = _add_grouped_rolling_mean(feature_df, "rainfall", 6, "rainfall_6m")
    feature_df = _add_grouped_rolling_mean(feature_df, "lst_c", 3, "lst_3m")
    feature_df = _add_grouped_rolling_mean(feature_df, "pdsi", 3, "pdsi_3m")
    feature_df = _add_grouped_rolling_mean(feature_df, "vpd", 3, "vpd_3m")

    feature_df["drought_class"] = feature_df["ndvi_anomaly"].apply(classify_drought_from_ndvi)
    feature_df["drought_label"] = feature_df["drought_class"].map(config.class_names)

    return feature_df


def default_feature_columns() -> list[str]:
    return [
        "ndvi",
        "ndvi_anomaly",
        "ndvi_anom_3m",
        "ndvi_anom_6m",
        "rainfall",
        "rainfall_3m",
        "rainfall_6m",
        "rainfall_anomaly",
        "lst_c",
        "lst_3m",
        "lst_c_anomaly",
        "pdsi",
        "pdsi_3m",
        "vpd",
        "vpd_3m",
        "aet",
        "def",
        "month",
        "lat",
        "lon",
    ]


def build_forecast_dataset(
    feature_df: pd.DataFrame,
    horizon_months: int,
    feature_columns: list[str],
    config: ProjectConfig,
) -> tuple[pd.DataFrame, str]:
    target_column = f"drought_class_t_plus_{horizon_months}"
    target_label_column = f"drought_label_t_plus_{horizon_months}"

    model_df = feature_df.copy()
    model_df[target_column] = model_df.groupby("cell_id")["drought_class"].shift(-horizon_months)
    model_df[target_label_column] = model_df[target_column].map(config.class_names)

    required_columns = feature_columns + ["drought_class", target_column]
    model_df = model_df.dropna(subset=required_columns).copy()
    model_df[target_column] = model_df[target_column].astype(int)

    return model_df, target_column
