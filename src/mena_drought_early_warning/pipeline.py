from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .features import add_forecast_features, build_forecast_dataset, default_feature_columns
from .modeling import (
    SPATIAL_REGION_COLUMN,
    assign_spatial_regions,
    classification_report_text,
    evaluate_predictions,
    feature_importance_series,
    fit_random_forest,
    fit_xgboost_classifier,
    predict_persistence,
    predict_class_probabilities,
    spatial_holdout_split,
    temporal_split,
    xgboost_available,
)
from .reporting import (
    create_forecast_map,
    create_risk_map,
    export_forecast_table,
    save_confusion_matrix,
    save_feature_importance_plot,
    save_metric_comparison,
    save_risk_distribution,
)
from .utils import ensure_project_directories


REQUIRED_MONTHLY_COLUMNS = [
    "cell_id",
    "date",
    "ndvi",
    "rainfall",
    "lst_c",
    "pdsi",
    "vpd",
    "aet",
    "def",
    "lat",
    "lon",
]


@dataclass
class BaselinePipelineResult:
    feature_table: pd.DataFrame
    metrics: pd.DataFrame
    spatial_metrics: pd.DataFrame
    test_predictions: dict[int, pd.DataFrame]
    forecast_table_path: Path | None
    forecast_map_path: Path | None
    forecast_risk_map_path: Path | None
    spatial_metrics_path: Path | None
    summary_path: Path | None


def validate_monthly_grid_table(df: pd.DataFrame) -> None:
    missing_columns = sorted(set(REQUIRED_MONTHLY_COLUMNS) - set(df.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Monthly grid table is missing required columns: {missing_text}")


def load_monthly_grid_table(path: Path | str) -> pd.DataFrame:
    table_path = Path(path)
    df = pd.read_csv(table_path)
    validate_monthly_grid_table(df)
    df["date"] = pd.to_datetime(df["date"])
    return df


def select_forecast_issue(
    forecast_df: pd.DataFrame,
    requested_issue_date: str | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    if forecast_df.empty:
        raise ValueError("Cannot select a forecast issue from an empty forecast table.")

    issue_date = pd.Timestamp(requested_issue_date)
    selected_issue_df = forecast_df[forecast_df["date"] == issue_date].copy()

    if selected_issue_df.empty:
        issue_date = pd.Timestamp(forecast_df["date"].max())
        selected_issue_df = forecast_df[forecast_df["date"] == issue_date].copy()

    return issue_date, selected_issue_df


def write_classification_report(
    y_true,
    y_pred,
    config: ProjectConfig,
    title: str,
    save_path: Path,
) -> None:
    labels = sorted(config.class_names)
    target_names = [config.class_names[index] for index in labels]
    report = classification_report_text(y_true, y_pred, labels, target_names)
    save_path.write_text(f"{title}\n\n{report}\n", encoding="utf-8")


def write_run_summary(
    metrics_df: pd.DataFrame,
    spatial_metrics_df: pd.DataFrame,
    config: ProjectConfig,
    save_path: Path,
    include_xgboost: bool,
) -> None:
    lines = [
        "# Baseline Forecast Run",
        "",
        f"- Train period ends: `{config.train_end_date}`",
        f"- Validation period ends: `{config.valid_end_date}`",
        f"- Horizons: `{', '.join(str(h) for h in sorted(metrics_df['horizon'].unique()))}`",
        f"- Optional XGBoost requested: `{'yes' if include_xgboost else 'no'}`",
        "",
        "## Best Model by Horizon",
        "",
    ]

    if metrics_df.empty:
        lines.append("No metrics were produced.")
    else:
        best_rows = metrics_df.loc[metrics_df.groupby("horizon")["macro_f1"].idxmax()]
        for _, row in best_rows.sort_values("horizon").iterrows():
            lines.append(
                "- "
                f"t+{int(row['horizon'])}: `{row['model']}` "
                f"(macro F1 `{row['macro_f1']:.3f}`, "
                f"balanced accuracy `{row['balanced_accuracy']:.3f}`)"
        )

    lines.extend(
        [
            "",
            "## Spatial Holdout Check",
            "",
        ]
    )
    if spatial_metrics_df.empty:
        lines.append("No spatial holdout metrics were produced.")
    else:
        rf_spatial = spatial_metrics_df[spatial_metrics_df["model"] == "Random Forest"]
        for horizon, group in rf_spatial.groupby("horizon"):
            mean_f1 = group["macro_f1"].mean()
            lines.append(
                f"- t+{int(horizon)} Random Forest mean macro F1 across held-out regions: "
                f"`{mean_f1:.3f}`"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Metrics are computed on the held-out test period after the validation cutoff.",
            "- Monthly climatologies are estimated from the training period by default.",
            "- Persistence is the minimum benchmark to beat: it predicts future drought class from the issue-month class.",
            "",
        ]
    )
    save_path.write_text("\n".join(lines), encoding="utf-8")


def run_spatial_holdout_validation(
    feature_df: pd.DataFrame,
    config: ProjectConfig,
    horizons: list[int],
    feature_columns: list[str],
) -> pd.DataFrame:
    spatial_results: list[dict[str, float | int | str]] = []
    regions = sorted(feature_df[SPATIAL_REGION_COLUMN].dropna().unique())

    for horizon in horizons:
        model_df, target_column = build_forecast_dataset(
            feature_df,
            horizon,
            feature_columns,
            config,
        )

        for region in regions:
            split = spatial_holdout_split(
                model_df,
                holdout_region=region,
                train_end_date=config.train_end_date,
                valid_end_date=config.valid_end_date,
            )
            if split.train.empty or split.test.empty:
                continue

            persistence_pred = predict_persistence(split.test)
            persistence_metrics = evaluate_predictions(split.test[target_column], persistence_pred)
            spatial_results.append(
                {
                    "horizon": horizon,
                    "holdout_region": region,
                    "model": "Persistence",
                    "train_rows": len(split.train),
                    "test_rows": len(split.test),
                    **persistence_metrics,
                }
            )

            rf_model = fit_random_forest(split.train, feature_columns, target_column)
            rf_test_pred = rf_model.predict(split.test[feature_columns])
            rf_metrics = evaluate_predictions(split.test[target_column], rf_test_pred)
            spatial_results.append(
                {
                    "horizon": horizon,
                    "holdout_region": region,
                    "model": "Random Forest",
                    "train_rows": len(split.train),
                    "test_rows": len(split.test),
                    **rf_metrics,
                }
            )

    return pd.DataFrame(spatial_results)


def run_baseline_pipeline(
    raw_df: pd.DataFrame,
    config: ProjectConfig,
    horizons: list[int] | tuple[int, ...] | None = None,
    include_xgboost: bool = False,
    selected_horizon: int | None = None,
    issue_date: str | pd.Timestamp | None = None,
    export_map: bool = True,
) -> BaselinePipelineResult:
    validate_monthly_grid_table(raw_df)
    ensure_project_directories(config)

    horizons = list(horizons or config.forecast_horizons)
    if not horizons:
        raise ValueError("At least one forecast horizon is required.")

    selected_horizon = selected_horizon or horizons[0]
    if selected_horizon not in horizons:
        raise ValueError("Selected horizon must be included in the horizons being evaluated.")

    issue_date = issue_date or config.forecast_issue_date
    feature_columns = default_feature_columns()
    feature_df = assign_spatial_regions(add_forecast_features(raw_df, config))
    results: list[dict[str, float | int | str]] = []
    test_predictions: dict[int, pd.DataFrame] = {}

    for horizon in horizons:
        model_df, target_column = build_forecast_dataset(
            feature_df,
            horizon,
            feature_columns,
            config,
        )
        target_label_column = f"drought_label_t_plus_{horizon}"
        split = temporal_split(model_df, config.train_end_date, config.valid_end_date)

        if split.train.empty or split.test.empty:
            raise ValueError(
                f"Horizon t+{horizon} does not have enough rows after temporal splitting."
            )

        persistence_pred = predict_persistence(split.test)
        persistence_metrics = evaluate_predictions(split.test[target_column], persistence_pred)
        results.append({"horizon": horizon, "model": "Persistence", **persistence_metrics})

        save_confusion_matrix(
            split.test[target_column],
            persistence_pred,
            config.class_names,
            f"Persistence Confusion Matrix - Horizon t+{horizon}",
            config.figures_dir / f"persistence_confusion_matrix_h{horizon}.png",
        )
        write_classification_report(
            split.test[target_column],
            persistence_pred,
            config,
            f"Persistence Classification Report - Horizon t+{horizon}",
            config.reports_dir / f"persistence_classification_report_h{horizon}.txt",
        )

        rf_model = fit_random_forest(split.train, feature_columns, target_column)
        rf_test_pred = rf_model.predict(split.test[feature_columns])
        rf_probability_df = predict_class_probabilities(
            rf_model,
            split.test[feature_columns],
            config.class_names,
        )
        rf_metrics = evaluate_predictions(split.test[target_column], rf_test_pred)
        results.append({"horizon": horizon, "model": "Random Forest", **rf_metrics})

        save_confusion_matrix(
            split.test[target_column],
            rf_test_pred,
            config.class_names,
            f"Random Forest Confusion Matrix - Horizon t+{horizon}",
            config.figures_dir / f"rf_confusion_matrix_h{horizon}.png",
        )
        save_feature_importance_plot(
            feature_importance_series(rf_model, feature_columns),
            f"Random Forest Feature Importance - Horizon t+{horizon}",
            config.figures_dir / f"rf_feature_importance_h{horizon}.png",
        )
        write_classification_report(
            split.test[target_column],
            rf_test_pred,
            config,
            f"Random Forest Classification Report - Horizon t+{horizon}",
            config.reports_dir / f"rf_classification_report_h{horizon}.txt",
        )

        horizon_test_df = split.test.copy()
        horizon_test_df["persistence_prediction"] = persistence_pred
        horizon_test_df["persistence_prediction_label"] = horizon_test_df[
            "persistence_prediction"
        ].map(config.class_names)
        horizon_test_df["rf_prediction"] = rf_test_pred
        horizon_test_df["rf_prediction_label"] = horizon_test_df["rf_prediction"].map(
            config.class_names
        )
        horizon_test_df = pd.concat([horizon_test_df, rf_probability_df], axis=1)

        if include_xgboost and xgboost_available():
            xgb_model = fit_xgboost_classifier(
                split.train,
                split.valid,
                feature_columns,
                target_column,
            )
            xgb_test_pred = xgb_model.predict(split.test[feature_columns])
            xgb_metrics = evaluate_predictions(split.test[target_column], xgb_test_pred)
            results.append({"horizon": horizon, "model": "XGBoost", **xgb_metrics})
            horizon_test_df["xgb_prediction"] = xgb_test_pred
            horizon_test_df["xgb_prediction_label"] = horizon_test_df[
                "xgb_prediction"
            ].map(config.class_names)
            write_classification_report(
                split.test[target_column],
                xgb_test_pred,
                config,
                f"XGBoost Classification Report - Horizon t+{horizon}",
                config.reports_dir / f"xgb_classification_report_h{horizon}.txt",
            )

        horizon_test_df["observed_future_label"] = horizon_test_df[target_label_column]
        test_predictions[horizon] = horizon_test_df

    metrics_df = pd.DataFrame(results)
    metrics_path = config.tables_dir / "forecast_model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    spatial_metrics_df = run_spatial_holdout_validation(
        feature_df,
        config,
        horizons,
        feature_columns,
    )
    spatial_metrics_path = config.tables_dir / "spatial_validation_metrics.csv"
    spatial_metrics_df.to_csv(spatial_metrics_path, index=False)

    for horizon in horizons:
        metric_subset = metrics_df[metrics_df["horizon"] == horizon].set_index("model")[
            ["accuracy", "balanced_accuracy", "macro_f1"]
        ]
        save_metric_comparison(
            metric_subset,
            f"Forecast Metric Comparison - Horizon t+{horizon}",
            config.figures_dir / f"forecast_metric_comparison_h{horizon}.png",
        )

    selected_issue_date, selected_issue_df = select_forecast_issue(
        test_predictions[selected_horizon],
        issue_date,
    )
    forecast_table_path = (
        config.tables_dir
        / f"forecast_issue_table_h{selected_horizon}_{selected_issue_date.strftime('%Y_%m')}.csv"
    )
    export_columns = [
        "cell_id",
        "date",
        "lat",
        "lon",
        "ndvi",
        "rainfall",
        "lst_c",
        "pdsi",
        "vpd",
        SPATIAL_REGION_COLUMN,
        "drought_label",
        "observed_future_label",
        "rf_prediction_label",
        "persistence_prediction_label",
        "rf_probability_normal_wet",
        "rf_probability_mild_drought",
        "rf_probability_moderate_drought",
        "rf_probability_severe_drought",
        "rf_confidence",
        "rf_uncertainty",
        "rf_drought_risk",
        "rf_moderate_plus_risk",
        "rf_severe_risk",
    ]
    if "xgb_prediction_label" in selected_issue_df.columns:
        export_columns.append("xgb_prediction_label")

    export_forecast_table(selected_issue_df, forecast_table_path, export_columns)

    forecast_map_path = None
    forecast_risk_map_path = None
    if export_map:
        forecast_map_path = (
            config.maps_dir
            / f"forecast_map_h{selected_horizon}_{selected_issue_date.strftime('%Y_%m')}.html"
        )
        create_forecast_map(
            selected_issue_df,
            predicted_label_column="rf_prediction_label",
            save_path=forecast_map_path,
            config=config,
            title_prefix=f"Random Forest Forecast t+{selected_horizon}",
        )
        forecast_risk_map_path = (
            config.maps_dir
            / f"forecast_risk_map_h{selected_horizon}_{selected_issue_date.strftime('%Y_%m')}.html"
        )
        create_risk_map(
            selected_issue_df,
            risk_column="rf_moderate_plus_risk",
            uncertainty_column="rf_uncertainty",
            save_path=forecast_risk_map_path,
            title_prefix=f"Random Forest Risk Forecast t+{selected_horizon}",
        )
        save_risk_distribution(
            selected_issue_df,
            risk_column="rf_moderate_plus_risk",
            title=f"Moderate-or-worse Risk Distribution - Horizon t+{selected_horizon}",
            save_path=config.figures_dir / f"rf_risk_distribution_h{selected_horizon}.png",
        )

    summary_path = config.reports_dir / "baseline_run_summary.md"
    write_run_summary(metrics_df, spatial_metrics_df, config, summary_path, include_xgboost)

    return BaselinePipelineResult(
        feature_table=feature_df,
        metrics=metrics_df,
        spatial_metrics=spatial_metrics_df,
        test_predictions=test_predictions,
        forecast_table_path=forecast_table_path,
        forecast_map_path=forecast_map_path,
        forecast_risk_map_path=forecast_risk_map_path,
        spatial_metrics_path=spatial_metrics_path,
        summary_path=summary_path,
    )
