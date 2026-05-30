from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ProjectConfig


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_confusion_matrix(
    y_true,
    y_pred,
    class_names: dict[int, str],
    title: str,
    save_path: Path,
) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay

    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=sorted(class_names),
        display_labels=[class_names[index] for index in sorted(class_names)],
        cmap="Blues",
        ax=ax,
        xticks_rotation=30,
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_plot(
    feature_importance: pd.Series,
    title: str,
    save_path: Path,
) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    feature_importance.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metric_comparison(
    metric_df: pd.DataFrame,
    title: str,
    save_path: Path,
) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    metric_df.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_risk_distribution(
    df: pd.DataFrame,
    risk_column: str,
    title: str,
    save_path: Path,
) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    df[risk_column].plot(kind="hist", bins=12, range=(0, 1), ax=ax, color="#C73E1D")
    ax.set_title(title)
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Grid cells")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_forecast_table(df: pd.DataFrame, save_path: Path, columns: list[str]) -> None:
    df[columns].sort_values(["date", "cell_id"]).to_csv(save_path, index=False)


def create_forecast_map(
    df: pd.DataFrame,
    predicted_label_column: str,
    save_path: Path,
    config: ProjectConfig,
    title_prefix: str,
) -> object:
    import folium

    color_map = {
        "Normal / Wet": "#2E86AB",
        "Mild Drought": "#F18F01",
        "Moderate Drought": "#E76F51",
        "Severe Drought": "#C73E1D",
    }

    map_object = folium.Map(location=[28, 10], zoom_start=4, tiles="CartoDB positron")

    for _, row in df.iterrows():
        label = row[predicted_label_column]
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=color_map.get(label, "#999999"),
            fill=True,
            fill_opacity=0.8,
            popup=(
                f"{title_prefix}<br>"
                f"Cell: {int(row['cell_id'])}<br>"
                f"Issue date: {row['date'].date()}<br>"
                f"Observed current class: {row['drought_label']}<br>"
                f"Forecast class: {label}"
            ),
        ).add_to(map_object)

    map_object.save(str(save_path))
    return map_object


def create_risk_map(
    df: pd.DataFrame,
    risk_column: str,
    uncertainty_column: str,
    save_path: Path,
    title_prefix: str,
) -> object:
    import folium

    def risk_color(value: float) -> str:
        if value >= 0.75:
            return "#C73E1D"
        if value >= 0.50:
            return "#E76F51"
        if value >= 0.25:
            return "#F18F01"
        return "#2E86AB"

    map_object = folium.Map(location=[28, 10], zoom_start=4, tiles="CartoDB positron")

    for _, row in df.iterrows():
        risk = float(row[risk_column])
        uncertainty = float(row[uncertainty_column])
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4 + 5 * risk,
            color=risk_color(risk),
            fill=True,
            fill_opacity=0.8,
            popup=(
                f"{title_prefix}<br>"
                f"Cell: {int(row['cell_id'])}<br>"
                f"Issue date: {row['date'].date()}<br>"
                f"Moderate-or-worse risk: {risk:.1%}<br>"
                f"Uncertainty: {uncertainty:.1%}<br>"
                f"Forecast class: {row['rf_prediction_label']}"
            ),
        ).add_to(map_object)

    map_object.save(str(save_path))
    return map_object
