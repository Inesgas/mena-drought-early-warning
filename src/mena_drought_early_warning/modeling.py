from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


@dataclass
class EncodedXGBoostClassifier:
    model: object
    index_to_label: dict[int, int]

    def predict(self, X) -> np.ndarray:
        encoded_predictions = self.model.predict(X)
        return np.asarray(
            [self.index_to_label[int(prediction)] for prediction in encoded_predictions],
            dtype=int,
        )


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:
        return False


def _build_label_maps(target: pd.Series) -> tuple[dict[int, int], dict[int, int]]:
    labels = sorted(target.dropna().astype(int).unique())
    if not labels:
        raise ValueError("XGBoost needs at least one target label.")
    if len(labels) == 1:
        raise ValueError("XGBoost needs at least two target classes.")

    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    return label_to_index, index_to_label


def temporal_split(
    df: pd.DataFrame,
    train_end_date: str,
    valid_end_date: str,
    date_column: str = "date",
) -> TemporalSplit:
    date_series = pd.to_datetime(df[date_column])
    train_end = pd.Timestamp(train_end_date)
    valid_end = pd.Timestamp(valid_end_date)

    train_df = df[date_series <= train_end].copy()
    valid_df = df[(date_series > train_end) & (date_series <= valid_end)].copy()
    test_df = df[date_series > valid_end].copy()

    return TemporalSplit(train=train_df, valid=valid_df, test=test_df)


def fit_random_forest(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    random_state: int = 42,
) -> object:
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_df[feature_columns], train_df[target_column])
    return model


def fit_xgboost_classifier(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    random_state: int = 42,
):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise ImportError(
            "xgboost is not installed. Install it with `pip install xgboost` to run this benchmark."
        ) from exc

    label_to_index, index_to_label = _build_label_maps(train_df[target_column])
    train_target = train_df[target_column].astype(int).map(label_to_index)
    valid_target = valid_df[target_column].astype(int)
    valid_mask = valid_target.isin(label_to_index.keys())

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(label_to_index),
        random_state=random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    fit_kwargs = {"verbose": False}
    if valid_mask.any():
        fit_kwargs["eval_set"] = [
            (
                valid_df.loc[valid_mask, feature_columns],
                valid_target.loc[valid_mask].map(label_to_index),
            )
        ]

    model.fit(train_df[feature_columns], train_target, **fit_kwargs)
    return EncodedXGBoostClassifier(model=model, index_to_label=index_to_label)


def predict_persistence(test_df: pd.DataFrame) -> np.ndarray:
    return test_df["drought_class"].astype(int).to_numpy()


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def classification_report_text(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: list[int],
    target_names: list[str],
) -> str:
    from sklearn.metrics import classification_report

    return classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)


def feature_importance_series(model, feature_columns: list[str]) -> pd.Series:
    return pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
