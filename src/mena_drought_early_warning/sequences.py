from __future__ import annotations

import numpy as np
import pandas as pd


def build_sequence_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int = 6,
    group_column: str = "cell_id",
    date_column: str = "date",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ordered_df = df.sort_values([group_column, date_column]).reset_index(drop=True)
    sequences: list[np.ndarray] = []
    targets: list[int] = []
    metadata: list[dict] = []

    for cell_id, group in ordered_df.groupby(group_column):
        group = group.reset_index(drop=True)
        values = group[feature_columns].to_numpy()
        target_values = group[target_column].to_numpy()

        for end_idx in range(sequence_length - 1, len(group)):
            start_idx = end_idx - sequence_length + 1
            sequences.append(values[start_idx : end_idx + 1])
            targets.append(int(target_values[end_idx]))
            metadata.append(
                {
                    group_column: cell_id,
                    date_column: group.loc[end_idx, date_column],
                    "sequence_start_date": group.loc[start_idx, date_column],
                    "sequence_end_date": group.loc[end_idx, date_column],
                }
            )

    X = np.stack(sequences) if sequences else np.empty((0, sequence_length, len(feature_columns)))
    y = np.asarray(targets, dtype=int)
    meta_df = pd.DataFrame(metadata)
    return X, y, meta_df


def split_sequence_metadata(
    meta_df: pd.DataFrame,
    train_end_date: str,
    valid_end_date: str,
    date_column: str = "date",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = pd.to_datetime(meta_df[date_column])
    train_mask = dates <= pd.Timestamp(train_end_date)
    valid_mask = (dates > pd.Timestamp(train_end_date)) & (dates <= pd.Timestamp(valid_end_date))
    test_mask = dates > pd.Timestamp(valid_end_date)
    return train_mask.to_numpy(), valid_mask.to_numpy(), test_mask.to_numpy()
