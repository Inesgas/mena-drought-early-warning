import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mena_drought_early_warning.sequences import build_sequence_dataset, split_sequence_metadata


class SequenceTests(unittest.TestCase):
    def test_build_sequence_dataset_returns_expected_windows_and_metadata(self):
        df = pd.DataFrame(
            {
                "cell_id": [1, 1, 1],
                "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
                "feature": [10.0, 20.0, 30.0],
                "target": [0, 1, 2],
            }
        )

        X, y, meta = build_sequence_dataset(
            df,
            feature_columns=["feature"],
            target_column="target",
            sequence_length=2,
        )

        self.assertEqual(X.shape, (2, 2, 1))
        self.assertEqual(y.tolist(), [1, 2])
        self.assertEqual(meta["sequence_start_date"].tolist(), list(pd.to_datetime(["2020-01-01", "2020-02-01"])))

    def test_split_sequence_metadata_returns_temporal_masks(self):
        meta = pd.DataFrame(
            {
                "date": pd.to_datetime(["2018-12-01", "2019-01-01", "2022-01-01"]),
            }
        )

        train_mask, valid_mask, test_mask = split_sequence_metadata(
            meta,
            train_end_date="2018-12-01",
            valid_end_date="2021-12-01",
        )

        self.assertEqual(train_mask.tolist(), [True, False, False])
        self.assertEqual(valid_mask.tolist(), [False, True, False])
        self.assertEqual(test_mask.tolist(), [False, False, True])


if __name__ == "__main__":
    unittest.main()
