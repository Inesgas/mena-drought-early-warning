import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mena_drought_early_warning.modeling import _build_label_maps, temporal_split


class ModelingTests(unittest.TestCase):
    def test_temporal_split_respects_cutoff_dates(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2018-12-01", "2019-01-01", "2022-01-01"]),
                "value": [1, 2, 3],
            }
        )

        split = temporal_split(df, train_end_date="2018-12-01", valid_end_date="2021-12-01")

        self.assertEqual(split.train["value"].tolist(), [1])
        self.assertEqual(split.valid["value"].tolist(), [2])
        self.assertEqual(split.test["value"].tolist(), [3])

    def test_xgboost_label_maps_allow_missing_intermediate_class(self):
        train_labels = pd.Series([0, 1, 3])

        label_to_index, index_to_label = _build_label_maps(train_labels)

        self.assertEqual(label_to_index, {0: 0, 1: 1, 3: 2})
        self.assertEqual(index_to_label, {0: 0, 1: 1, 2: 3})


if __name__ == "__main__":
    unittest.main()
