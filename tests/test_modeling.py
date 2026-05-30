import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mena_drought_early_warning.modeling import (
    _build_label_maps,
    assign_spatial_regions,
    predict_class_probabilities,
    spatial_holdout_split,
    temporal_split,
)


class DummyProbabilityModel:
    classes_ = [0, 2]

    def predict_proba(self, X):
        return [[0.7, 0.3] for _ in range(len(X))]


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

    def test_assign_spatial_regions_uses_longitude_bands(self):
        df = pd.DataFrame({"lon": [-10.0, -5.0, 0.0, 5.0, 10.0]})

        result = assign_spatial_regions(df)

        self.assertEqual(
            result["spatial_region"].tolist(),
            ["west", "west", "central", "east", "east"],
        )

    def test_spatial_holdout_split_keeps_region_for_later_test(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2018-12-01", "2019-01-01", "2022-01-01"]),
                "spatial_region": ["west", "east", "east"],
                "value": [1, 2, 3],
            }
        )

        split = spatial_holdout_split(df, "east", "2018-12-01", "2021-12-01")

        self.assertEqual(split.train["value"].tolist(), [1])
        self.assertEqual(split.valid["value"].tolist(), [])
        self.assertEqual(split.test["value"].tolist(), [3])

    def test_predict_class_probabilities_fills_missing_classes_and_risk(self):
        X = pd.DataFrame({"feature": [1, 2]})
        class_names = {
            0: "Normal / Wet",
            1: "Mild Drought",
            2: "Moderate Drought",
            3: "Severe Drought",
        }

        result = predict_class_probabilities(DummyProbabilityModel(), X, class_names)

        self.assertEqual(result["rf_probability_mild_drought"].tolist(), [0.0, 0.0])
        self.assertEqual(result["rf_probability_moderate_drought"].tolist(), [0.3, 0.3])
        self.assertEqual(result["rf_moderate_plus_risk"].tolist(), [0.3, 0.3])
        self.assertEqual(result["rf_confidence"].tolist(), [0.7, 0.7])


if __name__ == "__main__":
    unittest.main()
