import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mena_drought_early_warning.config import ProjectConfig
from mena_drought_early_warning.features import (
    add_forecast_features,
    build_forecast_dataset,
    default_feature_columns,
)


class FeatureEngineeringTests(unittest.TestCase):
    def test_anomalies_use_training_period_climatology_by_default(self):
        config = ProjectConfig(train_end_date="2018-12-01")
        df = pd.DataFrame(
            {
                "cell_id": [1, 1],
                "date": ["2018-01-01", "2019-01-01"],
                "ndvi": [0.50, 0.10],
                "rainfall": [10.0, 30.0],
                "lst_c": [20.0, 30.0],
                "pdsi": [1.0, 1.0],
                "vpd": [2.0, 2.0],
                "aet": [3.0, 3.0],
                "def": [4.0, 4.0],
                "lat": [25.0, 25.0],
                "lon": [35.0, 35.0],
            }
        )

        features = add_forecast_features(df, config)
        jan_2019 = features.loc[features["date"] == pd.Timestamp("2019-01-01")].iloc[0]

        self.assertAlmostEqual(jan_2019["ndvi_climatology"], 0.50)
        self.assertAlmostEqual(jan_2019["ndvi_anomaly"], -0.40)
        self.assertAlmostEqual(jan_2019["rainfall_climatology"], 10.0)
        self.assertAlmostEqual(jan_2019["rainfall_anomaly"], 20.0)

    def test_forecast_target_shift_stays_within_each_cell(self):
        config = ProjectConfig()
        feature_df = pd.DataFrame(
            {
                "cell_id": [1, 1, 2, 2],
                "date": pd.to_datetime(
                    ["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"]
                ),
                "drought_class": [0, 3, 2, 1],
                "drought_label": [
                    "Normal / Wet",
                    "Severe Drought",
                    "Moderate Drought",
                    "Mild Drought",
                ],
                **{column: 1.0 for column in default_feature_columns()},
            }
        )

        model_df, target_column = build_forecast_dataset(
            feature_df,
            horizon_months=1,
            feature_columns=default_feature_columns(),
            config=config,
        )

        cell_1_target = model_df.loc[model_df["cell_id"] == 1, target_column].iloc[0]
        cell_2_target = model_df.loc[model_df["cell_id"] == 2, target_column].iloc[0]

        self.assertEqual(cell_1_target, 3)
        self.assertEqual(cell_2_target, 1)


if __name__ == "__main__":
    unittest.main()
