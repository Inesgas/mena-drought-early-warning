import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mena_drought_early_warning.pipeline import (
    REQUIRED_MONTHLY_COLUMNS,
    select_forecast_issue,
    validate_monthly_grid_table,
)


class PipelineHelperTests(unittest.TestCase):
    def test_validate_monthly_grid_table_reports_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "lst_c"):
            validate_monthly_grid_table(
                pd.DataFrame({column: [] for column in REQUIRED_MONTHLY_COLUMNS if column != "lst_c"})
            )

    def test_select_forecast_issue_falls_back_to_latest_available_month(self):
        forecast_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2022-01-01", "2022-02-01"]),
                "cell_id": [1, 1],
            }
        )

        issue_date, selected = select_forecast_issue(forecast_df, "2024-09-01")

        self.assertEqual(issue_date, pd.Timestamp("2022-02-01"))
        self.assertEqual(selected["cell_id"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
