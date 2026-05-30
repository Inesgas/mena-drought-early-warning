from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    start_date: str = "2001-01-01"
    end_date: str = "2024-12-31"
    train_end_date: str = "2018-12-01"
    valid_end_date: str = "2021-12-01"
    forecast_issue_date: str = "2024-09-01"

    bbox: tuple[float, float, float, float] = (-17.0, 12.0, 65.0, 42.0)
    grid_dx: float = 1.0
    grid_dy: float = 1.0
    reduce_scale: int = 5000
    forecast_horizons: tuple[int, ...] = (1, 3, 6)

    ndvi_dataset_id: str = "MODIS/061/MOD13A3"
    rainfall_dataset_id: str = "UCSB-CHG/CHIRPS/DAILY"
    lst_dataset_id: str = "MODIS/061/MOD11A2"
    terraclimate_dataset_id: str = "IDAHO_EPSCOR/TERRACLIMATE"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / "figures"

    @property
    def maps_dir(self) -> Path:
        return self.outputs_dir / "maps"

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    @property
    def tables_dir(self) -> Path:
        return self.outputs_dir / "tables"

    @property
    def class_names(self) -> dict[int, str]:
        return {
            0: "Normal / Wet",
            1: "Mild Drought",
            2: "Moderate Drought",
            3: "Severe Drought",
        }
