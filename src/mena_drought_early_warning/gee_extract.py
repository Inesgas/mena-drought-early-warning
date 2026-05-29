from __future__ import annotations

import pandas as pd

import ee

from .config import ProjectConfig
from .utils import month_starts


def initialize_earth_engine(authenticate: bool = False) -> None:
    if authenticate:
        ee.Authenticate()
    ee.Initialize()


def build_mena_bbox(config: ProjectConfig) -> ee.Geometry:
    return ee.Geometry.Rectangle(list(config.bbox))


def build_fishnet(config: ProjectConfig) -> ee.FeatureCollection:
    xmin, ymin, xmax, ymax = config.bbox
    features = []
    cell_id = 0
    x = xmin

    while x < xmax:
        y = ymin
        while y < ymax:
            x_max = min(x + config.grid_dx, xmax)
            y_max = min(y + config.grid_dy, ymax)
            geometry = ee.Geometry.Rectangle([x, y, x_max, y_max])
            feature = ee.Feature(
                geometry,
                {
                    "cell_id": cell_id,
                    "lon_min": x,
                    "lat_min": y,
                    "lon_max": x_max,
                    "lat_max": y_max,
                },
            )
            features.append(feature)
            cell_id += 1
            y += config.grid_dy
        x += config.grid_dx

    return ee.FeatureCollection(features)


def get_data_collections(config: ProjectConfig, bbox: ee.Geometry) -> dict[str, ee.ImageCollection]:
    return {
        "ndvi": (
            ee.ImageCollection(config.ndvi_dataset_id)
            .filterDate(config.start_date, config.end_date)
            .filterBounds(bbox)
            .select("NDVI")
        ),
        "rainfall": (
            ee.ImageCollection(config.rainfall_dataset_id)
            .filterDate(config.start_date, config.end_date)
            .filterBounds(bbox)
            .select("precipitation")
        ),
        "lst": (
            ee.ImageCollection(config.lst_dataset_id)
            .filterDate(config.start_date, config.end_date)
            .filterBounds(bbox)
            .select("LST_Day_1km")
        ),
        "terraclimate": (
            ee.ImageCollection(config.terraclimate_dataset_id)
            .filterDate(config.start_date, config.end_date)
            .filterBounds(bbox)
            .select(["pdsi", "vpd", "aet", "def"])
        ),
    }


def build_monthly_image(
    month_start: str,
    collections: dict[str, ee.ImageCollection],
    bbox: ee.Geometry,
) -> ee.Image:
    start = ee.Date(month_start)
    end = start.advance(1, "month")

    ndvi_img = (
        collections["ndvi"]
        .filterDate(start, end)
        .mean()
        .multiply(0.0001)
        .rename("ndvi")
    )

    rain_img = (
        collections["rainfall"]
        .filterDate(start, end)
        .sum()
        .rename("rainfall")
    )

    lst_img = (
        collections["lst"]
        .filterDate(start, end)
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .rename("lst_c")
    )

    terraclimate_img = (
        collections["terraclimate"]
        .filterDate(start, end)
        .mean()
        .select(["pdsi", "vpd", "aet", "def"], ["pdsi", "vpd", "aet", "def"])
    )

    terraclimate_scaled = ee.Image.cat(
        [
            terraclimate_img.select("pdsi").multiply(0.01).rename("pdsi"),
            terraclimate_img.select("vpd").multiply(0.01).rename("vpd"),
            terraclimate_img.select("aet").multiply(0.1).rename("aet"),
            terraclimate_img.select("def").multiply(0.1).rename("def"),
        ]
    )

    year_num = ee.Number.parse(start.format("Y"))
    month_num = ee.Number.parse(start.format("M"))

    return (
        ee.Image.cat([ndvi_img, rain_img, lst_img, terraclimate_scaled])
        .clip(bbox)
        .set(
            {
                "system:time_start": start.millis(),
                "date": start.format("YYYY-MM-dd"),
                "year": year_num,
                "month": month_num,
            }
        )
    )


def build_monthly_collection(config: ProjectConfig, bbox: ee.Geometry) -> ee.ImageCollection:
    collections = get_data_collections(config, bbox)
    months = month_starts(config.start_date, config.end_date)
    return ee.ImageCollection.fromImages(
        [build_monthly_image(month, collections, bbox) for month in months]
    )


def reduce_month_to_grid(
    image: ee.Image, grid: ee.FeatureCollection, scale: int
) -> ee.FeatureCollection:
    reduced = image.reduceRegions(
        collection=grid,
        reducer=ee.Reducer.mean(),
        scale=scale,
    )

    def add_metadata(feature: ee.Feature) -> ee.Feature:
        return feature.set(
            {
                "date": image.get("date"),
                "year": image.get("year"),
                "month": image.get("month"),
            }
        )

    return reduced.map(add_metadata)


def build_monthly_feature_collection(
    monthly_images: ee.ImageCollection,
    grid: ee.FeatureCollection,
    scale: int,
) -> ee.FeatureCollection:
    return ee.FeatureCollection(
        monthly_images.map(lambda image: reduce_month_to_grid(ee.Image(image), grid, scale)).flatten()
    )


def feature_collection_to_dataframe(feature_collection: ee.FeatureCollection) -> pd.DataFrame:
    records = feature_collection.getInfo()["features"]
    rows = []

    for record in records:
        props = record["properties"]
        rows.append(
            {
                "cell_id": props.get("cell_id"),
                "lon_min": props.get("lon_min"),
                "lat_min": props.get("lat_min"),
                "lon_max": props.get("lon_max"),
                "lat_max": props.get("lat_max"),
                "date": props.get("date"),
                "year": props.get("year"),
                "month": props.get("month"),
                "ndvi": props.get("ndvi"),
                "rainfall": props.get("rainfall"),
                "lst_c": props.get("lst_c"),
                "pdsi": props.get("pdsi"),
                "vpd": props.get("vpd"),
                "aet": props.get("aet"),
                "def": props.get("def"),
            }
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["cell_id", "date"]).reset_index(drop=True)
    df["lon"] = (df["lon_min"] + df["lon_max"]) / 2
    df["lat"] = (df["lat_min"] + df["lat_max"]) / 2

    return df


def extract_monthly_grid_dataframe(
    config: ProjectConfig,
    grid: ee.FeatureCollection,
    bbox: ee.Geometry,
) -> pd.DataFrame:
    collections = get_data_collections(config, bbox)
    rows = []

    for month in month_starts(config.start_date, config.end_date):
        image = build_monthly_image(month, collections, bbox)
        reduced_records = reduce_month_to_grid(image, grid, config.reduce_scale).getInfo()["features"]

        for record in reduced_records:
            props = record["properties"]
            rows.append(
                {
                    "cell_id": props.get("cell_id"),
                    "lon_min": props.get("lon_min"),
                    "lat_min": props.get("lat_min"),
                    "lon_max": props.get("lon_max"),
                    "lat_max": props.get("lat_max"),
                    "date": props.get("date"),
                    "year": props.get("year"),
                    "month": props.get("month"),
                    "ndvi": props.get("ndvi"),
                    "rainfall": props.get("rainfall"),
                    "lst_c": props.get("lst_c"),
                    "pdsi": props.get("pdsi"),
                    "vpd": props.get("vpd"),
                    "aet": props.get("aet"),
                    "def": props.get("def"),
                }
            )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["cell_id", "date"]).reset_index(drop=True)
    df["lon"] = (df["lon_min"] + df["lon_max"]) / 2
    df["lat"] = (df["lat_min"] + df["lat_max"]) / 2
    return df
