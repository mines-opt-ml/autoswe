from functools import reduce
from typing import Tuple
import pandas as pd

def melt_dynamic(path: str, var_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df_long = df.melt(id_vars="Date", var_name="Station", value_name=var_name)
    df_long["Station"] = df_long["Station"].str.replace("_", " ").str.lower()
    return df_long


def preprocess(cfg: object) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start_date = f"{cfg.train_start_year}-01-01"
    end_date = f"{cfg.test_end_year}-12-31"

    swe = pd.read_csv(cfg.swe_path)
    swe["Date"] = pd.to_datetime(swe["Date"])
    swe = swe[
        (pd.to_datetime(swe['Date']) >= start_date) &
        (pd.to_datetime(swe['Date']) <= end_date)
    ]
    swe_long = swe.melt(id_vars="Date", var_name="Station", value_name="SWE")
    swe_long["Station"] = swe_long["Station"].str.replace("_", " ").str.lower()
    
    meta = pd.read_csv(cfg.meta_path)
    meta["Station_clean"] = meta["Station Name"].str.lower()

    dynamic_inputs = [
        (cfg.max_temp_path, "Tmax"),
        (cfg.min_temp_path, "Tmin"),
        (cfg.precip_path, "Precip"),
        (cfg.obs_temp_path, "Tobs"),
        (cfg.tb19_path, "TB_19"),
        (cfg.tb37_path, "TB_37"),
        (cfg.tbdiff_path, "TB_diff"),
    ]

    dynamic_dfs = []
    for path, name in dynamic_inputs:
        df = melt_dynamic(path, name)
        df = df[
            (pd.to_datetime(df['Date']) >= start_date) &
            (pd.to_datetime(df['Date']) <= end_date)
        ]
        dynamic_dfs.append(df)

    dynamic_merged = reduce(lambda left, right: pd.merge(left, right, on=["Date", "Station"]), dynamic_dfs)

    merged = pd.merge(swe_long, dynamic_merged, on=["Date", "Station"], how="inner")
    merged = pd.merge(merged, meta, left_on="Station", right_on="Station_clean", how="inner")
    dynamic_forcing_and_swe = merged.dropna()
    
    snotel_attributes = meta[["Station Name", "Elevation_x", "Slope_tif1_x", "Aspect_tif_x", "Latitude_x", "Longitude_x"]].copy()
    snotel_attributes = snotel_attributes.rename(columns={
        "Station Name": "Station",
        "Elevation_x": "Elevation",
        "Slope_tif1_x": "Slope",
        "Aspect_tif_x": "Aspect",
        "Latitude_x": "Latitude",
        "Longitude_x": "Longitude",
    })
    snotel_attributes["Station"] = snotel_attributes["Station"].str.lower()

    return dynamic_forcing_and_swe, snotel_attributes