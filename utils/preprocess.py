from functools import reduce
from typing import Tuple
import pandas as pd


def melt_dynamic(path: str, var_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df_long = df.melt(id_vars="Date", var_name="Station", value_name=var_name)
    df_long["Station"] = df_long["Station"].str.replace("_", " ").str.lower()
    return df_long


def preprocess(cfg: object) -> Tuple[pd.DataFrame, pd.DataFrame]
    swe = pd.read_csv(cfg.swe_path)
    swe["Date"] = pd.to_datetime(swe["Date"]).dt.strftime("%Y-%m-%d")
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

    dynamic_dfs = [melt_dynamic(path, name) for path, name in dynamic_inputs]
    dynamic_merged = reduce(lambda left, right: pd.merge(left, right, on=["Date", "Station"]), dynamic_dfs)

    merged = pd.merge(swe_long, dynamic_merged, on=["Date", "Station"], how="inner")
    # merged = pd.merge(merged, meta, left_on="Station", right_on="Station_clean", how="inner")
    merged = merged.rename(columns={"Station_clean": "Station"})
    dynamic_forcing_and_swe = merged.dropna()
    snotel_attributes = meta.rename(columns={"Station Name": "Station"})

    return dynamic_forcing_and_swe, snotel_attributes
