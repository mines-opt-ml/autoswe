from functools import reduce

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from SpatialTransform import SpatialTransformer


class SWEDataLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def prepare(self):
        swe = pd.read_csv(self.cfg.swe_path)
        swe["Date"] = pd.to_datetime(swe["Date"]).dt.strftime("%Y-%m-%d")
        meta = pd.read_csv(self.cfg.meta_path)

        def melt_dynamic(path, var_name):
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")  # normalize date format
            df_long = df.melt(id_vars="Date", var_name="Station", value_name=var_name)
            df_long["Station"] = df_long["Station"].str.replace("_", " ").str.lower()
            return df_long

        max_temp = melt_dynamic(self.cfg.max_temp_path, "Tmax")
        min_temp = melt_dynamic(self.cfg.min_temp_path, "Tmin")
        precip = melt_dynamic(self.cfg.precip_path, "Precip")
        obs_temp = melt_dynamic(self.cfg.obs_temp_path, "Tobs")
        tb19 = melt_dynamic(self.cfg.tb19_path, "TB_19")
        tb37 = melt_dynamic(self.cfg.tb37_path, "TB_37")
        tb_diff = melt_dynamic(self.cfg.tbdiff_path, "TB_diff")

        dynamic_dfs = [max_temp, min_temp, precip, obs_temp, tb19, tb37, tb_diff]
        dynamic_merged = reduce(lambda left, right: pd.merge(left, right, on=["Date", "Station"]), dynamic_dfs)

        swe_long = swe.melt(id_vars="Date", var_name="Station", value_name="SWE")
        swe_long["Station"] = swe_long["Station"].str.replace("_", " ").str.lower()
        meta["Station_clean"] = meta["Station Name"].str.lower()

        merged = pd.merge(swe_long, dynamic_merged, on=["Date", "Station"], how="inner")
        merged = pd.merge(merged, meta, left_on="Station", right_on="Station_clean", how="inner")
        merged = merged.rename(columns={"Station_clean": "Station"})
        merged = merged.dropna()

        daily = merged[merged["Date"] == self.cfg.sample_date].copy()  # filtering now using cfg

        train_df, val_df = train_test_split(daily, test_size=0.2, random_state=42)

        def build_features(df):
            return pd.DataFrame(
                {
                    "Elevation": df["Elevation_x"],
                    "Slope": df["Slope_tif1_x"],
                    "Aspect": df["Aspect_tif_x"],
                    "Latitude": df["Latitude_x"],
                    "Longitude": df["Longitude_x"],
                    "DayOfYear": pd.to_datetime(df["Date"]).dt.dayofyear,
                    "Tmax": df["Tmax"],
                    "Tmin": df["Tmin"],
                    "Precip": df["Precip"],
                    "Tobs": df["Tobs"],
                    "TB_19": df["TB_19"],
                    "TB_37": df["TB_37"],
                    "TB_diff": df["TB_diff"],
                }
            )

        X_train = build_features(train_df)
        X_train["SWE"] = train_df["SWE"]
        X_val = build_features(val_df)
        X_val["SWE"] = val_df["SWE"]

        transformer = SpatialTransformer()
        spatial_obj = transformer.transform_to_ind(
            target="SWE",
            trainData=X_train,
            trainLocs=train_df[["Latitude_x", "Longitude_x"]].to_numpy(),
            testData=X_val,
            testLocs=val_df[["Latitude_x", "Longitude_x"]].to_numpy(),
            smoothness=0.5,
            range_param=1.0,
            nugget=0.01,
            M=30,
            ncores=1,  # set ncores to 1 but could be increased if we start doing more in-depth predictions/incorporate temporal data
        )

        train_dataset = SWEStationDataset(
            spatial_obj["trainData"].drop(columns=["SWE"]).values, spatial_obj["trainData"]["SWE"].values
        )
        val_dataset = SWEStationDataset(spatial_obj["testData"].values, val_df["SWE"].values)

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=self.cfg.shuffle)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size)

        return train_loader, val_loader, transformer, val_df, spatial_obj
