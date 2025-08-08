from torch.utils.data import DataLoader

from dataset import SWEStationDataset
from utils.preprocess import preprocess
from utils.SpatialTransform import SpatialTransformer


class SWEDataLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def prepare(self) -> Tuple[DataLoader, DataLoader, SpatialTransformer, pd.DataFrame, dict]:
        merged = preprocess(self.cfg)
        daily = merged[merged["Date"] == self.cfg.sample_date].copy()
        train_df, val_df = train_test_split(daily, test_size=0.2, random_state=42)

        def build_features(df):
            return pd.DataFrame(
                {
                    "Elevation": df["Elevation"],
                    "Slope": df["Slope"],
                    "Aspect": df["Aspect"],
                    "Latitude": df["Latitude"],
                    "Longitude": df["Longitude"],
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
