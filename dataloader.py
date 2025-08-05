from torch.utils.data import DataLoader

from utils.SpatialTransform import SpatialTransformer


class SWEDataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
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
