from typing import Tuple
from torch.utils.data import DataLoader, Subset
from dataset import SWEStationDataset
from utils.SpatialTransform import SpatialTransformer
import numpy as np

class SWEDataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_years = range(cfg.beginning_year, cfg.train_end_year)
        self.val_years = range(cfg.val_start_year, cfg.end_year)

    def prepare(self) -> Tuple[DataLoader, DataLoader, SpatialTransformer, dict]:
        """
        Create train and validation data loaders with spatial decorrelation.
        
        Returns:
            Tuple containing:
            - train_loader: DataLoader for training data (2001-2014)
            - val_loader: DataLoader for validation data (2015-2018)
            - transformer: SpatialTransformer object for back-transformation
            - spatial_obj: Dictionary with transformation information
        """
        dataset = SWEStationDataset(self.cfg)
        
        train_indices = [
            idx for idx, entry in enumerate(dataset.lookup_table)
            if entry["Year"] in self.train_years
        ]
        val_indices = [
            idx for idx, entry in enumerate(dataset.lookup_table)
            if entry["Year"] in self.val_years
        ]

        train_locs = dataset.snotel_attributes[["Latitude", "Longitude"]].values
        
        transformer = SpatialTransformer()
        
        spatial_objs = {}

        unique_dates = sorted(dataset.dynamic_forcing_and_swe["Date"].unique())
        for dt in unique_dates:
            time_mask = dataset.dynamic_forcing_and_swe["Date"] == dt
            train_data = dataset.dynamic_forcing_and_swe[time_mask].copy()

            train_data_num = train_data.select_dtypes(include=[np.number]).copy()

            locs_dt = (
                train_data[["Station"]]
                .merge(
                    dataset.snotel_attributes[["Station", "Latitude", "Longitude"]],
                    on="Station",
                    how="left",
                )[["Latitude", "Longitude"]]
                .values
            )

            transformer = SpatialTransformer()
            spatial_obj_dt = transformer.transform_to_ind(
                target="SWE",
                trainData=train_data_num,     
                trainLocs=locs_dt,
                testData=train_data_num,     
                testLocs=locs_dt,
                smoothness=self.cfg.smoothness,
                range_param=self.cfg.range_param,
                nugget=self.cfg.nugget,
                M=self.cfg.M,
                ncores=self.cfg.ncores
            )

            dataset.dynamic_forcing_and_swe.loc[time_mask, "SWE"] = spatial_obj_dt["trainData"]["SWE"].values
            spatial_objs[dt] = spatial_obj_dt

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=self.cfg.shuffle)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size)

        return train_loader, val_loader, SpatialTransformer(), spatial_objs