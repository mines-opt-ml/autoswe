from typing import Dict, List, Tuple
import sys
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset

from dataset import SWEStationDataset
from utils.SpatialTransform import SpatialTransformer, fast_transform_with_weights, precompute_weights

class SWEDataLoader:
    def __init__(self, cfg, dataset = None):
        self.cfg = cfg
        self.dataset = dataset
        self.conformal_cfg = getattr(cfg, "conformal", None)
        self.conformal_enabled = bool(getattr(self.conformal_cfg, "enabled", False))
        self.train_years = range(cfg.train_start_year, cfg.train_end_year + 1)
        self.val_years = range(cfg.val_start_year, cfg.val_end_year + 1)
        self.test_years = range(cfg.test_start_year, cfg.test_end_year + 1)
        self.calibration_years = None

        if self.conformal_enabled:
            split_name = getattr(self.conformal_cfg, "split", "calibration")
            if split_name != "calibration":
                raise ValueError("Only conformal.split='calibration' is currently supported.")
            missing = [
                name
                for name in ["calibration_start_year", "calibration_end_year"]
                if not hasattr(self.conformal_cfg, name)
            ]
            if missing:
                raise ValueError(
                    "conformal.enabled=True requires conformal field(s): "
                    + ", ".join(missing)
                )
            self.calibration_years = range(
                self.conformal_cfg.calibration_start_year,
                self.conformal_cfg.calibration_end_year + 1,
            )
        
        split_sets = {
            "train": set(self.train_years),
            "val": set(self.val_years),
            "test": set(self.test_years),
        }
        if self.conformal_enabled:
            split_sets["calibration"] = set(self.calibration_years)

        split_names = list(split_sets)
        for i, left in enumerate(split_names):
            for right in split_names[i + 1:]:
                years = split_sets[left] & split_sets[right]
                if years:
                    raise AssertionError(
                        f"Year split overlap detected in {left}_{right}: {sorted(years)}. "
                        f"Check cfg.*_start_year / cfg.*_end_year."
                    )

    def prepare(self) -> Tuple[Dict[str, DataLoader], SpatialTransformer, dict]:
        """
        Create train, validation, and test data loaders, optionally applying
        spatial decorrelation.
        """
        dataset = self.dataset or SWEStationDataset(self.cfg)
        use_spatial_decorrelation = getattr(
            self.cfg, "spatial_decorrelation", True
        )

        orig_swe = dataset.dynamic_forcing_and_swe.reset_index()
        orig_swe = orig_swe[["Date", "Station", "SWE"]].copy()

        dataset.dynamic_forcing_and_swe = dataset.dynamic_forcing_and_swe.reset_index()

        weights = None
        station_index = None
        backtrans_cache = None

        all_dates = orig_swe["Date"].unique()
        complete_date_range = pd.date_range(
            start=all_dates.min(), end=all_dates.max(), freq="D"
        )
        min_coverage = 95.0
        complete_stations = []

        for station in dataset.snotel_attributes["Station"].values:
            station_data = orig_swe[orig_swe["Station"] == station]
            station_dates = station_data["Date"].unique()
            coverage = len(station_dates) / len(complete_date_range) * 100
            if coverage >= min_coverage:
                complete_stations.append(station)

        dataset.snotel_attributes = dataset.snotel_attributes[
            dataset.snotel_attributes["Station"].isin(complete_stations)
        ].copy()
        dataset.dynamic_forcing_and_swe = dataset.dynamic_forcing_and_swe[
            dataset.dynamic_forcing_and_swe["Station"].isin(complete_stations)
        ].copy()

        if use_spatial_decorrelation:
            station_order = dataset.snotel_attributes["Station"].values
            station_index = {s: i for i, s in enumerate(station_order)}
            locs_all = dataset.snotel_attributes[["Latitude", "Longitude"]].values

            weights = precompute_weights(
                trainLocs=locs_all,
                M=self.cfg.M,
                range_param=self.cfg.range_param,
                smoothness=self.cfg.smoothness,
                nugget=self.cfg.nugget,
            )

            backtrans_cache = {}
            unique_dates = sorted(dataset.dynamic_forcing_and_swe["Date"].unique())

            for dt in unique_dates:
                swe_dt = pd.DataFrame({"Station": station_order}).merge(
                    orig_swe[orig_swe["Date"] == dt][["Station", "SWE"]],
                    on="Station",
                    how="left",
                )
                if swe_dt["SWE"].isna().any():
                    swe_dt["SWE"] = swe_dt["SWE"].fillna(swe_dt["SWE"].mean())

                y_vec = swe_dt["SWE"].to_numpy(dtype=np.float32)
                offsets = np.empty(len(station_order), dtype=np.float32)
                for i, nbrs in enumerate(weights.nn_index):
                    offsets[i] = np.dot(weights.A[i], y_vec[nbrs])

                backtrans_cache[pd.Timestamp(dt).strftime("%Y-%m-%d")] = offsets

            def apply_decorrelation_for_dates(dates_iterable):
                for dt in dates_iterable:
                    time_mask = dataset.dynamic_forcing_and_swe["Date"] == dt
                    df_dt = dataset.dynamic_forcing_and_swe[time_mask].copy()
                    aligned = pd.DataFrame({"Station": station_order}).merge(
                        df_dt, on="Station", how="left"
                    )

                    numeric_cols = aligned.select_dtypes(
                        include=[np.number]
                    ).columns.tolist()
                    for col in numeric_cols:
                        if aligned[col].isna().any():
                            aligned[col] = aligned[col].fillna(aligned[col].mean())

                    dynamic_cols = getattr(
                        self.cfg,
                        "dynamic_features",
                        [
                            "Tmax",
                            "Tmin",
                            "Precip",
                            "Tobs",
                            "TB_19",
                            "TB_37",
                            "TB_diff",
                        ],
                    )
                    cols_to_decor = ["SWE"] + list(dynamic_cols)

                    transformed = fast_transform_with_weights(
                        trainData=aligned.copy(),
                        target="SWE",
                        weights=weights,
                        cols_to_transform=list(dynamic_cols),
                        static_cols=None,
                        station_col="Station",
                    )

                    updated = dataset.dynamic_forcing_and_swe.loc[time_mask].merge(
                        transformed[["Station"] + cols_to_decor],
                        on="Station",
                        suffixes=("", "_decor"),
                        how="left",
                    )
                    for col in cols_to_decor:
                        dataset.dynamic_forcing_and_swe.loc[time_mask, col] = (
                            updated[f"{col}_decor"].values
                        )

            train_dates = [
                d for d in unique_dates if pd.Timestamp(d).year in self.train_years
            ]
            val_dates = [
                d for d in unique_dates if pd.Timestamp(d).year in self.val_years
            ]
            calibration_dates = (
                [
                    d
                    for d in unique_dates
                    if pd.Timestamp(d).year in self.calibration_years
                ]
                if self.conformal_enabled
                else []
            )
            test_dates = [
                d for d in unique_dates if pd.Timestamp(d).year in self.test_years
            ]

            apply_decorrelation_for_dates(train_dates)
            apply_decorrelation_for_dates(val_dates)
            apply_decorrelation_for_dates(calibration_dates)
            apply_decorrelation_for_dates(test_dates)

        available_stations = set(dataset.dynamic_forcing_and_swe["Station"].unique())
        train_indices = [
            idx
            for idx, entry in enumerate(dataset.lookup_table)
            if entry["Year"] in self.train_years
            and entry["Station"] in available_stations
        ]
        val_indices = [
            idx
            for idx, entry in enumerate(dataset.lookup_table)
            if entry["Year"] in self.val_years
            and entry["Station"] in available_stations
        ]
        calibration_indices = (
            [
                idx
                for idx, entry in enumerate(dataset.lookup_table)
                if entry["Year"] in self.calibration_years
                and entry["Station"] in available_stations
            ]
            if self.conformal_enabled
            else []
        )
        test_indices = [
            idx
            for idx, entry in enumerate(dataset.lookup_table)
            if entry["Year"] in self.test_years
            and entry["Station"] in available_stations
        ]

        if getattr(self.cfg, "anomaly_target", False):
            df_all = dataset.dynamic_forcing_and_swe.copy()
            df_all["DOY"] = pd.to_datetime(df_all["Date"]).dt.dayofyear
            train_mask = df_all["Date"].dt.year.isin(self.train_years)
            climo_model = (
                df_all.loc[train_mask]
                .groupby(["Station", "DOY"])["SWE"]
                .mean()
                .reset_index()
                .rename(columns={"SWE": "SWE_climo"})
            )
            station_train_mean = df_all.loc[train_mask].groupby("Station")["SWE"].mean().rename("SWE_climo_fallback")
            df_all = df_all.merge(climo_model, on=["Station", "DOY"], how="left")
            df_all = df_all.merge(station_train_mean, on="Station", how="left")
            df_all["SWE_climo"] = df_all["SWE_climo"].fillna(df_all["SWE_climo_fallback"]).fillna(0.0)
            df_all = df_all.drop(columns=["SWE_climo_fallback", "DOY"])
            dataset.dynamic_forcing_and_swe = df_all

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        calibration_dataset = (
            Subset(dataset, calibration_indices) if self.conformal_enabled else None
        )
        test_dataset = Subset(dataset, test_indices)

        generator = None
        if hasattr(self.cfg, "seed"):
            generator = torch.Generator()
            generator.manual_seed(self.cfg.seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            collate_fn=self.collate_fn,
            generator=generator,
        )

        val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size, collate_fn=self.collate_fn)
        calibration_loader = (
            DataLoader(
                calibration_dataset,
                batch_size=self.cfg.batch_size,
                collate_fn=self.collate_fn,
            )
            if self.conformal_enabled
            else None
        )
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size, collate_fn=self.collate_fn)
        bt_info = {
            "spatial_decorrelation": use_spatial_decorrelation,
            "weights": weights,
            "station_index": station_index,
            "backtrans_cache": backtrans_cache,
        }
        loaders = {
            "train": train_loader,
            "val": val_loader,
            "calibration": calibration_loader,
            "test": test_loader,
        }

        return loaders, SpatialTransformer(), bt_info

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to pad sequences in batch to same length.
        """
        dynamic = [item["dynamic forcing"] for item in batch]
        swe = [item["swe"] for item in batch]

        metadata = {key: [item[key] for item in batch] for key in batch[0].keys() if key not in ["dynamic forcing", "swe"]}

        dynamic_pad = pad_sequence(dynamic, batch_first=True)
        swe_pad = pad_sequence(swe, batch_first=True)
        lengths = torch.tensor([len(x) for x in dynamic])
        max_len = lengths.max()
        mask = torch.arange(max_len)[None, :] < lengths[:, None]

        return {"dynamic forcing": dynamic_pad, "swe": swe_pad, "mask": mask, **metadata}
