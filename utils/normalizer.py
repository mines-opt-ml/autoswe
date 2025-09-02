import numpy as np
import pandas as pd
import torch

class Normalizer:
    def __init__(self, method: str = "zscore"):
        """
        method: 'zscore' or 'minmax'
        """
        self.method = method
        self.stats = {}

    def fit(self, df: pd.DataFrame):
        """Compute normalization stats per column from training data."""
        for col in df.columns:
            vals = df[col].dropna().values
            if self.method == "zscore":
                mean = np.mean(vals)
                std = np.std(vals)
                if std < 1e-6:
                    std = 1.0
                self.stats[col] = {"mean": mean, "std": std}
            elif self.method == "minmax":
                self.stats[col] = {"min": np.min(vals), "max": np.max(vals)}
            else:
                raise ValueError(f"Unknown method: {self.method}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization using stored stats."""
        df_norm = df.copy()
        for col in df.columns:
            if col not in self.stats:
                continue
            if self.method == "zscore":
                mean, std = self.stats[col]["mean"], self.stats[col]["std"]
                df_norm[col] = (df[col] - mean) / std
            elif self.method == "minmax":
                minv, maxv = self.stats[col]["min"], self.stats[col]["max"]
                df_norm[col] = (df[col] - minv) / (maxv - minv + 1e-6)
        return df_norm

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Undo normalization."""
        df_inv = df.copy()
        for col in df.columns:
            if col not in self.stats:
                continue
            if self.method == "zscore":
                mean, std = self.stats[col]["mean"], self.stats[col]["std"]
                df_inv[col] = df[col] * std + mean
            elif self.method == "minmax":
                minv, maxv = self.stats[col]["min"], self.stats[col]["max"]
                df_inv[col] = df[col] * (maxv - minv + 1e-6) + minv
        return df_inv