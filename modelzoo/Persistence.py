import torch
import torch.nn as nn

class Persistence(nn.Module):
    """
    Persistence baseline:
    predict today's SWE as SWE_lag_1 (yesterday's SWE).
    """

    def __init__(self, lag_days: int):
        super().__init__()
        if lag_days < 1:
            raise ValueError("Persistence baseline requires cfg.lag_days >= 1.")
        self.lag_days = lag_days

    def forward(self, x: torch.Tensor, stations=None, dates=None) -> torch.Tensor:
        swe_lag1_idx = x.shape[-1] - self.lag_days
        return x[:, :, swe_lag1_idx]