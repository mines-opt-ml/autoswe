import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, Sequence, Tuple

class HistoricalMean(nn.Module):
    def __init__(self, climo_lookup: Dict[Tuple[str, int], float], station_mean: Dict[str, float]):
        super().__init__()
        self.climo = climo_lookup
        self.station_mean = station_mean

    @torch.no_grad()
    def forward(self, x: torch.Tensor, stations: Sequence[str], dates: Sequence[Sequence[str]]) -> torch.Tensor:
        device = x.device if isinstance(x, torch.Tensor) else "cpu"
        B = len(stations)
        lengths = [len(d) for d in dates] if B > 0 else []
        T = max(lengths) if lengths else 0
        out = torch.empty((B, T), dtype=torch.float32, device=device)
        for i, st in enumerate(stations):
            st_mean = float(self.station_mean.get(st, 0.0))
            seq_dates = dates[i]
            for t in range(len(seq_dates)):
                doy = pd.Timestamp(seq_dates[t]).dayofyear
                out[i, t] = float(self.climo.get((st, doy), st_mean))
            if len(seq_dates) < T:
                out[i, len(seq_dates):] = st_mean
        return out