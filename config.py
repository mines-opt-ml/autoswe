from dataclasses import dataclass

import torch


# This is styled similar to Antony's Lattice work
@dataclass
class SWEConfig:
    def __init__(
        self,
        swe_path,
        meta_path,
        sample_date,
        # below are mostly defaults for hyperparameters; can be adjusted later
        hidden_dims=[128, 64],
        batch_size=1000,
        lr=0.001,
        n_epochs=100,
        smoothness=0.5,
        range_param=1.0,
        nugget=0.01,
        M=30,
        ncores=1,
        device=torch.device("cpu"),  # could be cuda
        shuffle=True,
    ):
        self.swe_path = swe_path
        self.meta_path = meta_path
        self.sample_date = sample_date
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.smoothness = smoothness
        self.range_param = range_param
        self.nugget = nugget
        self.M = M
        self.ncores = ncores
        self.device = device
        self.shuffle = shuffle
        self.save = True
        self.save_path = "results/model.pt"
