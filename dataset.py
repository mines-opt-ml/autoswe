from torch.utils.data import Dataset
import torch
import pandas as pd

class SWEStationDataset(Dataset):
    '''
    The dataset class should handle loading the csv files and processing the data.
    Each item in the dataset should be (X: torch.Tensor[num_features, window_length], y: torch.Tensor[, window_length])
    where X is the features and y is the target SWE value.
    Windoow_length is the length of a snow year.
    '''
    def __init__(self, cfg: ):
        super(SWEStationDataset, self).__init__()
        self.cfg = cfg
        # self.X = torch.tensor(X, dtype=torch.float32)
        # self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1) 
        self.beginning_of_snow_year = self.cfg.beginning_of_snow_year
        self.peak_swe_date = self.cfg.peak_swe_date
        self.beginning_year = self.cfg.beginning_year
        self.end_year = self.cfg.end_year
        self._get_start_and_end_dates() # Todo: implement this method. Should read these dates from the config
    


    def __len__(self):
        return len(self.X) 

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx] 

    def _get_(self):
        
        
