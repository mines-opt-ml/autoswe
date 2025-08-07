import pytest
from omegaconf import OmegaConf
import pandas as pd
from dataset import SWEStationDataset
import os
import torch

@pytest.fixture
def config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    return OmegaConf.load(config_path)

def test_dataset_time_series(config):
    dataset = SWEStationDataset(config)
    
    # Test lookup table
    assert len(dataset.lookup_table) > 0, "Lookup table is empty"
    print(f"\nLookup table has {len(dataset.lookup_table)} entries")
    
    sample = dataset[0]


    assert "dynamic forcing" in sample
    assert "swe" in sample
    assert "snotel attributes" in sample
    
    forcing = sample["dynamic forcing"]
    swe = sample["swe"]
    
    #Expect data from Oct 1 to Apr 1 (approximately 182 days)
    expected_min_length = 150  #allows some flexibility for missing data - can increase later
    assert forcing.shape[0] >= expected_min_length
    assert swe.shape[0] == forcing.shape[0]  
    
    #Test if forcing data contains all 13 features
    assert forcing.shape[1] == 13
    
    #data types
    assert forcing.dtype == torch.float32
    assert swe.dtype == torch.float32