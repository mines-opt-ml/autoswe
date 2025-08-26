import torch
import torch.nn as nn
from types import SimpleNamespace

class SWE_Net(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.input_size = 13  
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(self.hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out.squeeze(-1)