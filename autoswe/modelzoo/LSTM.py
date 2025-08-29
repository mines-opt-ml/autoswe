import torch
import torch.nn as nn
from types import SimpleNamespace

class SWE_Net(nn.Module):
    def __init__(self, cfg: SimpleNamespace, station_stats=None):
        super().__init__()
        self.input_size = 13  
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers
        
        self.station_stats = station_stats or {}
        
        self.use_station_context = len(self.station_stats) > 0
        if self.use_station_context:
            self.station_context_size = 2
            lstm_input_size = self.input_size + self.station_context_size
        else:
            lstm_input_size = self.input_size
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2 if self.num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(0.1)
        
        self.fc = nn.Linear(self.hidden_size, 1)
        
        self.output_scale = nn.Parameter(torch.tensor(2.0))
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data, gain=2.0)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data, gain=1.0)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'fc.weight' in name:
                torch.nn.init.xavier_uniform_(param.data, gain=2.0)
            elif 'fc.bias' in name:
                param.data.fill_(0.0)
            elif 'weight' in name and 'fc' in name:
                torch.nn.init.xavier_uniform_(param.data)
                param.data *= 10.0
    
    def forward(self, x, stations=None):
        if self.use_station_context and stations is not None:
            batch_size, seq_len, _ = x.shape
            
            station_context = torch.zeros(batch_size, seq_len, self.station_context_size, 
                                        device=x.device, dtype=x.dtype)
            
            for i, station in enumerate(stations):
                if station in self.station_stats:
                    mean_swe, std_swe = self.station_stats[station]
                    station_context[i, :, 0] = mean_swe / 20.0
                    station_context[i, :, 1] = std_swe / 10.0
            
            x = torch.cat([x, station_context], dim=-1)
        
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        out = out * self.output_scale
        return out.squeeze(-1)