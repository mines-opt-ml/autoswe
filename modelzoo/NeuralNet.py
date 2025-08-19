import torch
import torch.nn as nn

class SWE_Net(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64)):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            batch_first=True 
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  
        
        predictions = []
        for t in range(lstm_out.size(1)):
            time_slice = lstm_out[:, t, :]  
            pred = self.mlp(time_slice)   
            predictions.append(pred)
            
        return torch.stack(predictions, dim=1).squeeze(-1)  # [batch_size, seq_length]