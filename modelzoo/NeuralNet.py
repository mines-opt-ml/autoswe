import torch.nn as nn

#Define simple MLP (multi-layer perceptron)
class SWE_Net(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
    #forward pass:
    def forward(self, x):
        return self.model(x)