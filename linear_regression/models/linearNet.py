import torch
from torch import nn

class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
        self.linear[0].weight.data.normal_(0, 0.01)
        self.linear[0].bias.data.fill_(0) 
