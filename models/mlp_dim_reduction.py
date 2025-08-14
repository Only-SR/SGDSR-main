import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDimReduction(nn.Module):
    def __init__(self, input_dim, output_dim,device):
        super(MLPDimReduction, self).__init__()
        self.device=device
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()  

    def forward(self, x):
        x=x.to(self.device)
        x = self.fc(x)
        x = self.activation(x)
        return x

