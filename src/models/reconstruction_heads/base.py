
import torch
import torch.nn as nn

class BaseReconstructionHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, shape: tuple):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shape = shape

        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x.view(-1, *self.shape)
    
