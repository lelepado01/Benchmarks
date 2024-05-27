


import torch
import torch.nn as nn

class ClimaXReconstructionHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, shape: tuple, embed_dim=256, decoder_depth=3):
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        self.decoder_depth = decoder_depth
        self.out_features = out_features
        self.shape = shape

        # prediction head
        self.head = nn.ModuleList()
        self.head.append(nn.Linear(self.in_features, self.embed_dim))
        for _ in range(self.decoder_depth-1):
            self.head.append(nn.Linear(self.embed_dim, self.embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(self.embed_dim, out_features))
        self.head = nn.Sequential(*self.head)

        # self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        return x.view(-1, *self.shape)
    
