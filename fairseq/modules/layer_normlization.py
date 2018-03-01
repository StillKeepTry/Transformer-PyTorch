import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """Layer normalization for module"""

    def __init__(self, hidden_size, eps=1e-6, affine=True):
        super(LayerNormalization, self).__init__()

        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(hidden_size))
            self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
