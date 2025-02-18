from utils import GELU
from torch import nn
import torch

# -------------------------------
# Feedforward Module
# -------------------------------
class Feedforward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(config['emb_dim'], 4 * config['emb_dim']),
            GELU(),
            nn.Linear(4 * config['emb_dim'], config['emb_dim'])
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.layer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layer(x)

# -------------------------------
# LayerNorm Module (Using PyTorch's built-in LayerNorm)
# -------------------------------
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(emb_dim, eps=1e-5)

    def forward(self, x):
        return self.layer_norm(x)
