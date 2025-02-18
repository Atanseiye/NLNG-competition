from utils import GELU
from torch import nn
import torch

class Feedforward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(config['emb_dim'], 4 * config['emb_dim']),
            GELU(),  # Keep GELU or experiment with ReLU
            nn.Linear(4 * config['emb_dim'], config['emb_dim'])
        )

        # Weight initialization can help improve model performance
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.layer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to 0

    def forward(self, x):
        return self.layer(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # Using PyTorch's built-in LayerNorm for better stability and efficiency
        self.layer_norm = nn.LayerNorm(emb_dim, eps=1e-5)

    def forward(self, x):
        return self.layer_norm(x)
