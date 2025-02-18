import torch
from torch import nn

# -------------------------------
# GELU Activation
# -------------------------------
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))

# -------------------------------
# Softmax (if needed)
# -------------------------------
class Softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.exp(x) / torch.sum(torch.exp(x), dim=self.dim, keepdim=True)