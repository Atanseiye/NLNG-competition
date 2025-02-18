import torch
from torch import nn
from layers import LayerNorm, Feedforward
from attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config['emb_dim'],   # Input dimension = number of sensor features
            d_out=config['emb_dim'],  # Output dimension = number of sensor features
            num_head=config['n_heads'],
            dropout=config['drop_rate'],
            qvk_bias=config['qkv_bias']  # Fixed typo (qvk_bais -> qvk_bias)
        )
        self.ff = Feedforward(config)  
        self.norm1 = LayerNorm(config['emb_dim'])  
        self.norm2 = LayerNorm(config['emb_dim'])  
        self.drop_shortcut = nn.Dropout(config['drop_rate'])  

    def forward(self, x):
        # x shape: (batch_size, time_steps, num_sensors)
        
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Attention across time steps
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        # Shortcut connection for Feedforward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)  # Per time-step transformation
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        return x
