import torch
from torch import nn
from layers import LayerNorm, Feedforward
from attention import MultiHeadAttention

# -------------------------------
# TransformerBlock Module
# -------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config['emb_dim'],
            d_out=config['emb_dim'],
            num_head=config['n_heads'],
            dropout=config['drop_rate'],
            qkv_bias=config['qkv_bias']
        )
        self.ff = Feedforward(config)
        self.norm1 = LayerNorm(config['emb_dim'])
        self.norm2 = LayerNorm(config['emb_dim'])
        self.drop_shortcut = nn.Dropout(config['drop_rate'])

    def forward(self, x):
        # Residual connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Residual connection for feedforward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

# -------------------------------
# TimeSeriesTransformer Model for Sensor Data
# -------------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, config):
        """
        Expects:
         - feature_dim: Number of numeric sensor features per time step.
         - emb_dim: Transformer embedding dimension.
         - context_lenght: Maximum sequence length.
         - drop_rate: Dropout probability.
         - output_dim: Number of classes (or 1 for regression).
         - n_layers: Number of transformer layers.
         - n_heads: Number of attention heads.
        """
        super().__init__()
        # 1) Project numeric features into the embedding space.
        self.feature_proj = nn.Linear(config['feature_dim'], config['emb_dim'])
        # 2) Positional embeddings.
        self.pos_emd = nn.Embedding(config['context_lenght'], config['emb_dim'])
        # 3) Dropout.
        self.drop_emd = nn.Dropout(config['drop_rate'])
        # 4) Transformer blocks.
        self.trf_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['n_layers'])]
        )
        # 5) Final normalization and output head.
        self.final_norm = LayerNorm(config['emb_dim'])
        self.out_head = nn.Linear(config['emb_dim'], config['output_dim'], bias=True)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, feature_dim)
        """
        b, seq_len, _ = x.shape
        # (A) Project to embedding dimension.
        x = self.feature_proj(x)  # (b, seq_len, emb_dim)
        # (B) Add positional embeddings.
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        pos_emb = self.pos_emd(positions)  # (1, seq_len, emb_dim)
        x = x + pos_emb
        # (C) Apply dropout.
        x = self.drop_emd(x)
        # (D) Transformer blocks.
        x = self.trf_block(x)  # (b, seq_len, emb_dim)
        # (E) Final normalization.
        x = self.final_norm(x)
        # (F) Pool the output using the final time step.
        x = x[:, -1, :]  # (b, emb_dim)
        # (G) Output projection.
        logits = self.out_head(x)  # (b, output_dim)
        return logits