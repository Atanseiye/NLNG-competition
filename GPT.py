import torch
import torch.nn as nn
from transformer import TransformerBlock
from layers import LayerNorm

import os

# ================================
# GPT Model Definition for Time-Series Fault Detection
class TimeSeriesTransformer(nn.Module):
    def __init__(self, config):
        """
        config should include:
          - feature_dim: Number of input features per time step (e.g., 10, 20, etc.)
          - emb_dim: Dimension of the Transformer embeddings (e.g., 128, 256, etc.)
          - n_layers: Number of Transformer blocks (e.g., 4, 6, etc.)
          - context_length: Maximum sequence length (seq_len)
          - drop_rate: Dropout probability
          - output_dim: Dimension of the output (e.g., # of classes for classification or 1 for regression)
          - n_heads: Number of attention heads in each block
          - (Any other hyperparameters your TransformerBlock needs)
        """
        super().__init__()

        # 1) Project numeric features into an embedding dimension
        self.feature_proj = nn.Linear(config['feature_dim'], config['emb_dim'])

        # 2) Learnable positional embeddings for time steps
        self.pos_emd = nn.Embedding(config['context_lenght'], config['emb_dim'])

        # 3) Dropout for regularization
        self.drop_emd = nn.Dropout(config['drop_rate'])

        # 4) Stacked Transformer blocks
        self.trf_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['n_layers'])]
        )

        # 5) Final layer normalization and output head
        self.final_norm = LayerNorm(config['emb_dim'])

        # --- Two separate heads ---
        self.fault_head = nn.Linear(config['emb_dim'], config['output_dim'])  # Classification
        self.time_head  = nn.Linear(config['emb_dim'], 1)                            # Regression

  
        # self.out_head = nn.Linear(config['emb_dim'], config['output_dim'], bias=True)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, feature_dim)
        """
        b, seq_len, _ = x.shape

        # (A) Project features to the embedding dimension
        x = self.feature_proj(x)  # -> (batch_size, seq_len, emb_dim)

        # (B) Add positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # shape (1, seq_len)
        pos_emb = self.pos_emd(positions)  # shape (1, seq_len, emb_dim)
        x = x + pos_emb  # broadcast addition

        # (C) Apply dropout
        x = self.drop_emd(x)

        # (D) Pass through Transformer blocks
        x = self.trf_block(x)  # -> (batch_size, seq_len, emb_dim)

        # (E) Final normalization
        x = self.final_norm(x)  # -> (batch_size, seq_len, emb_dim)

        # (F) For classification or regression, we often use the last time step
        #     or some pooling across all time steps. Below is last-step usage:
        x = x[:, -1, :]  # -> (batch_size, emb_dim)

        # (G) Output projection (classification or regression)
        # --- Output from each head ---
        fault_logits   = self.fault_head(x)  # shape: (b, num_fault_classes)
        time_to_failure = self.time_head(x)  # shape: (b, 1)

        return fault_logits, time_to_failure

