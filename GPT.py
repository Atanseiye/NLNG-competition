import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import YOR_GPT_CONFIG_124M
from v2_preprocessing import TokenizerV2, vocabs, create_dataloader_v1
from transformer import TransformerBlock
from layers import LayerNorm
from tokenizers import Tokenizer
from tokenizers.models import BPE
import os

# ================================
# GPT Model Definition for Time-Series Fault Detection
# ================================
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emd = nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.pos_emd = nn.Embedding(config['context_lenght'], config['emb_dim'])
        self.drop_emd = nn.Dropout(config['drop_rate'])

        # Transformer Blocks
        self.trf_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['n_layers'])]
        )

        # Layer Normalization & Output (adapted for continuous regression task)
        self.final_norm = LayerNorm(config['emb_dim'])
        self.out_head = nn.Linear(config['emb_dim'], config['vocab_size'], bias=False)  # Consider regression for continuous prediction

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emd(in_idx)
        
        # Temporal position embedding: You can adjust or experiment with these
        pos_embeds = self.pos_emd(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emd(x)

        # Pass through transformer blocks
        x = self.trf_block(x)

        # Final normalization
        x = self.final_norm(x)

        # Adapt output layer for continuous prediction (e.g., fault prediction)
        logits = self.out_head(x)  # For time-series prediction, logits could be continuous sensor values

        return logits  # Output for continuous regression


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
        self.pos_emd = nn.Embedding(config['context_length'], config['emb_dim'])

        # 3) Dropout for regularization
        self.drop_emd = nn.Dropout(config['drop_rate'])

        # 4) Stacked Transformer blocks
        self.trf_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['n_layers'])]
        )

        # 5) Final layer normalization and output head
        self.final_norm = LayerNorm(config['emb_dim'])
        self.out_head = nn.Linear(config['emb_dim'], config['output_dim'], bias=True)

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
        logits = self.out_head(x)  # -> (batch_size, output_dim)

        return logits

