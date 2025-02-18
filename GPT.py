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

