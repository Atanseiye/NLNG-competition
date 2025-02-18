import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_head, dropout, qvk_bias=True):
        super().__init__()
        assert d_out % num_head == 0, "d_out must be divisible by num_head"

        self.d_out = d_out
        self.num_head = num_head
        self.head_dim = d_out // num_head

        self.W_query = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x).view(b, num_tokens, self.num_head, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(b, num_tokens, self.num_head, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_head, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        attn_scores = (queries @ keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # No causal mask -> Bidirectional Attention
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context_vec = attn_weights @ values

        # Reshape and project output
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
