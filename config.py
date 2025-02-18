# Description: Configuration file for the GPT Model

YOR_GPT_CONFIG_124M = {
    'feature_dim': 12,           # Number of sensor features per time step
    'context_lenght': 256,       # Maximum sequence length (e.g., 256 time steps)
    'emb_dim': 768,              # Transformer embedding dimension
    'n_heads': 12,               # Number of attention heads
    'n_layers': 12,              # Number of transformer layers
    'drop_rate': 0.1,            # Dropout probability
    'qkv_bias': False,           # Use bias in QKV linear layers
    'output_dim': 5              # Number of fault classes (for classification)
}




config = {
    'feature_dim': 12,        # e.g., 12 numeric sensor features per time step
    'emb_dim': 128,
    'n_layers': 4,
    'context_length': 50,     # maximum sequence length (e.g., 50 time steps)
    'drop_rate': 0.1,
    'output_dim': 5,          # e.g., 5 fault classes OR 1 for regression
    'n_heads': 4,
    # Add other hyperparams your TransformerBlock needs
}
