# Description: Configuration file for the GPT Model


YOR_GPT_CONFIG_124M = {
    'feature_dim': 21,           # Number of sensor features per time step
    'context_lenght': 128,       # Maximum sequence length (e.g., 256 time steps)
    'emb_dim': 128,              # Transformer embedding dimension
    'n_heads': 4,               # Number of attention heads
    'n_layers': 4,              # Number of transformer layers
    'drop_rate': 0.1,            # Dropout probability
    'qkv_bias': False,           # Use bias in QKV linear layers
    'output_dim': 5              # Number of fault classes (for classification)
}




