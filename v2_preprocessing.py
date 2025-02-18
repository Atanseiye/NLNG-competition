import re
import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
import tiktoken

# ================================
# Dataset Class (Sliding Window for Sensor Data)
# ================================
class PowerGridDataset(Dataset):
    def __init__(self, df, seq_len=10, label_col='fault_type'):
        """
        df: your dataframe with numeric sensor readings
        seq_len: how many time steps per sample
        label_col: which column is your target label
        """
        self.seq_len = seq_len
        self.label_col = label_col

        # Convert entire DataFrame to a numeric tensor (except label if needed separately)
        self.data = torch.tensor(df.drop(columns=[label_col, 'timestamp']).values, dtype=torch.float)
        self.labels = torch.tensor(df[label_col].values, dtype=torch.long)  # or float for regression

        self.samples = []
        self.targets = []

        # Create sliding windows
        for i in range(len(self.data) - seq_len):
            x = self.data[i:i+seq_len]         # shape: (seq_len, feature_dim)
            y = self.labels[i+seq_len]         # label for the sequence end, or however you define it
            self.samples.append(x)
            self.targets.append(y)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]



def create_powergrid_dataloader(df, seq_len=10, batch_size=32, shuffle=True):
    dataset = PowerGridDataset(df, seq_len=seq_len, label_col='fault_type')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# ================================
# Tokenizer for Sensor Data (Binned Data)
# ================================
class SensorTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, sensor_data_point):
        # Encode sensor data (either numerical value or category)
        return [self.str_to_int.get(sensor_data_point, self.str_to_int['<|unk|>'])]

    def decode(self, token_ids):
        # Decode token ids back to sensor values (for interpretation)
        return ' '.join([self.int_to_str.get(s, '<|unk|>') for s in token_ids])

# ================================
# Vocabulary Builder for Sensor Data
# ================================
def build_sensor_vocab(sensor_data):
    # Build vocab based on numerical ranges or sensor categories
    # For example, you can bin numerical sensor readings into ranges
    vocab = {f"sensor_{i}": i for i in range(len(sensor_data))}
    vocab.update({'<|unk|>': len(vocab)})  # Handle unknown sensor values
    return vocab

# ================================
# Embedding Class for Power Grid
# ================================
class PowerGridEmbedding:
    @staticmethod
    def token_embedding_layer(vocab_size, output_dim):
        torch.manual_seed(123)
        return torch.nn.Embedding(vocab_size, output_dim)

# Example usage:
sensor_data = ["sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5"]  # Replace with actual sensor data
vocab = build_sensor_vocab(sensor_data)
tokenizer = SensorTokenizer(vocab)
dataloader = create_dataloader_v1(sensor_data, tokenizer=tokenizer, batch_size=4)

