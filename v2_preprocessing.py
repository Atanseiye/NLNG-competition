import re
import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
import tiktoken

# ================================
# Dataset Class (Sliding Window for Sensor Data)
# ================================
# -------------------------------
# Dataset Class for Power Grid Data (Sliding Window)
# -------------------------------
class PowerGridDataset(Dataset):
    def __init__(self, df, seq_len=10, label_col='fault_type'):
        """
        df: DataFrame with sensor readings, a fault_type label, and a timestamp.
        seq_len: Number of time steps per sample.
        label_col: Column name for the target fault type.
        """
        self.seq_len = seq_len
        self.label_col = label_col
        # Drop label and timestamp columns; the remaining columns are features.
        self.data = torch.tensor(df.drop(columns=[label_col, 'timestamp']).values, dtype=torch.float)
        self.labels = torch.tensor(df[label_col].values, dtype=torch.long)
        self.samples = []
        self.targets = []
        for i in range(len(self.data) - seq_len):
            x = self.data[i:i+seq_len]      # (seq_len, feature_dim)
            y = self.labels[i+seq_len]        # Label corresponding to the end of the window.
            self.samples.append(x)
            self.targets.append(y)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

def create_powergrid_dataloader(df, seq_len=10, batch_size=32, shuffle=True):
    dataset = PowerGridDataset(df, seq_len=seq_len, label_col='fault_type')
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

