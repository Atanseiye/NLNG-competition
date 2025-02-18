import re
import torch
from torch.utils.data import DataLoader, Dataset

# ================================
# Dataset Class (Sliding Window for Sensor Data)
# ================================
# -------------------------------
# Dataset Class for Power Grid Data (Sliding Window)
# -------------------------------
class PowerGridDataset(Dataset):
    def __init__(self, df, seq_len=10, fault_col='fault_type', ttf_col='time_to_failure'):
        """
        df: DataFrame with sensor readings, a fault_type label, and a timestamp.
        seq_len: Number of time steps per sample.
        label_col: Column name for the target fault type.
        """
        self.seq_len = seq_len
        self.fault_col = fault_col
        self.ttf_col = ttf_col

        # We'll drop the classification and regression columns + timestamp from features.
        # Modify the drop list if your columns differ.
        self.data = torch.tensor(
            df.drop(columns=[fault_col, ttf_col, 'timestamp']).values,
            dtype=torch.float
        )
        # Classification labels (fault type)
        self.labels_cls = torch.tensor(df[fault_col].values, dtype=torch.long)
        
        # Regression labels (time to failure)
        self.labels_reg = torch.tensor(df[ttf_col].values, dtype=torch.float)

        self.samples = []
        self.targets_cls = []
        self.targets_reg = []

        for i in range(len(self.data) - seq_len):
            x = self.data[i : i+seq_len]                 # shape: (seq_len, feature_dim)
            y_cls = self.labels_cls[i+seq_len]          # classification label at end of window
            y_reg = self.labels_reg[i+seq_len]          # regression label at end of window
            
            self.samples.append(x)
            self.targets_cls.append(y_cls)
            self.targets_reg.append(y_reg)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            self.samples[idx],                # features
            (self.targets_cls[idx], self.targets_reg[idx])  # (classification label, regression label)
        )
    
def create_powergrid_dataloader(df, seq_len=10, batch_size=32, shuffle=True,
                                fault_col='fault_type', ttf_col='time_to_failure'):
    dataset = PowerGridDataset(df, seq_len=seq_len, 
                                        fault_col=fault_col,
                                        ttf_col=ttf_col)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

