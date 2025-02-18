import torch
import torch.nn as nn
from config import YOR_GPT_CONFIG_124M
import pandas as pd
from v2_preprocessing import create_powergrid_dataloader
from transformer import TimeSeriesTransformer
import torch.optim as optim


# -------------------------------
# Training Loop
# -------------------------------
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)      # shape: (batch_size, seq_len, feature_dim)
            labels = labels.to(device)      # shape: (batch_size,)
            
            optimizer.zero_grad()
            outputs = model(inputs)         # shape: (batch_size, output_dim)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
    print("Training complete.")

# -------------------------------
# Testing the Training Loop on Dummy Data
# -------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a dummy DataFrame.
    # We need 12 feature columns, a 'fault_type' column (target), and a 'timestamp' column.
    num_rows = 300  # For example, 300 time steps.
    feature_columns = [f'f{i+1}' for i in range(12)]
    data_dict = {col: torch.randn(num_rows).tolist() for col in feature_columns}
    # Create random fault_type labels (5 classes: 0 to 4).
    data_dict['fault_type'] = torch.randint(0, 5, (num_rows,)).tolist()
    # Create dummy timestamps.
    data_dict['timestamp'] = pd.date_range(start='2023-01-01', periods=num_rows, freq='H').astype(str).tolist()
    df = pd.DataFrame(data_dict)
    
    # Create DataLoader.
    seq_len = 10  # Use a sliding window of 10 time steps.
    batch_size = 16
    dataloader = create_powergrid_dataloader(df, seq_len=seq_len, batch_size=batch_size)

    # Update config for our TimeSeriesTransformer.
    # (Make sure to use 'context_lenght' as in the config.)
    config = YOR_GPT_CONFIG_124M

    # Instantiate the model.
    model = TimeSeriesTransformer(config).to(device)

    # Loss function and optimizer.
    criterion = nn.CrossEntropyLoss()  # For classification of fault types.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model.
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=5)