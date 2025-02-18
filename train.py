import numpy as np
import torch
import torch.nn as nn
from config import YOR_GPT_CONFIG_124M
import pandas as pd
from v2_preprocessing import create_powergrid_dataloader
from GPT import TimeSeriesTransformer
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -------------------------------
# Training Loop
# -------------------------------
def train_model(model, dataloader, optimizer, device, num_epochs=5, alpha=1.0):
    """
    model: your TimeSeriesTransformer with two heads.
    dataloader: yields (inputs, (labels_cls, labels_reg))
    optimizer: e.g., Adam
    device: 'cuda' or 'cpu'
    num_epochs: number of training epochs
    alpha: weighting factor for the regression loss
    """
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    model.train()
    # Open a file to store training logs
    with open("training_log.txt", "w") as f:
        for epoch in range(num_epochs):
            running_loss = 0.0
            total_cls = 0
            correct_cls = 0
            total_reg_loss = 0.0
            total_samples = 0

            for i, (inputs, (labels_cls, labels_reg)) in enumerate(dataloader):
                inputs    = inputs.to(device)           # (batch_size, seq_len, feature_dim)
                labels_cls = labels_cls.to(device)      # (batch_size,)
                labels_reg = labels_reg.to(device)      # (batch_size,)
                
                optimizer.zero_grad()
                # Forward pass (two outputs)
                fault_logits, ttf_pred = model(inputs)

                # Classification loss
                loss_cls = criterion_cls(fault_logits, labels_cls)
                
                # Regression loss
                loss_reg = criterion_reg(ttf_pred.squeeze(), labels_reg)

                # Combined loss
                loss = loss_cls + alpha * loss_reg

                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                # Classification Accuracy Calculation
                _, predicted_cls = torch.max(fault_logits, dim=1)
                total_samples += labels_cls.size(0)
                correct_cls += (predicted_cls == labels_cls).sum().item()

                # Track Regression Loss (MSE)
                total_reg_loss += loss_reg.item()


                # if (i + 1) % 10 == 0:
                #     print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                #         f"Loss: {loss.item():.4f} (CE: {loss_cls.item():.4f}, MSE: {loss_reg.item():.4f})")
                

            # Compute epoch-level accuracy and average loss
            avg_loss = running_loss / len(dataloader)
            accuracy = correct_cls / total_samples
            avg_reg_loss = total_reg_loss / len(dataloader)

            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Avg Loss: {avg_loss:.4f}, "
                f"Accuracy: {accuracy:.4f}, "
                f"Regression MSE: {avg_reg_loss:.4f}")
            
            # Write to file
            f.write(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Regression MSE: {avg_reg_loss:.4f}\n")
        print("Training complete.")

# -------------------------------
# Testing the Training Loop on Dummy Data
# -------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------------
    # 1. Load CSV
    # -------------------------------
    df = pd.read_csv('../data/power_dataset.csv')
    df = df.head(500)  # For testing purposes
    df['time_to_failure'] = np.log1p(df['time_to_failure'])  # log(1 + ttf)

    # '"C:\Users\HP\Documents\Area Of Self Development\Competition\Innovation Competition\data\power_dataset.csv"'
   
    # -------------------------------
    # 2. Train/Test Split
    # -------------------------------
    train_df = df.iloc[:450].copy()  # Example
    test_df  = df.iloc[50:].copy()  # Example

    # 3) Identify the columns to scale
    fault_col = 'fault_type'
    ttf_col   = 'time_to_failure'
    timestamp_col = 'timestamp'

    # These are the numeric columns you want to scale.
    # Exclude the fault_col, ttf_col, and timestamp.
    feature_cols = df.columns.difference([fault_col, ttf_col, timestamp_col])


    # 4) Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    # 5) Transform both train & test
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    print(train_df.head())
    
    # -------------------------------
    # 5. Create Dataloaders
    # -------------------------------
    seq_len = 50
    batch_size = 16
    train_dataloader = create_powergrid_dataloader(train_df, seq_len=seq_len, batch_size=batch_size, shuffle=True)
    test_dataloader  = create_powergrid_dataloader(test_df,  seq_len=seq_len, batch_size=batch_size, shuffle=False)


    # -------------------------------
    # 4. Instantiate Model + Setup
    # -------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    config = YOR_GPT_CONFIG_124M

    
    model = TimeSeriesTransformer(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    # -------------------------------
    # 5. Train the Model
    # -------------------------------
    num_epochs = 50
    train_model(model, train_dataloader, optimizer, device, num_epochs=num_epochs, alpha=1.0)

    # -------------------------------
    # 6. Evaluate on Test Data
    # -------------------------------
    model.eval()
    correct_cls = 0
    total_cls = 0
    mse_accum = 0.0
    count_batches = 0
    criterion_reg = nn.MSELoss()

    with torch.no_grad():
        for inputs, (labels_cls, labels_reg) in test_dataloader:
            inputs     = inputs.to(device)
            labels_cls = labels_cls.to(device)
            labels_reg = labels_reg.to(device)
            
            fault_logits, ttf_pred = model(inputs)
            
            # Classification accuracy
            _, predicted_cls = torch.max(fault_logits, dim=1)
            total_cls += labels_cls.size(0)
            correct_cls += (predicted_cls == labels_cls).sum().item()
            
            # Regression MSE
            mse_accum += criterion_reg(ttf_pred.squeeze(), labels_reg).item()
            count_batches += 1

    avg_mse = mse_accum / count_batches
    accuracy = correct_cls / total_cls
    print(f"Test Classification Accuracy: {accuracy:.4f}")
    print(f"Test Regression MSE:         {avg_mse:.4f}")

    # -------------------------------
    # 7. Save Model
    # -------------------------------
    model_path = 'models/model_1.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")