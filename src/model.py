from dataset import DatasetLoader
from dataset_encode import DataEncoder
from feature import FeatureSelectorEvaluator
from data_split import DataSplitter

import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report


class CNN_LSTM_Model(nn.Module):
    def __init__(
        self,
        input_dim,
        cnn_channels=32,
        lstm_hidden_dim=64,
        lstm_layers=1,
        output_dim=3,
    ):
        super(CNN_LSTM_Model, self).__init__()

        # 1D CNN layer
        self.cnn = nn.Conv1d(
            in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # Fully connected output layer
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        # Transpose to (batch_size, features, seq_len) for Conv1d
        x = x.transpose(1, 2)

        cnn_out = self.relu(self.cnn(x))  # shape: (batch_size, cnn_channels, seq_len)

        # Transpose back for LSTM: (batch_size, seq_len, cnn_channels)
        cnn_out = cnn_out.transpose(1, 2)

        lstm_out, _ = self.lstm(
            cnn_out
        )  # shape: (batch_size, seq_len, lstm_hidden_dim)

        # Use last timestep output for classification
        last_out = lstm_out[:, -1, :]

        out = self.fc(last_out)

        return out


def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            print(batch_X.shape, batch_y.shape)  # Check shapes
            outputs = model(X_batch)
            print(outputs.shape)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))
    return acc, f1


"""

csv_file = "../data/raw/cybersecurity_attacks_v1.0.csv"
loader = DatasetLoader(csv_file)
X_df, y_df = loader.load_data()
encoder = DataEncoder(target_column="Attack Type", scale_features=True)

X, y = encoder.encode(pd.concat([X_df, y_df], axis=1))
import torch

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)  # Features as floats
y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.long)  # Target as longs for classification
X_tensor = X_tensor.unsqueeze(1)  # shape becomes (num_samples, 1, num_features)
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_tensor, y_tensor)
batch_size = 64

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

input_dim = X.shape[1]  # number of features
output_dim = len(torch.unique(y_tensor))  # number of classes

model = CNN_LSTM_Model(input_dim=input_dim, output_dim=output_dim)

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
"""

"""
df = pd.read_csv("../data/raw/cybersecurity_attacks_v1.0.csv")

# Step 2: Encode features
encoder = DataEncoder()
X, y = encoder.encode(df)

====# Step 3: Reattach Source IP and Timestamp for windowing logic
X = pd.DataFrame(X)
X["Source IP Address"] = df["Source IP Address"].values
X["Timestamp"] = df["Timestamp"].values
X["Attack Type"] = y

# Step 4: Save processed DataFrame to a temporary CSV and use DatasetLoader
X.to_csv("../data/processed/processed_temp.csv", index=False)
print("saved successfully")
loader = DatasetLoader("../data/processed/processed_temp.csv", window_size=5, stride=1)
sequences, labels = loader.get_data_target()===

splitter = DataSplitter("../data/processed/processed_temp.csv)
train_loader, val_loader = splitter.get_loaders()
input_dim, output_dim = splitter.get_input_output_dims()

model = CNN_LSTM_Model(input_dim=input_dim, output_dim=output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

"""
