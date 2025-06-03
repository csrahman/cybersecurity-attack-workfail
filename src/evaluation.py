from dataset import DatasetLoader
from dataset_encode import DataEncoder
from feature import FeatureSelectorEvaluator
from model import CNN_LSTM_Model

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader




def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))
    return acc, f1


# X, y = encoder.encode(pd.concat([X_df, y_df], axis=1))

# # Convert to torch tensors
# X_tensor = torch.tensor(X, dtype=torch.float32)  # Features as floats
# y_tensor = torch.tensor(
#     y.values if hasattr(y, "values") else y, dtype=torch.long
# )  # Target as longs for classification
# X_tensor = X_tensor.unsqueeze(1)  # shape becomes (num_samples, 1, num_features)

# dataset = TensorDataset(X_tensor, y_tensor)
# batch_size = 64

# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# input_dim = X.shape[1]  # number of features
# output_dim = len(torch.unique(y_tensor))  # number of classes

# model = CNN_LSTM_Model(input_dim=input_dim, output_dim=output_dim)


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 20

# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0

#     for batch_X, batch_y in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")
# input_dim = X.shape[1]
# model = CNN_LSTM_Model(input_dim)
# # Example usage:
# evaluate(model, val_loader)

# torch.save(model.state_dict(), "cnn_lstm_model.pth")
# print("Model saved to cnn_lstm_model.pth")
