import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd

from dataset import DatasetLoader
from dataset_encode import DataEncoder  # Assuming you have this class already


class DataSplitter:
    def __init__(
        self,
        X=None,
        y=None,
        csv_file=None,
        target_column="Attack Type",
        scale_features=True,
        test_size=0.2,
        batch_size=64,
    ):
        self.X = X
        self.y = y
        self.csv_file = csv_file
        self.target_column = target_column
        self.scale_features = scale_features
        self.test_size = test_size
        self.batch_size = batch_size

        self.train_loader = None
        self.val_loader = None
        self.input_dim = None
        self.output_dim = None

        self._prepare()

    def _prepare(self):
        if self.X is None or self.y is None:
            # Step 1: Load and encode data
            loader = DatasetLoader(self.csv_file, window_size=5, stride=1)
            X_df, y_df = loader.get_data_target()

            self.X = X_df  # shape: [samples, time_steps, features]
            self.y = y_df  # shape: [samples]

        # Step 2: Convert to torch tensors (X should already be 3D)
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(
            self.y.values if hasattr(self.y, "values") else self.y, dtype=torch.long
        )

        # Step 3: Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=self.test_size, random_state=42
        )

        # Step 4: Wrap in DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Step 5: Set input/output dimensions
        self.input_dim = X_tensor.shape[2]  # features per timestep
        self.output_dim = len(torch.unique(y_tensor))

    def get_loaders(self):
        return self.train_loader, self.val_loader

    def get_input_output_dims(self):
        return self.input_dim, self.output_dim
