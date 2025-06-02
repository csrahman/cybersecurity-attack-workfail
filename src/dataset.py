import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# pip install -r requirements.txt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models, datasets
import os

csv_file = "../data/raw/cybersecurity_attacks_v1.0.csv"


class DatasetLoader:
    def __init__(self, csv_file, window_size=5, stride=1):
        self.csv_file = csv_file
        self.window_size = window_size
        self.stride = stride
        self.df = None
        self.sequences = []
        self.labels = []

    def load_data(self):
        self.df = pd.read_csv(self.csv_file)
        print("Raw shape:", self.df.shape)
        self._process_data()
        return self.sequences, self.labels

    def _process_data(self):
        df = self.df.copy()
        df = df.dropna()

        # Keep necessary columns for sequence logic
        keep_cols = ["Source IP Address", "Timestamp", "Attack Type"]
        feature_cols = [
            col
            for col in df.columns
            if col
            not in keep_cols
            + [
                "User Information",
                "Device Information",
                "Alerts/Warnings",
                "Log Source",
            ]
        ]

        df = df[keep_cols + feature_cols]
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        # Group by 'Source IP'
        for source_ip, group in df.groupby("Source IP Address"):
            group = group.sort_values("Timestamp")

            features = group[feature_cols].values
            targets = group["Attack Type"].values

            for i in range(0, len(group) - self.window_size + 1, self.stride):
                window = features[i : i + self.window_size]
                label = targets[
                    i + self.window_size - 1
                ]  # label of last item in window

                self.sequences.append(window)
                self.labels.append(label)

        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)

    def get_data_target(self):
        if self.df is None or not self.sequences:
            self.load_data()
        return self.sequences, self.labels


# class DatasetLoader:
#     def __init__(self, csv_file):
#         self.csv_file = csv_file
#         self.df = None
#         self.data = None
#         self.target = None

#     def load_data(self):
#         self.df = pd.read_csv(self.csv_file)
#         print(self.df.shape)
#         self._process_data()
#         return self.df

#     def _process_data(self):
#         # drop columns
#         columns_to_drop = [
#             "Timestamp",
#             "User Information",
#             "Device Information",
#             "Alerts/Warnings",
#             "Log Source",
#         ]
#         data_c = self.df.drop(columns=columns_to_drop)
#         data_c = data_c.dropna()

#         self.target = data_c["Attack Type"]
#         self.data = data_c.drop(columns=["Attack Type"])

#     def get_data_target(self):
#         if self.data is None or self.target is None:
#             self.load_data()
#         return self.data, self.target


"""
csv_file = "../data/raw/cybersecurity_attacks_v1.0.csv"
loader = DatasetLoader(csv_file)
loader.load_data()
print(loader.data.dtypes)
print(loader.target)
"""
