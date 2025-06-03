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
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.data = None
        self.target = None

    def load_data(self):
        self.df = pd.read_csv(self.csv_file)
        print(self.df.shape)
        self._process_data()
        return self.df

    def _process_data(self):
        # drop columns
        columns_to_drop = [
            "Timestamp",
            "User Information",
            "Device Information",
            "Alerts/Warnings",
            "Log Source",
        ]
        data_c = self.df.drop(columns=columns_to_drop)
        data_c = data_c.dropna()

        self.target = data_c["Attack Type"]
        self.data = data_c.drop(columns=["Attack Type"])

    def get_data_target(self):
        if self.data is None or self.target is None:
            self.load_data()
        return self.data, self.target


"""
csv_file = "../data/raw/cybersecurity_attacks_v1.0.csv"
loader = DatasetLoader(csv_file)
loader.load_data()
print(loader.data.dtypes)
print(loader.target)
"""
