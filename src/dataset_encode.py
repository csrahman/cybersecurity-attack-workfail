from dataset import DatasetLoader

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

class DataEncoder:
    def __init__(self, target_column="Attack Type", scale_features=True):
        self.target_column = target_column
        self.scale_features = scale_features
        self.label_encoders = {}  # For each categorical column
        self.target_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def encode(self, df):
        # --- Step 1: Convert textual payload to numerical feature ---
        if "Payload Data" in df.columns:
            df["Payload Length"] = df["Payload Data"].apply(lambda x: len(str(x)))
            df.drop(columns=["Payload Data"], inplace=True)

        # --- Step 2: Encode categorical features ---
        for col in df.select_dtypes(include=["object"]).columns:
            if col != self.target_column:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # --- Step 3: Encode target variable ---
        df[self.target_column] = self.target_encoder.fit_transform(
            df[self.target_column]
        )

        # --- Step 4: Split features and target ---
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # --- Step 5: Scale features (optional) ---
        if self.scale_features:
            X = self.scaler.fit_transform(X)

        return X, y


"""
csv_file = "../data/raw/cybersecurity_attacks_v1.0.csv"
loader = DatasetLoader(csv_file)
df = loader.load_data()

de = DataEncoder(target_column="Attack Type", scale_features=True)
X, y = de.encode(df)
"""
