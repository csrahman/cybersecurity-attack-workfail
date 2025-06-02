import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import DatasetLoader
from feature import DataPreprocessor

csv_file = "../data/raw/cybersecurity_attacks_v1.0.csv"
d_loader = DatasetLoader(csv_file)

# def checking_outlier():
# data, label = d_loader.get_data_target()
# numerical_columns = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
# len(numerical_columns)


# # Number of numerical columns
# num_columns = len(numerical_columns)

# # Calculate the number of rows and columns needed for the subplots
# num_cols = 2  # Fixed number of columns
# num_rows = math.ceil(num_columns / num_cols)  # Calculate rows needed

# plt.figure(figsize=(18, num_rows * 3))

# sns.set_palette("husl")
# sns.set(style="whitegrid")

# for i, col in enumerate(numerical_columns, 1):
#     plt.subplot(num_rows, num_cols, i)
#     sns.boxplot(x=data[col], color="skyblue", width=0.5)
#     plt.title(col)
#     plt.xlabel("")

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

data, target = d_loader.get_data_target()
data_m = pd.concat([data, target], axis=1)
data_m = data_m.drop(columns="Timestamp", axis=1)


# def pie_bar_plot(df, col):
#     plt.figure(figsize=(10, 6))

#     # Extract value counts for the specified column
#     value_counts = df[col].value_counts().sort_index()

#     ax1 = value_counts
#     plt.title(f"Distribution by {col}", fontweight="black", size=14, pad=15)
#     colors = sns.color_palette("Set2", len(ax1))
#     plt.pie(ax1.values, labels=None, autopct="", startangle=90, colors=colors)
#     center_circle = plt.Circle((0, 0), 0.4, fc="white")
#     fig = plt.gcf()
#     fig.gca().add_artist(center_circle)
#     plt.show()


# pie_bar_plot(data_m, "Attack Type")

plt.figure(figsize=(40, 20))
plt.title("Correlation Plot")
sns.heatmap(data_m.corr(), cmap="YlGnBu")
