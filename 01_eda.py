# Exploratory Data Analysis for Startup Dataset

import pandas as pd

# Load dataset
data = pd.read_csv("startup_data.csv")

# Basic overview
print(data.head())
print(data.info())
print(data.describe())

# Status distribution
print(data["Status"].value_counts())

# City-wise distribution
print(data["City"].value_counts())

# Sector-wise distribution
print(data["Sector"].value_counts())
