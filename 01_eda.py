"""
Exploratory Data Analysis for the project:
"A Self-Healing Machine Learning System for Predicting Startup Success and Failure in India"

This script supports Chapter 4 (Exploratory Data Analysis) of the project report.
"""

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
