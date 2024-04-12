import pandas as pd
import numpy as np


# Display the first few rows of the dataframe
print(df.head())

# Get information about the dataframe
print(df.info())

# Statistical summary of numerical features
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Check for duplicates rows
print(df.duplicated().sum())

outlier