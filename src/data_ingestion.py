"""
Load dataset into the working directory
"""
import pandas as pd 

def read_data():
    df = pd.read_csv("data\stud.csv")
    return df

df = read_data()

if __name__ == "__main__":
    print(df.head())