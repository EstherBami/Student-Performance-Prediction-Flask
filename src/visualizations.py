import matplotlib.pyplot as plt
import seaborn as sns

# Plot distribution of target variable
def data_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='math_score', palette='viridis')
    plt.title('Distribution of Target Variable')
    plt.xlabel('Math Score')
    plt.ylabel('Count')
    plt.show()

# Plot correlation matrix
def correlation_matrix(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Call the function read_data() from data_ingestion module
    from scripts import data_ingestion
    df = data_ingestion.read_data()
    
    # Call the plotting functions
    data_distribution(df)
    correlation_matrix(df)
