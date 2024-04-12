import matplotlib.pyplot as pyplot
import seaborn as sns
%matplotlib inline

# Plot distribution of target variable
def data_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='math_score')
    plt.title('Distribution of Target Variable')
    plt.xlabel('Math_score')
    plt.ylabel('Count')

# Plot correlation matrix
def correlation_matrix(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True)
    plt.title('Correlation Matrix')

# Plot model performance
