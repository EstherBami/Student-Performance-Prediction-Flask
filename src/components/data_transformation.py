import pandas as pd
from data_ingestion import read_data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Read data from data_ingestion file
df = read_data()
print(df)

# Split columns into categorical features and numerical features
cat_cols = df.columns[df.dtypes=='object']
num_cols = df.columns[df.dtypes!='object']
print('Categorical features:', cat_cols)
print('Numerical features:', num_cols)

# Feature encoding: Convert categorical variables into numerical representations
le=LabelEncoder()
df['gender']=le.fit_transform(df['gender'])
df['race_ethnicity']=le.fit_transform(df['race_ethnicity'])
df['parental_level_of_education']=le.fit_transform(df['parental_level_of_education'])
df['lunch']=le.fit_transform(df['lunch'])
df['test_preparation_course']=le.fit_transform(df['test_preparation_course'])

# Split data into features and target
X = df.drop('math_score', axis=1)  #Independent features
y = df['math_score']               #Target features

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)