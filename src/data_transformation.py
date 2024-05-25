from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import data_ingestion  # Import data_ingestion script
from joblib import dump

# Call the function read_data()
df = data_ingestion.read_data()

def preprocess_data(df):
    # Split columns into categorical features and numerical features
    cat_cols = df.columns[df.dtypes == 'object']
    num_cols = df.columns[df.dtypes != 'object']
    print('Categorical features:', cat_cols)
    print('Numerical features:', num_cols)

    # Feature encoding: Convert categorical variables into numerical representations
    le_gender = LabelEncoder()
    le_race_ethnicity = LabelEncoder()
    le_parental_level_of_education = LabelEncoder()
    le_lunch = LabelEncoder()
    le_test_preparation_course = LabelEncoder()
    
    df['gender'] = le_gender.fit_transform(df['gender'])
    df['race_ethnicity'] = le_race_ethnicity.fit_transform(df['race_ethnicity'])
    df['parental_level_of_education'] = le_parental_level_of_education.fit_transform(df['parental_level_of_education'])
    df['lunch'] = le_lunch.fit_transform(df['lunch'])
    df['test_preparation_course'] = le_test_preparation_course.fit_transform(df['test_preparation_course'])
    
    # Save the label encoders
    dump(le_gender, 'le_gender.joblib')
    dump(le_race_ethnicity, 'le_race_ethnicity.joblib')
    dump(le_parental_level_of_education, 'le_parental_level_of_education.joblib')
    dump(le_lunch, 'le_lunch.joblib')
    dump(le_test_preparation_course, 'le_test_preparation_course.joblib')

    # Split data into independent and target variables
    X=df.drop('math_score',axis=1)
    y = df['math_score']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    dump(scaler, 'scaler.joblib')

    return X_train_scaled, X_test_scaled, y_train, y_test
