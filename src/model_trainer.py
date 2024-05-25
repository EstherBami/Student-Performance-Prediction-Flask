import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import data_transformation  # Import data_transformation script
import data_ingestion  # Import data_ingestion script
from joblib import dump

# Call the function read_data()
df = data_ingestion.read_data()

# Call the function preprocess_data() and pass the DataFrame as an argument
X_train_scaled, X_test_scaled, y_train, y_test = data_transformation.preprocess_data(df)

# MODEL: GRADIENT BOOSTING REGRESSOR

# Initialize the model
regressor = GradientBoostingRegressor(random_state=42)

# Train the model with the training set
model = regressor.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print evaluation metrics
print("R^2 Score:", r2)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)

# Save the model
dump(model, 'model.joblib')
