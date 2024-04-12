import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# MODEL: GRADIENT BOOSTING REGRESSOR

# Initialize the model
gradient_reg = GradientBoostingRegressor(random_state=42)

# Train the model with the training set
gb_model = gradient_reg.fit(X_train,y_train)

# Make predictions
y_pred = gb_model.predict(X_test)

# Print predictions
print(y_pred[:5])

# Print test set to compare with the predictions
print(y_test[:5])

