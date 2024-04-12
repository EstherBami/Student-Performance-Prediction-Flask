import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def model_evaluation(true,predicted):
    mae = mean_absolute_error(true,predicted)
    mse = mean_squared_error(true,predicted)
    rmse = np.sqrt(mean_squared_error(true,predicted))
    r2 = r2_score(true,predicted)
    return mae, rmse, r2

    # Train the model
    model.fit(X_train,y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate Train and Test datasets
    model_train_mae, model_train_rmse, model_train_r2 = model_evaluation(y_train, y_train_pred)
    model_test_mae, model_test_rmse, model_test_r2 = model_evaluation(y_test, y_test_pred)
    
    # Model performance on Train set
    print('Model performance on Train set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))
    
    # Model performance on Test set
    print('----------------------------------')
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    
    print('=' * 35)
    print('\n')