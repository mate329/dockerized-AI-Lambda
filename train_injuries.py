import numpy as np 
import pandas as pd 
import optuna
import pickle
import joblib
import json
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

optuna.logging.set_verbosity(optuna.logging.INFO)

data_train_x = pd.read_csv('data/kaggle_x_train.csv')
data_train_y = pd.read_csv('data/kaggle_y_train.csv')

merged_train = pd.merge(data_train_x, data_train_y[['Id', 'injury_duration']], on='Id')
merged_train.drop(columns=['Id'], inplace=True)

# Separate features and target variable
X = merged_train.drop('injury_duration', axis=1)
y = merged_train['injury_duration']

# Save feature names for Lambda validation
feature_names = list(X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    # Define hyperparameter space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1, 100),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.3)
    }
    
    # Create and fit model
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Make predictions and return RMSE
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    return rmse

# Minimize RMSE
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

# Fetch the best hyperparameters found during the study
best_params = study.best_params
print(f"Best parameters: {best_params}")
print(f"Best RMSE: {study.best_value}")

# Train final model with best parameters and evaluate
final_model = XGBRegressor(**best_params)
final_model.fit(X_train, y_train)
final_preds = final_model.predict(X_test)

print("Final model performance:")
print(f"Best parameters: {best_params}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, final_preds)):.4f}")

# Export artifacts for AWS Lambda
print("\nExporting model artifacts...")

# 1. Save the trained model
joblib.dump(final_model, 'PredictInjuryDurationLambda/injury_model.pkl')
print("Model saved as 'PredictInjuryDurationLambda/injury_model.pkl'")

# 2. Save model metadata
metadata = {
    'feature_names': feature_names,
    'best_params': best_params,
    'model_performance': {
        'rmse': float(np.sqrt(mean_squared_error(y_test, final_preds)))
    },
    'training_date': pd.Timestamp.now().isoformat(),
    'n_features': len(feature_names)
}

with open('PredictInjuryDurationLambda/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("Metadata saved as 'PredictInjuryDurationLambda/model_metadata.json'")

