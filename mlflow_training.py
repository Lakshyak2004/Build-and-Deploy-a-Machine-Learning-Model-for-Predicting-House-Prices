# mlflow_training.py

import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Set MLflow experiment name
mlflow.set_experiment("HousePricePrediction")

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()
df['MedHouseVal'] = housing.target

# Scale numerical features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log parameters and metrics
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    
    # Log the model with MLflow
    mlflow.sklearn.log_model(best_rf, "model")
    
    # Save and log the scaler as an artifact
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    mlflow.log_artifact("scaler.pkl")
    
    # Also save the model to disk for deployment purposes
    with open("trained_house_price_model.pkl", "wb") as f:
        pickle.dump(best_rf, f)
    
    print("Best parameters:", grid_search.best_params_)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)
