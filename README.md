# Build-and-Deploy-a-Machine-Learning-Model-for-Predicting-House-Prices
1. Overview
This project builds and deploys a machine learning model to predict house prices using the California Housing dataset. The solution includes data preprocessing, model training with hyperparameter tuning, and deployment as a REST API using Flask. Bonus features such as logging, MLflow integration, and a simple frontend UI are also implemented.

2. Data Preprocessing & Feature Engineering
Steps Taken:
Dataset Loading & Initial EDA:
The California Housing dataset is loaded using scikit-learn's fetch_california_housing(as_frame=True). Initial exploratory data analysis (EDA) is performed to understand the data structure, distributions, and detect any missing values.

Missing Values Handling:
Missing numeric values are handled by replacing them with the median value. Although the California Housing dataset contains no missing values, this step is included to generalize the approach.

Feature Engineering:

Scaling: Numerical features are scaled using StandardScaler to normalize the data, which is essential for many machine learning algorithms.
Encoding: Although no categorical features exist in this dataset, the code includes a demonstration of one-hot encoding for any categorical variables.
Feature Selection: Features with an absolute correlation (|correlation| > 0.1) with the target variable are selected to improve model performance and reduce noise.
The well-commented code for these steps is provided in data_preprocessing_updated.py.

3. Model Selection & Optimization
Approach:
Baseline Model:
A simple Linear Regression model is trained as a baseline to evaluate the initial performance.

Advanced Model:
A Random Forest Regressor is chosen for its robustness. Hyperparameter tuning is performed using GridSearchCV to identify the best parameters across a range of values (e.g., n_estimators, max_depth, min_samples_split).

Evaluation Metrics:
The models are evaluated using:

RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
R² Score (Coefficient of Determination)
Model Persistence:
The best performing model is saved using Pickle (trained_house_price_model.pkl) along with the scaler (scaler.pkl) for use during deployment.

The training and optimization code is provided in model_training.py and an MLflow tracking script in mlflow_training.py for model versioning.

4. Deployment Strategy & API Usage
API Deployment:
Flask API:
The model is deployed using a Flask web application that exposes a /predict endpoint. This endpoint:

Accepts JSON input with a key "features" (a list of feature values).
Applies the saved scaler to the input data.
Returns the predicted house price from the trained model.
Enhanced Logging & Error Handling:
Robust logging (using Python’s logging module and a Rotating File Handler) is implemented to capture API calls and errors. Detailed error handling ensures that meaningful error messages are returned.

Frontend UI:
A simple HTML page (index.html) allows users to input feature values and see the predicted price without using external tools like Postman or CURL.

Containerization:
A Dockerfile is provided to containerize the Flask application. This enables easier deployment across different environments, including cloud platforms like AWS, GCP, Azure, or Render.

API Testing:
Using CURL:
bash
Copy
Edit
curl -X POST -H "Content-Type: application/json" \
-d '{"features": [8.3252,41.0,6.9841,1.0238,322.0,2.5556,37.88,-122.23]}' \
http://127.0.0.1:5000/predict
Using the Frontend UI:
Open index.html in a browser, input comma-separated feature values, and click the Predict button to view the result.
The complete deployment code is available in app.py, and containerization instructions along with the Dockerfile are included in the repository.

5. Code Repository Structure
The GitHub repository (or Jupyter Notebook) includes the following files:

graphql
Copy
Edit
├── data_preprocessing.py     # Data loading, EDA, and feature engineering
├── model_training.py                  # Model training, evaluation, and saving using Pickle
├── mlflow_training.py                 # MLflow integration for model versioning
├── app.py                             # Flask API with enhanced logging and error handling
├── index.html                         # Simple frontend UI for interacting with the API
├── ML_Project_Dockerfile              # Dockerfile for containerizing the application
├── ML_Project_requirements.txt        # Required Python packages
└── README.md                          # This project report and documentation
Each script includes detailed inline comments to explain every step of the process, ensuring clarity and maintainability.

6. Conclusion
This project demonstrates an end-to-end solution for predicting house prices using machine learning, covering:

Data preprocessing and feature engineering
Model selection, hyperparameter tuning, and evaluation
Deployment of a robust API with logging and error handling
Additional bonus features like MLflow for model tracking and a simple frontend UI
The complete, well-documented code can be found in the repository, which is structured to facilitate both development and deployment.
