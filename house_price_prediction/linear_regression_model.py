# linear_regression_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load data
from data_preprocessing import X_train_scaled, X_test_scaled, y_train, y_test

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f"Training Mean Squared Error: {train_mse}")
print(f"Testing Mean Squared Error: {test_mse}")

# Save the model
with open('models/housing_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Linear regression model training complete and saved as 'housing_model.pkl'.")
