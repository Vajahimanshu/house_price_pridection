import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

# Load the data
data = pd.read_csv('E:/download/elevate/house_price_prediction/house_prices.csv')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
data = pd.get_dummies(data)

# Normalize numerical features
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Split the data
X = data_scaled.drop('price', axis=1)
y = data_scaled['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the TensorFlow model
model_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model_tf.compile(optimizer='adam', loss='mse')

# Train the model
model_tf.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Save the trained model
model_path = 'E:/download/elevate/house_price_prediction/house_price_model.h5'
model_tf.save(model_path)

print(f"Model saved to {model_path}")
