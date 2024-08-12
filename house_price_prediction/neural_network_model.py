# neural_network_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import numpy as np
import pandas as pd
import pickle

# Load preprocessed data
from data_preprocessing import X_train_scaled, y_train, X_test_scaled, y_test

# Function to build and compile the neural network model
def build_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Initialize and train the model
try:
    model = build_model(X_train_scaled.shape[1])
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1  # Show progress during training
    )
except Exception as e:
    print(f"An error occurred during model training: {e}")
    raise

# Save the model
try:
    model.save('models/housing_nn_model.h5')
    print("Neural network model training complete and saved as 'housing_nn_model.h5'.")
except Exception as e:
    print(f"An error occurred while saving the model: {e}")
    raise

# Optionally, save the training history for further analysis
try:
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('models/training_history.csv', index=False)
    print("Training history saved as 'training_history.csv'.")
except Exception as e:
    print(f"An error occurred while saving the training history: {e}")
    raise
