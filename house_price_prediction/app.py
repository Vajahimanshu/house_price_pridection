from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os

# Load the trained model
model_tf = tf.keras.models.load_model('E:/download/elevate/house_price_prediction/house_price_model.h5')

# Initialize the Flask app with the template folder path
app = Flask(__name__, template_folder='E:/download/elevate/house_price_prediction')

# Load the scaler
scaler = StandardScaler()
data = pd.read_csv('E:/download/elevate/house_price_prediction/house_prices.csv')
data.ffill(inplace=True)
data = pd.get_dummies(data)
scaler.fit(data.drop('price', axis=1))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received data:", data)  # Debug print
    input_data = pd.DataFrame(data, index=[0])
    input_data_scaled = scaler.transform(input_data)
    prediction = model_tf.predict(input_data_scaled)
    print("Prediction:", prediction)  # Debug print
    
    # Calculate total price (example calculation)
    size = data.get('size', 0)
    price_per_sqft = 500  # Example price per square foot
    total_price = size * price_per_sqft + prediction[0][0]
    
    return jsonify({'prediction': prediction[0][0], 'total_price': total_price})

if __name__ == '__main__':
    app.run(debug=True)
