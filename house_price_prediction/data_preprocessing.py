import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('E:/download/elevate/house_price_prediction/house_prices.csv')

# Handle missing values
data.ffill(inplace=True)

# Encode categorical variables
data = pd.get_dummies(data)

# Normalize numerical features
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Split the data
X = data_scaled.drop('price', axis=1)
y = data_scaled['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
