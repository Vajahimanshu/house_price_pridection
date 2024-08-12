from sklearn.metrics import mean_squared_error, r2_score
import joblib
from data_preparation import load_and_preprocess_data
from sklearn.model_selection import train_test_split

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')

if __name__ == "__main__":
    # Load and preprocess data
    X, y = load_and_preprocess_data(r'E:\download\elevate\house_price_prediction\housing_data.csv')
    
    # Load the model
    model = joblib.load('house_price_model.pkl')
    
    # Split the data for evaluation
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
