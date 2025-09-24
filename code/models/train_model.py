import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    X = np.random.randn(n_samples, 4)
    y = (X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3 + X[:, 3] * 0.2 + np.random.randn(n_samples) * 0.1) > 0
    
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y.astype(int)
    
    return df, feature_names

def train_model():
    """Train and save the machine learning model"""
    print("Creating sample data...")
    df, feature_names = create_sample_data()
    
    # Split data
    X = df[feature_names]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save model
    model_path = os.path.join('models', 'random_forest_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature names
    feature_info = {
        'feature_names': feature_names,
        'model_accuracy': accuracy
    }
    joblib.dump(feature_info, os.path.join('models', 'feature_info.pkl'))
    
    return model, feature_names, accuracy

if __name__ == "__main__":
    train_model()