import joblib
import pandas as pd
import numpy as np

try:
    print("Loading models...")
    log_model = joblib.load('logistic_regression.pkl')
    dt_model = joblib.load('decision_tree.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoder.pkl')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Failed to load models: {e}")
    exit(1)

print("Creating test data...")
test_data = {
    'PlayerID': [10001, 10002],
    'Age': [25, 30],
    'Gender': ['Male', 'Female'],
    'Location': ['USA', 'Europe'],
    'GameGenre': ['Action', 'Strategy'],
    'PlayTimeHours': [5.5, 12.0],
    'InGamePurchases': [1, 0],
    'GameDifficulty': ['Medium', 'Hard'],
    'SessionsPerWeek': [3, 10],
    'AvgSessionDurationMinutes': [45, 90],
    'PlayerLevel': [20, 50],
    'AchievementsUnlocked': [5, 25]
}
df = pd.DataFrame(test_data)

print("Preprocessing test data...")
try:
    df_processed = df.copy()
    cat_features = ['Gender', 'Location', 'GameGenre', 'GameDifficulty']
    for col in cat_features:
        le = label_encoders[col]
        df_processed.loc[~df_processed[col].isin(le.classes_), col] = le.classes_[0]
        df_processed[col] = le.transform(df_processed[col])
    
    required_cols = ['Age', 'Gender', 'Location', 'GameGenre', 'PlayTimeHours', 
                     'InGamePurchases', 'GameDifficulty', 'SessionsPerWeek', 
                     'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
    df_processed = df_processed[required_cols]

    X_scaled = scaler.transform(df_processed)
    X_unscaled = df_processed

    print("Running predictions...")
    log_pred = log_model.predict(X_scaled)
    dt_pred = dt_model.predict(X_unscaled)

    print(f"Logistic Regression Predictions: {log_pred}")
    print(f"Decision Tree Predictions: {dt_pred}")

    print("Verification Successful!")
except Exception as e:
    print(f"Verification Failed: {e}")
    exit(1)
