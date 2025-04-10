import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data_url = 'https://raw.githubusercontent.com/ananyavj/CareerPredictionDataset/refs/heads/main/Data_final.csv'
data = pd.read_csv(data_url)

# Encode Career labels
le = LabelEncoder()
data['Career_Label'] = le.fit_transform(data['Career'])

# Separate features and target
X = data.drop(columns=['Career', 'Career_Label'])
y = data['Career_Label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and preprocessors
joblib.dump(model, "career_prediction_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(le, "label_encoder.joblib")

print("✅ Model and files saved successfully!")
