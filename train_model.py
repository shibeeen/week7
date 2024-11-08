import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Use RandomForestClassifier for classification tasks
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load dataset
data = pd.read_csv('heart.csv')

# Data preprocessing (encode categorical variables)
data['thal'] = data['thal'].astype('category').cat.codes  # Encode 'thal'
data['cp'] = data['cp'].astype('category').cat.codes      # Encode 'cp'

# Prepare feature and target variables
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]  # Feature columns
y = data['target']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model for classification
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n {classification_report(y_test, y_pred)}")

# Save the model
joblib.dump(model, 'heart_disease_model.pkl')
