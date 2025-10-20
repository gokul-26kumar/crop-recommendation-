# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Step 1: Data Preprocessing
# Encode the target variable (crop types) into numeric form
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Split features and labels
X = data.drop(columns=['label'])  # Features (N, P, K, temperature, humidity, ph, rainfall)
y = data['label']                # Target (crop types)

# Standardize the features to bring them on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 3: Model Selection and Training (Random Forest Classifier)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 4: Model Evaluation
# Predict the crop types for the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Display the confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 5: Prediction (Example)
# Sample input data for prediction
sample_data = [[90, 40, 42, 25.5, 82.5, 6.8, 250]]  # Example: N, P, K, temperature, humidity, ph, rainfall
sample_data_scaled = scaler.transform(sample_data)

# Predict crop type for the new sample
predicted_crop = rf_classifier.predict(sample_data_scaled)
predicted_crop_name = label_encoder.inverse_transform(predicted_crop)
print(f"Predicted Crop: {predicted_crop_name[0]}")
import joblib

# After training your model, save it
joblib.dump(rf_classifier, 'crop_recommendation_model.pkl')
joblib.dump(scaler, 'scaler.pkl')