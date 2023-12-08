import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import keras
import joblib

# Load the trained models
model1 = load_model('kddresults/dnn3layer/dnn3layer_model.hdf5')
model2 = load_model('kddresults/testing/test_model.hdf5')

# Load the Random Forest classifier model
rf_classifier = joblib.load('kddresults/random_forest_classifier_model.pkl')

# Load the kddcup99 dataset
testdata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Testing.csv', header=None)

# Prepare the dataset for evaluation
X_test = testdata.iloc[:, :-1].values  # Features
y_test = testdata.iloc[:, -1].values  # Labels

# Convert labels to integer type
y_test = y_test.astype(int)

# Standardize the input features
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# Create an array to hold the ensemble predictions
ensemble_predictions = np.zeros((X_test.shape[0], 5))  # 5 classes

# Make predictions using the models
pred_model1 = model1.predict(X_test)
pred_model2 = model2.predict(X_test)
pred_rf = rf_classifier.predict(X_test)

# Reshape pred_rf to be compatible with pred_model1 and pred_model2
pred_rf = pred_rf.reshape(-1, 1)

# Add the predictions to the ensemble_predictions array
ensemble_predictions[:, 0] = pred_model1[:, 0]
ensemble_predictions[:, 1] = pred_model2[:, 0]
ensemble_predictions[:, 2] = pred_rf[:, 0]

# Calculate the majority vote for each example
for i in range(X_test.shape[0]):
    ensemble_predictions[i, 3] = np.argmax(ensemble_predictions[i, :3])  # Majority class
    ensemble_predictions[i, 4] = ensemble_predictions[i, int(ensemble_predictions[i, 3])]

# Convert majority vote results to integer type
ensemble_predictions[:, 3] = ensemble_predictions[:, 3].astype(int)
ensemble_predictions[:, 4] = ensemble_predictions[:, 4].astype(int)

# Calculate accuracy
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions[:, 3])

print("Ensemble Accuracy:", ensemble_accuracy)

# Print the classification report for the majority vote ensemble
classification_rep = classification_report(y_test, ensemble_predictions[:, 3], zero_division=1)
print("Classification Report for Majority Vote Ensemble:\n", classification_rep)
