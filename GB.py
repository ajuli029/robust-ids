import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import Normalizer

np.random.seed(1337)  # for reproducibility

# Load data
traindata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Training.csv', header=None)
testdata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Testing.csv', header=None)

X_train = traindata.iloc[:, 1:42]
y_train = traindata.iloc[:, 0]

X_test = testdata.iloc[:, 1:42]
y_test = testdata.iloc[:, 0]

# Normalize data
scaler = Normalizer().fit(X_train)
X_train_normalized = scaler.transform(X_train)

scaler = Normalizer().fit(X_test)
X_test_normalized = scaler.transform(X_test)

# Create Gradient Boosting model
model_gb = GradientBoostingClassifier(random_state=1337)

# Train the Gradient Boosting model
model_gb.fit(X_train_normalized, y_train)

# Make predictions on the test set
y_pred_gb = model_gb.predict(X_test_normalized)

# Calculate evaluation metrics for Gradient Boosting
accuracy_gb = accuracy_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb, average="binary")
precision_gb = precision_score(y_test, y_pred_gb, average="binary")
f1_gb = f1_score(y_test, y_pred_gb, average="binary")

# Create a DataFrame to display the results for Gradient Boosting
results_gb = pd.DataFrame({
    'Metric': ['Accuracy (GB)', 'Recall (GB)', 'Precision (GB)', 'F1 Score (GB)'],
    'Value': [accuracy_gb, recall_gb, precision_gb, f1_gb]
})

print("Results for Gradient Boosting:")
print(results_gb)
