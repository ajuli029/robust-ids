'''import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(filename='model_logs.log', level=logging.INFO)

np.random.seed(1337)  # for reproducibility

# Load data
logging.info("Loading data...")
traindata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Training.csv', header=None)
testdata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Testing.csv', header=None)

X_train_rf = traindata.iloc[:, 1:42]
y_train_rf = traindata.iloc[:, 0]

X_test_rf = testdata.iloc[:, 1:42]
y_test_rf = testdata.iloc[:, 0]

X_train_dt = traindata.iloc[:, 1:42]
y_train_dt = traindata.iloc[:, 0]

X_test_dt = testdata.iloc[:, 1:42]
y_test_dt = testdata.iloc[:, 0]

X_train_gb = traindata.iloc[:, 1:42]
y_train_gb = traindata.iloc[:, 0]

X_test_gb = testdata.iloc[:, 1:42]
y_test_gb = testdata.iloc[:, 0]

# Normalize data for Random Forest
logging.info("Normalizing data for Random Forest...")
scaler_rf = Normalizer().fit(X_train_rf)
X_train_normalized_rf = scaler_rf.transform(X_train_rf)

scaler_rf = Normalizer().fit(X_test_rf)
X_test_normalized_rf = scaler_rf.transform(X_test_rf)

# Normalize data for Decision Tree
logging.info("Normalizing data for Decision Tree...")
scaler_dt = Normalizer().fit(X_train_dt)
X_train_normalized_dt = scaler_dt.transform(X_train_dt)

scaler_dt = Normalizer().fit(X_test_dt)
X_test_normalized_dt = scaler_dt.transform(X_test_dt)

# Normalize data for Gradient Boosting
logging.info("Normalizing data for Gradient Boosting...")
scaler_gb = Normalizer().fit(X_train_gb)
X_train_normalized_gb = scaler_gb.transform(X_train_gb)

scaler_gb = Normalizer().fit(X_test_gb)
X_test_normalized_gb = scaler_gb.transform(X_test_gb)

# Create Random Forest model
logging.info("Creating Random Forest model...")
model_rf = RandomForestClassifier(n_estimators=100, random_state=1337)
model_rf.fit(X_train_normalized_rf, y_train_rf)
y_pred_rf = model_rf.predict(X_test_normalized_rf)

# Calculate evaluation metrics for Random Forest
accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
recall_rf = recall_score(y_test_rf, y_pred_rf, average="binary")
precision_rf = precision_score(y_test_rf, y_pred_rf, average="binary")
f1_rf = f1_score(y_test_rf, y_pred_rf, average="binary")

# Log Random Forest results
logging.info("Results for Random Forest:")
logging.info(f"Accuracy (RF): {accuracy_rf}")
logging.info(f"Recall (RF): {recall_rf}")
logging.info(f"Precision (RF): {precision_rf}")
logging.info(f"F1 Score (RF): {f1_rf}")

# Create a DataFrame for Random Forest results
results_rf = pd.DataFrame({
    'Metric': ['Accuracy (RF)', 'Recall (RF)', 'Precision (RF)', 'F1 Score (RF)'],
    'Value': [accuracy_rf, recall_rf, precision_rf, f1_rf]
})

print("Results for Random Forest:")
print(results_rf)

# Create Decision Tree model
logging.info("Creating Decision Tree model...")
model_dt = DecisionTreeClassifier(random_state=1337)
model_dt.fit(X_train_normalized_dt, y_train_dt)
y_pred_dt = model_dt.predict(X_test_normalized_dt)

# Calculate evaluation metrics for Decision Tree
accuracy_dt = accuracy_score(y_test_dt, y_pred_dt)
recall_dt = recall_score(y_test_dt, y_pred_dt, average="binary")
precision_dt = precision_score(y_test_dt, y_pred_dt, average="binary")
f1_dt = f1_score(y_test_dt, y_pred_dt, average="binary")

# Log Decision Tree results
logging.info("Results for Decision Tree:")
logging.info(f"Accuracy (DT): {accuracy_dt}")
logging.info(f"Recall (DT): {recall_dt}")
logging.info(f"Precision (DT): {precision_dt}")
logging.info(f"F1 Score (DT): {f1_dt}")

# Create a DataFrame for Decision Tree results
results_dt = pd.DataFrame({
    'Metric': ['Accuracy (DT)', 'Recall (DT)', 'Precision (DT)', 'F1 Score (DT)'],
    'Value': [accuracy_dt, recall_dt, precision_dt, f1_dt]
})

print("Results for Decision Tree:")
print(results_dt)

# Create Gradient Boosting model
logging.info("Creating Gradient Boosting model...")
model_gb = GradientBoostingClassifier(random_state=1337)
model_gb.fit(X_train_normalized_gb, y_train_gb)
y_pred_gb = model_gb.predict(X_test_normalized_gb)

# Calculate evaluation metrics for Gradient Boosting
accuracy_gb = accuracy_score(y_test_gb, y_pred_gb)
recall_gb = recall_score(y_test_gb, y_pred_gb, average="binary")
precision_gb = precision_score(y_test_gb, y_pred_gb, average="binary")
f1_gb = f1_score(y_test_gb, y_pred_gb, average="binary")

# Log Gradient Boosting results
logging.info("Results for Gradient Boosting:")
logging.info(f"Accuracy (GB): {accuracy_gb}")
logging.info(f"Recall (GB): {recall_gb}")
logging.info(f"Precision (GB): {precision_gb}")
logging.info(f"F1 Score (GB): {f1_gb}")

# Create a DataFrame for Gradient Boosting results
results_gb = pd.DataFrame({
    'Metric': ['Accuracy (GB)', 'Recall (GB)', 'Precision (GB)', 'F1 Score (GB)'],
    'Value': [accuracy_gb, recall_gb, precision_gb, f1_gb]
})

print("Results for Gradient Boosting:")
print(results_gb)

# Majority Rule Vote
y_pred_majority = np.vstack((y_pred_rf, y_pred_dt, y_pred_gb)).mean(axis=0) >= 0.5

# Calculate evaluation metrics for Majority Rule Vote
accuracy_majority = accuracy_score(y_test_rf, y_pred_majority)
recall_majority = recall_score(y_test_rf, y_pred_majority, average="binary")
precision_majority = precision_score(y_test_rf, y_pred_majority, average="binary")
f1_majority = f1_score(y_test_rf, y_pred_majority, average="binary")

# Log Majority Rule Vote results
logging.info("Results for Majority Rule Vote:")
logging.info(f"Accuracy (Majority): {accuracy_majority}")
logging.info(f"Recall (Majority): {recall_majority}")
logging.info(f"Precision (Majority): {precision_majority}")
logging.info(f"F1 Score (Majority): {f1_majority}")

# Create a DataFrame for Majority Rule Vote results
results_majority = pd.DataFrame({
    'Metric': ['Accuracy (Majority)', 'Recall (Majority)', 'Precision (Majority)', 'F1 Score (Majority)'],
    'Value': [accuracy_majority, recall_majority, precision_majority, f1_majority]
})

print("Results for Majority Rule Vote:")
print(results_majority)

# Combine all results for plotting
results_rf['Model'] = 'Random Forest'
results_dt['Model'] = 'Decision Tree'
results_gb['Model'] = 'Gradient Boosting'
results_majority['Model'] = 'Majority Rule'

all_results = pd.concat([results_rf, results_dt, results_gb, results_majority])

# Plotting
metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score']

plt.figure(figsize=(12, 8))

bar_width = 0.2
bar_positions = np.arange(len(all_results['Model'].unique()))

for i, metric in enumerate(metrics):
    values = all_results[all_results['Metric'].str.contains(metric)]['Value']
    plt.bar(bar_positions + i * bar_width, values, width=bar_width, label=metric)

    # Add text labels on top of each bar
    for j, value in enumerate(values):
        plt.text(bar_positions[j] + i * bar_width, value + 0.01, f'{value:.3f}', ha='center')

plt.title('Performance Metrics Comparison')
plt.xlabel('Models')
plt.ylabel('Value')
plt.xticks(bar_positions + bar_width * (len(metrics) - 1) / 2, all_results['Model'].unique())
plt.legend()
plt.tight_layout()
plt.show()'''

'''import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(filename='model_logs.log', level=logging.INFO)

np.random.seed(1337)  # for reproducibility

# Load data
logging.info("Loading data...")
traindata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Training.csv', header=None)
testdata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Testing.csv', header=None)

X_train_rf = traindata.iloc[:, 1:42]
y_train_rf = traindata.iloc[:, 0]

X_test_rf = testdata.iloc[:, 1:42]
y_test_rf = testdata.iloc[:, 0]

X_train_dt = traindata.iloc[:, 1:42]
y_train_dt = traindata.iloc[:, 0]

X_test_dt = testdata.iloc[:, 1:42]
y_test_dt = testdata.iloc[:, 0]

X_train_gb = traindata.iloc[:, 1:42]
y_train_gb = traindata.iloc[:, 0]

X_test_gb = testdata.iloc[:, 1:42]
y_test_gb = testdata.iloc[:, 0]

# Normalize data for Random Forest
logging.info("Normalizing data for Random Forest...")
scaler_rf = Normalizer().fit(X_train_rf)
X_train_normalized_rf = scaler_rf.transform(X_train_rf)

scaler_rf = Normalizer().fit(X_test_rf)
X_test_normalized_rf = scaler_rf.transform(X_test_rf)

# Normalize data for Decision Tree
logging.info("Normalizing data for Decision Tree...")
scaler_dt = Normalizer().fit(X_train_dt)
X_train_normalized_dt = scaler_dt.transform(X_train_dt)

scaler_dt = Normalizer().fit(X_test_dt)
X_test_normalized_dt = scaler_dt.transform(X_test_dt)

# Normalize data for Gradient Boosting
logging.info("Normalizing data for Gradient Boosting...")
scaler_gb = Normalizer().fit(X_train_gb)
X_train_normalized_gb = scaler_gb.transform(X_train_gb)

scaler_gb = Normalizer().fit(X_test_gb)
X_test_normalized_gb = scaler_gb.transform(X_test_gb)

# Create Random Forest model
logging.info("Creating Random Forest model...")
model_rf = RandomForestClassifier(n_estimators=100, random_state=1337)
model_rf.fit(X_train_normalized_rf, y_train_rf)
y_pred_rf = model_rf.predict(X_test_normalized_rf)

# Calculate evaluation metrics for Random Forest
accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
recall_rf = recall_score(y_test_rf, y_pred_rf, average="binary")
precision_rf = precision_score(y_test_rf, y_pred_rf, average="binary")
f1_rf = f1_score(y_test_rf, y_pred_rf, average="binary")

# Log Random Forest results
logging.info("Results for Random Forest:")
logging.info(f"Accuracy (RF): {accuracy_rf}")
logging.info(f"Recall (RF): {recall_rf}")
logging.info(f"Precision (RF): {precision_rf}")
logging.info(f"F1 Score (RF): {f1_rf}")

# Create a DataFrame for Random Forest results
results_rf = pd.DataFrame({
    'Metric': ['Accuracy (RF)', 'Recall (RF)', 'Precision (RF)', 'F1 Score (RF)'],
    'Value': [accuracy_rf, recall_rf, precision_rf, f1_rf]
})

print("Results for Random Forest:")
print(results_rf)

# Create Decision Tree model
logging.info("Creating Decision Tree model...")
model_dt = DecisionTreeClassifier(random_state=1337)
model_dt.fit(X_train_normalized_dt, y_train_dt)
y_pred_dt = model_dt.predict(X_test_normalized_dt)

# Calculate evaluation metrics for Decision Tree
accuracy_dt = accuracy_score(y_test_dt, y_pred_dt)
recall_dt = recall_score(y_test_dt, y_pred_dt, average="binary")
precision_dt = precision_score(y_test_dt, y_pred_dt, average="binary")
f1_dt = f1_score(y_test_dt, y_pred_dt, average="binary")

# Log Decision Tree results
logging.info("Results for Decision Tree:")
logging.info(f"Accuracy (DT): {accuracy_dt}")
logging.info(f"Recall (DT): {recall_dt}")
logging.info(f"Precision (DT): {precision_dt}")
logging.info(f"F1 Score (DT): {f1_dt}")

# Create a DataFrame for Decision Tree results
results_dt = pd.DataFrame({
    'Metric': ['Accuracy (DT)', 'Recall (DT)', 'Precision (DT)', 'F1 Score (DT)'],
    'Value': [accuracy_dt, recall_dt, precision_dt, f1_dt]
})

print("Results for Decision Tree:")
print(results_dt)

# Create Gradient Boosting model
logging.info("Creating Gradient Boosting model...")
model_gb = GradientBoostingClassifier(random_state=1337)
model_gb.fit(X_train_normalized_gb, y_train_gb)
y_pred_gb = model_gb.predict(X_test_normalized_gb)

# Calculate evaluation metrics for Gradient Boosting
accuracy_gb = accuracy_score(y_test_gb, y_pred_gb)
recall_gb = recall_score(y_test_gb, y_pred_gb, average="binary")
precision_gb = precision_score(y_test_gb, y_pred_gb, average="binary")
f1_gb = f1_score(y_test_gb, y_pred_gb, average="binary")

# Log Gradient Boosting results
logging.info("Results for Gradient Boosting:")
logging.info(f"Accuracy (GB): {accuracy_gb}")
logging.info(f"Recall (GB): {recall_gb}")
logging.info(f"Precision (GB): {precision_gb}")
logging.info(f"F1 Score (GB): {f1_gb}")

# Create a DataFrame for Gradient Boosting results
results_gb = pd.DataFrame({
    'Metric': ['Accuracy (GB)', 'Recall (GB)', 'Precision (GB)', 'F1 Score (GB)'],
    'Value': [accuracy_gb, recall_gb, precision_gb, f1_gb]
})

print("Results for Gradient Boosting:")
print(results_gb)

# Majority Rule Vote
y_pred_majority = np.vstack((y_pred_rf, y_pred_dt, y_pred_gb)).mean(axis=0) >= 0.5

# Calculate evaluation metrics for Majority Rule Vote
accuracy_majority = accuracy_score(y_test_rf, y_pred_majority)
recall_majority = recall_score(y_test_rf, y_pred_majority, average="binary")
precision_majority = precision_score(y_test_rf, y_pred_majority, average="binary")
f1_majority = f1_score(y_test_rf, y_pred_majority, average="binary")

# Log Majority Rule Vote results
logging.info("Results for Majority Rule Vote:")
logging.info(f"Accuracy (Majority): {accuracy_majority}")
logging.info(f"Recall (Majority): {recall_majority}")
logging.info(f"Precision (Majority): {precision_majority}")
logging.info(f"F1 Score (Majority): {f1_majority}")

# Create a DataFrame for Majority Rule Vote results
results_majority = pd.DataFrame({
    'Metric': ['Accuracy (Majority)', 'Recall (Majority)', 'Precision (Majority)', 'F1 Score (Majority)'],
    'Value': [accuracy_majority, recall_majority, precision_majority, f1_majority]
})

print("Results for Majority Rule Vote:")
print(results_majority)

# Combine all results for plotting
results_rf['Model'] = 'Random Forest'
results_dt['Model'] = 'Decision Tree'
results_gb['Model'] = 'Gradient Boosting'
results_majority['Model'] = 'Majority Rule'

all_results = pd.concat([results_rf, results_dt, results_gb, results_majority])

# Plotting
metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score']

plt.figure(figsize=(12, 8))

bar_width = 0.2
bar_positions = np.arange(len(all_results['Model'].unique()))

for i, metric in enumerate(metrics):
    values = all_results[all_results['Metric'].str.contains(metric)]['Value']
    plt.bar(bar_positions + i * bar_width, values, width=bar_width, label=metric)

    # Add text labels on top of each bar
    for j, value in enumerate(values):
        plt.text(bar_positions[j] + i * bar_width, value + 0.01, f'{value:.3f}', ha='center')

plt.title('Performance Metrics Comparison')
plt.xlabel('Models')
plt.ylabel('Value')
plt.xticks(bar_positions + bar_width * (len(metrics) - 1) / 2, all_results['Model'].unique())

# Move the legend to the bottom right
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()'''

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(filename='model_logs.log', level=logging.INFO)

np.random.seed(1337)  # for reproducibility

# Load data
logging.info("Loading data...")
traindata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Training.csv', header=None)
testdata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Testing.csv', header=None)

X_train = traindata.iloc[:, 1:42]
y_train = traindata.iloc[:, 0]

X_test = testdata.iloc[:, 1:42]
y_test = testdata.iloc[:, 0]

# Normalize data
logging.info("Normalizing data...")
scaler = Normalizer().fit(X_train)
X_train_normalized = scaler.transform(X_train)

scaler = Normalizer().fit(X_test)
X_test_normalized = scaler.transform(X_test)

# Create Neural Network model
logging.info("Creating Neural Network model...")
model_nn = Sequential()
model_nn.add(Dense(64, input_dim=X_train_normalized.shape[1], activation='relu'))
model_nn.add(Dropout(0.5))
model_nn.add(Dense(1, activation='sigmoid'))
model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Neural Network
model_nn.fit(X_train_normalized, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

# Evaluate the Neural Network model
y_pred_nn = (model_nn.predict(X_test_normalized) >= 0.5).astype(int)

# Calculate evaluation metrics for Neural Network
accuracy_nn = accuracy_score(y_test, y_pred_nn)
recall_nn = recall_score(y_test, y_pred_nn, average="binary")
precision_nn = precision_score(y_test, y_pred_nn, average="binary")
f1_nn = f1_score(y_test, y_pred_nn, average="binary")

# Log Neural Network results
logging.info("Results for Neural Network:")
logging.info(f"Accuracy (NN): {accuracy_nn}")
logging.info(f"Recall (NN): {recall_nn}")
logging.info(f"Precision (NN): {precision_nn}")
logging.info(f"F1 Score (NN): {f1_nn}")

# Create a DataFrame for Neural Network results
results_nn = pd.DataFrame({
    'Metric': ['Accuracy (NN)', 'Recall (NN)', 'Precision (NN)', 'F1 Score (NN)'],
    'Value': [accuracy_nn, recall_nn, precision_nn, f1_nn]
})

print("Results for Neural Network:")
print(results_nn)

# Create SVM model
logging.info("Creating SVM model...")
model_svm = SVC(kernel='linear')
model_svm.fit(X_train_normalized, y_train)
y_pred_svm = model_svm.predict(X_test_normalized)

# Calculate evaluation metrics for SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm, average="binary")
precision_svm = precision_score(y_test, y_pred_svm, average="binary")
f1_svm = f1_score(y_test, y_pred_svm, average="binary")

# Log SVM results
logging.info("Results for SVM:")
logging.info(f"Accuracy (SVM): {accuracy_svm}")
logging.info(f"Recall (SVM): {recall_svm}")
logging.info(f"Precision (SVM): {precision_svm}")
logging.info(f"F1 Score (SVM): {f1_svm}")

# Create a DataFrame for SVM results
results_svm = pd.DataFrame({
    'Metric': ['Accuracy (SVM)', 'Recall (SVM)', 'Precision (SVM)', 'F1 Score (SVM)'],
    'Value': [accuracy_svm, recall_svm, precision_svm, f1_svm]
})

print("Results for SVM:")
print(results_svm)

# Create AdaBoost model
logging.info("Creating AdaBoost model...")
base_model = None  # No base model needed for AdaBoost
model_adaboost = AdaBoostClassifier(base_model, n_estimators=50, learning_rate=1.0)
model_adaboost.fit(X_train_normalized, y_train)
y_pred_adaboost = model_adaboost.predict(X_test_normalized)

# Calculate evaluation metrics for AdaBoost
accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
recall_adaboost = recall_score(y_test, y_pred_adaboost, average="binary")
precision_adaboost = precision_score(y_test, y_pred_adaboost, average="binary")
f1_adaboost = f1_score(y_test, y_pred_adaboost, average="binary")

# Log AdaBoost results
logging.info("Results for AdaBoost:")
logging.info(f"Accuracy (AdaBoost): {accuracy_adaboost}")
logging.info(f"Recall (AdaBoost): {recall_adaboost}")
logging.info(f"Precision (AdaBoost): {precision_adaboost}")
logging.info(f"F1 Score (AdaBoost): {f1_adaboost}")

# Create a DataFrame for AdaBoost results
results_adaboost = pd.DataFrame({
    'Metric': ['Accuracy (AdaBoost)', 'Recall (AdaBoost)', 'Precision (AdaBoost)', 'F1 Score (AdaBoost)'],
    'Value': [accuracy_adaboost, recall_adaboost, precision_adaboost, f1_adaboost]
})

print("Results for AdaBoost:")
print(results_adaboost)

# Weighted Majority Rule Vote
weights = {'nn': 0.3, 'svm': 0.2, 'adaboost': 0.5}
y_pred_majority_weighted = (weights['nn'] * y_pred_nn.ravel() + weights['svm'] * y_pred_svm + weights['adaboost'] * y_pred_adaboost.ravel()) >= 0.5

# Calculate evaluation metrics for Weighted Majority Rule Vote
accuracy_majority_weighted = accuracy_score(y_test, y_pred_majority_weighted)
recall_majority_weighted = recall_score(y_test, y_pred_majority_weighted, average="binary")
precision_majority_weighted = precision_score(y_test, y_pred_majority_weighted, average="binary")
f1_majority_weighted = f1_score(y_test, y_pred_majority_weighted, average="binary")

# Log Weighted Majority Rule Vote results
logging.info("Results for Weighted Majority Rule Vote:")
logging.info(f"Accuracy (Weighted Majority): {accuracy_majority_weighted}")
logging.info(f"Recall (Weighted Majority): {recall_majority_weighted}")
logging.info(f"Precision (Weighted Majority): {precision_majority_weighted}")
logging.info(f"F1 Score (Weighted Majority): {f1_majority_weighted}")

# Create a DataFrame for Weighted Majority Rule Vote results
results_majority_weighted = pd.DataFrame({
    'Metric': ['Accuracy (Weighted Majority)', 'Recall (Weighted Majority)', 'Precision (Weighted Majority)', 'F1 Score (Weighted Majority)'],
    'Value': [accuracy_majority_weighted, recall_majority_weighted, precision_majority_weighted, f1_majority_weighted]
})

print("Results for Weighted Majority Rule Vote:")
print(results_majority_weighted)

# Combine all results for plotting
results_nn['Model'] = 'Neural Network'
results_svm['Model'] = 'SVM'
results_adaboost['Model'] = 'AdaBoost'
results_majority_weighted['Model'] = 'Weighted Majority Rule'

all_results = pd.concat([results_nn, results_svm, results_adaboost, results_majority_weighted])

# Plotting
metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score']

plt.figure(figsize=(12, 8))

bar_width = 0.2
bar_positions = np.arange(len(all_results['Model'].unique()))

for i, metric in enumerate(metrics):
    values = all_results[all_results['Metric'].str.contains(metric)]['Value']
    plt.bar(bar_positions + i * bar_width, values, width=bar_width, label=metric)

    # Add text labels on top of each bar
    for j, value in enumerate(values):
        plt.text(bar_positions[j] + i * bar_width, value + 0.01, f'{value:.3f}', ha='center')

plt.title('Performance Metrics Comparison')
plt.xlabel('Models')
plt.ylabel('Value')
plt.xticks(bar_positions + bar_width * (len(metrics) - 1) / 2, all_results['Model'].unique())
plt.legend()
plt.tight_layout()
plt.show()

