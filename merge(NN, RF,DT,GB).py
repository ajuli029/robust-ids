import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt

np.random.seed(1337)  # for reproducibility

# Load data
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

X_train_nn = traindata.iloc[:, 1:42]
y_train_nn = traindata.iloc[:, 0]

X_test_nn = testdata.iloc[:, 1:42]
y_test_nn = testdata.iloc[:, 0]

X_train_gb = traindata.iloc[:, 1:42]
y_train_gb = traindata.iloc[:, 0]

X_test_gb = testdata.iloc[:, 1:42]
y_test_gb = testdata.iloc[:, 0]

# Normalize data for Random Forest
scaler_rf = Normalizer().fit(X_train_rf)
X_train_normalized_rf = scaler_rf.transform(X_train_rf)

scaler_rf = Normalizer().fit(X_test_rf)
X_test_normalized_rf = scaler_rf.transform(X_test_rf)

# Normalize data for Decision Tree
scaler_dt = Normalizer().fit(X_train_dt)
X_train_normalized_dt = scaler_dt.transform(X_train_dt)

scaler_dt = Normalizer().fit(X_test_dt)
X_test_normalized_dt = scaler_dt.transform(X_test_dt)

# Normalize data for Neural Network
scaler_nn = Normalizer().fit(X_train_nn)
X_train_normalized_nn = scaler_nn.transform(X_train_nn)

scaler_nn = Normalizer().fit(X_test_nn)
X_test_normalized_nn = scaler_nn.transform(X_test_nn)

# Normalize data for Gradient Boosting
scaler_gb = Normalizer().fit(X_train_gb)
X_train_normalized_gb = scaler_gb.transform(X_train_gb)

scaler_gb = Normalizer().fit(X_test_gb)
X_test_normalized_gb = scaler_gb.transform(X_test_gb)

# Create Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=1337)
model_rf.fit(X_train_normalized_rf, y_train_rf)
y_pred_rf = model_rf.predict(X_test_normalized_rf)

# Calculate evaluation metrics for Random Forest
accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
recall_rf = recall_score(y_test_rf, y_pred_rf, average="binary")
precision_rf = precision_score(y_test_rf, y_pred_rf, average="binary")
f1_rf = f1_score(y_test_rf, y_pred_rf, average="binary")

# Create a DataFrame for Random Forest results
results_rf = pd.DataFrame({
    'Metric': ['Accuracy (RF)', 'Recall (RF)', 'Precision (RF)', 'F1 Score (RF)'],
    'Value': [accuracy_rf, recall_rf, precision_rf, f1_rf]
})

print("Results for Random Forest:")
print(results_rf)

# Create Decision Tree model
model_dt = DecisionTreeClassifier(random_state=1337)
model_dt.fit(X_train_normalized_dt, y_train_dt)
y_pred_dt = model_dt.predict(X_test_normalized_dt)

# Calculate evaluation metrics for Decision Tree
accuracy_dt = accuracy_score(y_test_dt, y_pred_dt)
recall_dt = recall_score(y_test_dt, y_pred_dt, average="binary")
precision_dt = precision_score(y_test_dt, y_pred_dt, average="binary")
f1_dt = f1_score(y_test_dt, y_pred_dt, average="binary")

# Create a DataFrame for Decision Tree results
results_dt = pd.DataFrame({
    'Metric': ['Accuracy (DT)', 'Recall (DT)', 'Precision (DT)', 'F1 Score (DT)'],
    'Value': [accuracy_dt, recall_dt, precision_dt, f1_dt]
})

print("Results for Decision Tree:")
print(results_dt)

# Create Neural Network model
model_nn = Sequential()
model_nn.add(Dense(1, input_dim=41, activation='relu'))
model_nn.add(Dropout(0.01))
model_nn.add(Dense(64, activation='relu'))
model_nn.add(Dropout(0.01))
model_nn.add(Dense(1, activation='sigmoid'))
model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="kddresults/neuralnetwork/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('kddresults/neuralnetwork/training_set_dnnanalysis.csv', separator=',', append=False)

epochs = 10

history_nn = model_nn.fit(X_train_normalized_nn, y_train_nn, batch_size=64, epochs=epochs, verbose=1)
y_pred_nn = model_nn.predict(X_test_normalized_nn)

# Ensure the dimensions match for the Neural Network predictions
y_pred_nn = (y_pred_nn >= 0.5).astype(int).flatten()

# Calculate evaluation metrics for Neural Network
accuracy_nn = accuracy_score(y_test_nn, y_pred_nn)
recall_nn = recall_score(y_test_nn, y_pred_nn, average="binary")
precision_nn = precision_score(y_test_nn, y_pred_nn, average="binary")
f1_nn = f1_score(y_test_nn, y_pred_nn, average="binary")

# Create a DataFrame for Neural Network results
results_nn = pd.DataFrame({
    'Metric': ['Accuracy (NN)', 'Recall (NN)', 'Precision (NN)', 'F1 Score (NN)'],
    'Value': [accuracy_nn, recall_nn, precision_nn, f1_nn]
})

print("Results for Neural Network:")
print(results_nn)

# Create Gradient Boosting model
model_gb = GradientBoostingClassifier(random_state=1337)
model_gb.fit(X_train_normalized_gb, y_train_gb)
y_pred_gb = model_gb.predict(X_test_normalized_gb)

# Calculate evaluation metrics for Gradient Boosting
accuracy_gb = accuracy_score(y_test_gb, y_pred_gb)
recall_gb = recall_score(y_test_gb, y_pred_gb, average="binary")
precision_gb = precision_score(y_test_gb, y_pred_gb, average="binary")
f1_gb = f1_score(y_test_gb, y_pred_gb, average="binary")

# Create a DataFrame for Gradient Boosting results
results_gb = pd.DataFrame({
    'Metric': ['Accuracy (GB)', 'Recall (GB)', 'Precision (GB)', 'F1 Score (GB)'],
    'Value': [accuracy_gb, recall_gb, precision_gb, f1_gb]
})

print("Results for Gradient Boosting:")
print(results_gb)

# Majority Rule Vote
y_pred_majority = np.vstack((y_pred_rf, y_pred_dt, y_pred_nn, y_pred_gb)).mean(axis=0) >= 0.5

# Calculate evaluation metrics for Majority Rule Vote
accuracy_majority = accuracy_score(y_test_rf, y_pred_majority)
recall_majority = recall_score(y_test_rf, y_pred_majority, average="binary")
precision_majority = precision_score(y_test_rf, y_pred_majority, average="binary")
f1_majority = f1_score(y_test_rf, y_pred_majority, average="binary")

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
results_nn['Model'] = 'Neural Network'
results_gb['Model'] = 'Gradient Boosting'
results_majority['Model'] = 'Majority Rule'

all_results = pd.concat([results_rf, results_dt, results_nn, results_gb, results_majority])

# Plotting
metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score']

plt.figure(figsize=(12, 8))

for metric in metrics:
    values = all_results[all_results['Metric'].str.contains(metric)]['Value']
    plt.plot(all_results['Model'].unique(), values, marker='o', label=metric)

plt.title('Performance Metrics Comparison')
plt.xlabel('Models')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()
