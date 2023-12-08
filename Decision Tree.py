'''import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from keras.callbacks import ModelCheckpoint, CSVLogger

np.random.seed(1337)  # for reproducibility

# Load data
traindata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Training.csv', header=None)
testdata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Testing.csv', header=None)

X = traindata.iloc[:, 1:42]
Y = traindata.iloc[:, 0]
C = testdata.iloc[:, 0]
T = testdata.iloc[:, 1:42]

# Normalize data
scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

# Convert to NumPy arrays
y_train = np.array(Y)
y_test = np.array(C)

X_train = np.array(trainX)
X_test = np.array(testT)

# Create Decision Tree model
model_dt = DecisionTreeClassifier(random_state=1337)

# Train the Decision Tree model
model_dt.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = model_dt.predict(X_test)

# Calculate evaluation metrics for Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt, average="binary")
precision_dt = precision_score(y_test, y_pred_dt, average="binary")
f1_dt = f1_score(y_test, y_pred_dt, average="binary")

# Create a DataFrame to display the results for Decision Tree
results_dt = pd.DataFrame({
    'Metric': ['Accuracy (DT)', 'Recall (DT)', 'Precision (DT)', 'F1 Score (DT)'],
    'Value': [accuracy_dt, recall_dt, precision_dt, f1_dt]
})

print("Results for Decision Tree:")
print(results_dt)

# Decision Tree doesn't have training loss and accuracy metrics like neural networks,
# so we'll skip the plotting part for Decision Tree.

# Plot training state figures for Neural Network (similar to the previous code)
batch_size = 64
epochs = 10

model_nn = Sequential()
model_nn.add(Dense(1, input_dim=41, activation='relu'))
model_nn.add(Dropout(0.01))
model_nn.add(Dense(64, activation='relu'))
model_nn.add(Dropout(0.01))
model_nn.add(Dense(1, activation='sigmoid'))

model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="kddresults/neuralnetwork/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('kddresults/neuralnetwork/training_set_dnnanalysis.csv', separator=',', append=False)

# Lists to store training metrics
loss_history_nn = []
accuracy_history_nn = []

# Train the neural network model
for epoch in range(epochs):
    history = model_nn.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=1)
    loss_history_nn.append(history.history['loss'][0])
    accuracy_history_nn.append(history.history['accuracy'][0])

# Save the trained neural network model
model_nn.save("Trained Models/neuralnetwork_model.hdf5")

# Create training state figures for Neural Network
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), loss_history_nn, marker='o', linestyle='-', color='b')
plt.title('Training Loss (NN)')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), accuracy_history_nn, marker='o', linestyle='-', color='r')
plt.title('Training Accuracy (NN)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()'''

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
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

# Create Decision Tree model
model_dt = DecisionTreeClassifier(random_state=1337)

# Train the Decision Tree model
model_dt.fit(X_train_normalized, y_train)

# Make predictions on the test set
y_pred_dt = model_dt.predict(X_test_normalized)

# Calculate evaluation metrics for Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt, average="binary")
precision_dt = precision_score(y_test, y_pred_dt, average="binary")
f1_dt = f1_score(y_test, y_pred_dt, average="binary")

# Create a DataFrame to display the results for Decision Tree
results_dt = pd.DataFrame({
    'Metric': ['Accuracy (DT)', 'Recall (DT)', 'Precision (DT)', 'F1 Score (DT)'],
    'Value': [accuracy_dt, recall_dt, precision_dt, f1_dt]
})

print("Results for Decision Tree:")
print(results_dt)

