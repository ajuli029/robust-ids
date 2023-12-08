import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import Normalizer
from keras.callbacks import ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt

np.random.seed(1337)  # for reproducibility

# Load data
traindata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Training.csv', header=None)
testdata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Testing.csv', header=None)

X = traindata.iloc[:, 1:42]
Y = traindata.iloc[:, 0]
C = testdata.iloc[:, 0]
T = testdata.iloc[:, 1:42]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)

X_train = np.array(trainX)
X_test = np.array(testT)

batch_size = 64

# Define the network
model = Sequential()
model.add(Dense(1, input_dim=41, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Add CSVLogger callback to save accuracy values during training
csv_logger = CSVLogger('kddresults/dnn3layer/training_set_dnnanalysis.csv', separator=',', append=False)

# Train the model and log accuracy during training
history = model.fit(X_train, y_train, batch_size=batch_size, callbacks=[csv_logger])

# Load the logged accuracy values from the CSV file
training_results = pd.read_csv('kddresults/dnn3layer/training_set_dnnanalysis.csv')


# Calculate and display accuracy, recall, precision, and F1 score
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)

# Create and display the features table with weights
feature_weights = pd.DataFrame({
    'Feature': traindata.columns[1:42],  # Assuming the column names contain feature names
    'Weight': model.layers[-1].get_weights()[0].flatten()
})

print("Features Table:")
print(feature_weights.sort_values(by='Weight', ascending=False))

# Plot the accuracy graph
plt.figure(figsize=(10, 6))
plt.plot(training_results['accuracy'])
plt.title('Model Accuracy during Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()
