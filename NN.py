import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import Normalizer
from keras.callbacks import ModelCheckpoint, CSVLogger

np.random.seed(1337)  # for reproducibility

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
checkpointer = ModelCheckpoint(filepath="kddresults/neuralnetwork/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('kddresults/neuralnetwork/training_set_dnnanalysis.csv', separator=',', append=False)

# Set the number of epochs to 10
epochs = 10

# Create lists to store training metrics
loss_history = []
accuracy_history = []

for epoch in range(epochs):
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=1)
    loss_history.append(history.history['loss'][0])
    accuracy_history.append(history.history['accuracy'][0])

model.save("Trained Models/neuralnetwork_model.hdf5")

# Create training state figures
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-', color='b')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), accuracy_history, marker='o', linestyle='-', color='r')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

score = []

y_train1 = y_test
y_pred = model.predict(X_test)

threshold = 0.5
y_pred_thresholded = (y_pred >= threshold).astype(int)

accuracy = accuracy_score(y_train1, y_pred_thresholded)
recall = recall_score(y_train1, y_pred_thresholded, average="binary")
precision = precision_score(y_train1, y_pred_thresholded, average="binary")
f1 = f1_score(y_train1, y_pred_thresholded, average="binary")

# Create a DataFrame to display the results
results = pd.DataFrame({
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1 Score'],
    'Value': [accuracy, recall, precision, f1]
})

print("Results:")
print(results)