'''import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import Normalizer
from keras import callbacks
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
checkpointer = ModelCheckpoint(filepath="kddresults/dnn3layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('kddresults/dnn3layer/training_set_dnnanalysis.csv', separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size, callbacks=[checkpointer, csv_logger])
model.save("kddresults/dnn3layer/dnn3layer_model.hdf5")

score = []

y_train1 = y_test
y_pred = model.predict(X_test)

threshold = 0.5
y_pred_thresholded = (y_pred >= threshold).astype(int)

accuracy = accuracy_score(y_train1, y_pred_thresholded)
recall = recall_score(y_train1, y_pred_thresholded, average="binary")
precision = precision_score(y_train1, y_pred_thresholded, average="binary")
f1 = f1_score(y_train1, y_pred_thresholded, average="binary")
print("----------------------------------------------")
print("accuracy")
print("%.3f" % accuracy)
print("recall")
print("%.3f" % recall)
print("precision")
print("%.3f" % precision)
print("f1score")
print("%.3f" % f1)
score.append(accuracy)

# Get the weights of the first layer
weights = model.layers[0].get_weights()[0].flatten()

# Check if the length of weights matches the number of features
if len(weights) != X_train.shape[1]:
    print("Error: The number of weights does not match the number of features.")
else:
    feature_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]

    # Create a DataFrame for features and their corresponding weights
    features_table = pd.DataFrame({
        'Feature': feature_names,
        'Weight': weights
    })

    # Sort the features table by weight in descending order
    features_table = features_table.sort_values(by='Weight', ascending=False)

    # Print the features table
    print("Features Table:")
    print(features_table.to_string(index=False))
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import keras
import joblib
from sklearn.svm import SVC

# Load the trained models
model1 = load_model('kddresults/dnn3layer/dnn3layer_model.hdf5')
model2 = load_model('kddresults/testing/test_model.hdf5')

# Load the kddcup99 dataset
testdata = pd.read_csv(
    'https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Testing.csv',
    header=None)

# Prepare the dataset for evaluation
X_test = testdata.iloc[:, :-1].values  # Features
y_test = testdata.iloc[:, -1].values  # Labels

# Convert labels to integer type
y_test = y_test.astype(int)

# Create a validation set to tune the ensemble weights
X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert labels to integer type
y_val = y_val.astype(int)


# Define a custom callback to record accuracy during training
class AccuracyHistoryCallback(keras.callbacks.Callback):
    def __init__(self, X_val):
        self.X_val = X_val
        self.accuracy_history = []
        self.best_accuracy = 0.0

    def on_epoch_end(self, epoch, logs={}):
        y_pred_model1 = model1.predict(self.X_val)
        y_pred_model2 = model2.predict(self.X_val)
        y_pred_svm = svm_model.predict(self.X_val)

        # Reshape y_pred_svm to be compatible with y_pred_model1 and y_pred_model2
        y_pred_svm = y_pred_svm.reshape(-1, 1)

        y_pred_weighted = (
                    best_weight_model1 * y_pred_model1 + best_weight_model2 * y_pred_model2 + best_weight_svm * y_pred_svm)

        accuracy = keras.metrics.sparse_categorical_accuracy(y_val, y_pred_weighted)
        self.accuracy_history.append(np.mean(accuracy))

        # Keep track of the best accuracy
        if np.mean(accuracy) > self.best_accuracy:
            self.best_accuracy = np.mean(accuracy)


# Create an instance of the callback with X_val
accuracy_history_callback = AccuracyHistoryCallback(X_val)

best_accuracy = 0.0
best_weight_model1 = 0.0
best_weight_model2 = 0.0
best_weight_svm = 0.0

# Try different weight combinations for the ensemble
for weight_model1 in np.arange(0.05, 1.0, 0.05):
    for weight_model2 in np.arange(0.05, 1.0 - weight_model1, 0.05):
        weight_svm = 1.0 - weight_model1 - weight_model2

        # Load the SVM model
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)  # Fit the SVM model

        # Make predictions using the models
        pred_model1 = model1.predict(X_val)
        pred_model2 = model2.predict(X_val)
        pred_svm = svm_model.predict(X_val)

        # Reshape pred_svm to be compatible with pred_model1 and pred_model2
        pred_svm = pred_svm.reshape(-1, 1)

        # Calculate the weighted merging of predictions
        y_pred_weighted = (weight_model1 * pred_model1 + weight_model2 * pred_model2 + weight_svm * pred_svm)

        # Calculate accuracy without shape validation
        accuracy = keras.metrics.sparse_categorical_accuracy(y_val, y_pred_weighted)

        # Keep track of the best weight combination
        if np.mean(accuracy) > best_accuracy:
            best_accuracy = np.mean(accuracy)
            best_weight_model1 = weight_model1
            best_weight_model2 = weight_model2
            best_weight_svm = weight_svm

# Save the accuracy of the best weight combination
accuracy_file = "best_accuracy.txt"
with open(accuracy_file, 'w') as f:
    f.write(str(best_accuracy))

print("Best Weight for Model1:", best_weight_model1)
print("Best Weight for Model2:", best_weight_model2)
print("Best Weight for SVM:", best_weight_svm)

# Train the callback model and record accuracy during training with early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model_history = model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=41,
                           callbacks=[accuracy_history_callback, early_stopping])

# Save the accuracy history to a NumPy file
accuracy_history = {
    'train_accuracy': model_history.history['accuracy'],
    'val_accuracy': model_history.history['val_accuracy']
}

np.save("kddresults/testing/accuracy_history.npy", accuracy_history)

# Use the best weight combination to make predictions on the test set
y_pred_model1 = model1.predict(X_test)
y_pred_model2 = model2.predict(X_test)
y_pred_svm = svm_model.predict(X_test)

# Reshape y_pred_svm to be compatible with y_pred_model1 and y_pred_model2
y_pred_svm = y_pred_svm.reshape(-1, 1)

y_pred_weighted = (
            best_weight_model1 * y_pred_model1 + best_weight_model2 * y_pred_model2 + best_weight_svm * y_pred_svm)

# Calculate accuracy and print the results
accuracy = accuracy_score(y_test, np.argmax(y_pred_weighted, axis=1))
print("Ensemble Accuracy on Test Set:", accuracy)

# You can also print classification reports or any other desired analysis here
