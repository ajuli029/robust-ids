import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, CSVLogger

# Load the UNSW-NB15 dataset (replace with the actual dataset URL)
train_data = pd.read_csv('https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv', header=None)
test_data = pd.read_csv('https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.csv', header=None)

# Define column names
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_cat", "label"
] # Add all column names

# Assign column names to the dataset
train_data.columns = column_names
test_data.columns = column_names

# Select features and target
X_train = train_data.drop("label", axis=1)  # Replace "label" with the actual target column name
y_train = train_data["label"]  # Replace "label" with the actual target column name

X_test = test_data.drop("label", axis=1)  # Replace "label" with the actual target column name
y_test = test_data["label"]  # Replace "label" with the actual target column name

# Standardize numeric features
numeric_features = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]  # Add all numeric feature names

scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# One-hot encode categorical features
categorical_features = [
    "protocol_type", "service", "flag"
]


encoder = OneHotEncoder(drop='first', sparse=False)
X_train_encoded = encoder.fit_transform(X_train[categorical_features])
X_test_encoded = encoder.transform(X_test[categorical_features])

# Combine numeric and encoded categorical features
X_train = np.concatenate((X_train[numeric_features].values, X_train_encoded), axis=1)
X_test = np.concatenate((X_test[numeric_features].values, X_test_encoded), axis=1)

# Define and train the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define callbacks
checkpointer = ModelCheckpoint(filepath="unsw_results/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_loss')
csv_logger = CSVLogger('unsw_results/training_set_dnnanalysis.csv', separator=',', append=False)

# Train the model
batch_size = 64
epochs = 10
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs, callbacks=[checkpointer, csv_logger])

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_thresholded = (y_pred >= 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_thresholded)
precision = precision_score(y_test, y_pred_thresholded, average='weighted')
recall = recall_score(y_test, y_pred_thresholded, average='weighted')
f1 = f1_score(y_test, y_pred_thresholded, average='weighted')

print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))
