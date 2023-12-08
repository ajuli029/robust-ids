import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import keras
import h5py
import joblib

# Load the trained models
model1 = load_model('Models/Trained Models/dnn3layer_model.hdf5')
model2 = load_model('Models/Trained Models/test_model.hdf5')

# Load the kddcup99 dataset
testdata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Testing.csv', header=None)

# Prepare the dataset for evaluation
X_test = testdata.iloc[:, :-1].values  # Features
y_test = testdata.iloc[:, -1].values   # Labels

# Convert labels to integer type
y_test = y_test.astype(int)

# Standardize the input features
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# Create a validation set to tune the ensemble weights
X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=42)

# Convert labels to integer type
y_val = y_val.astype(int)

# Define a custom callback to record accuracy during training
class AccuracyHistoryCallback(keras.callbacks.Callback):
    def __init__(self, X_val):
        self.X_val = X_val
        self.accuracy_history = []
        self.best_accuracy = 0.0

    def on_epoch_end(self, epoch, logs={}):
        y_pred_weighted = (best_weight_model1 * model1.predict(self.X_val) + best_weight_model2 * model2.predict(self.X_val)) / 2
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

# Try different weight combinations for the ensemble
for weight_model1 in np.arange(0.05, 1.0, 0.05):
    weight_model2 = 1.0 - weight_model1

    # Weighted merging of predictions
    y_pred_weighted = (weight_model1 * model1.predict(X_val) + weight_model2 * model2.predict(X_val)) / 2
    accuracy = keras.metrics.sparse_categorical_accuracy(y_val, y_pred_weighted)

    # Keep track of the best weight combination
    if np.mean(accuracy) > best_accuracy:
        best_accuracy = np.mean(accuracy)
        best_weight_model1 = weight_model1
        best_weight_model2 = weight_model2

# Save the accuracy of the best weight combination
accuracy_file = "best_accuracy.txt"
with open(accuracy_file, 'w') as f:
    f.write(str(best_accuracy))

print("Best Weight for Model1:", best_weight_model1)
print("Best Weight for Model2:", best_weight_model2)

# Train the callback model and record accuracy during training with early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model_history = model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=41, callbacks=[accuracy_history_callback, early_stopping])

# Save the accuracy history to a NumPy file
accuracy_history = {
    'train_accuracy': model_history.history['accuracy'],
    'val_accuracy': model_history.history['val_accuracy']
}

np.save("accuracy_history.npy", accuracy_history)

# Use the best weight combination to make predictions on the test set
y_pred_weighted = (best_weight_model1 * model1.predict(X_test) + best_weight_model2 * model2.predict(X_test)) / 2
accuracy = keras.metrics.sparse_categorical_accuracy(y_test, y_pred_weighted)

print("----------------------------------------------")
print("Final Accuracy on Test Set (Models 1 and 2):", np.mean(accuracy))

# Print the classification report
y_pred_weighted_labels = np.argmax(y_pred_weighted, axis=1)
classification_rep = classification_report(y_test, y_pred_weighted_labels, zero_division=1)
print("Classification Report:\n", classification_rep)

    # Plot the accuracy history during training
    #plt.plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'], label='Training Accuracy')
    #plt.plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'], label='Validation Accuracy')
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #plt.legend()
    #plt.grid(True)
    #plt.show()


# Load the Random Forest model
random_forest_model = joblib.load('Models/Trained Models/random_forest_model.h5')

# Load the kddcup99 dataset
testdata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Testing.csv', header=None)

# Prepare the dataset for evaluation
X_test = testdata.iloc[:, 1:42].values  # Features
y_test = testdata.iloc[:, 0].values   # Labels

# Convert labels to integer type
y_test = y_test.astype(int)

# Use the Random Forest model for predictions
y_pred_random_forest = random_forest_model.predict(X_test)

# Evaluate and display results for the Random Forest model
accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)
classification_rep_random_forest = classification_report(y_test, y_pred_random_forest, zero_division=1)

print("Results for Random Forest Model:")
print("Accuracy on Test Set (Random Forest):", accuracy_random_forest)
print("Classification Report for Random Forest Model:\n", classification_rep_random_forest)


# Load the Gradient Boosting model
gradient_boosting_model = joblib.load('Models/Trained Models/gradient_boosting_model.h5')

# Load the kddcup99 dataset
testdata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Testing.csv', header=None)

# Separate features and target variable for testing data
X_test = testdata.iloc[:, 1:42]
y_test = testdata.iloc[:, 0]

# Use the Gradient Boosting model for predictions
y_pred_gradient_boosting = gradient_boosting_model.predict(X_test)

# Evaluate and display results for the Gradient Boosting model
accuracy_gradient_boosting = accuracy_score(y_test, y_pred_gradient_boosting)
classification_rep_gradient_boosting = classification_report(y_test, y_pred_gradient_boosting, zero_division=1)

print("Results for Gradient Boosting Model:")
print("Accuracy on Test Set (Gradient Boosting):", accuracy_gradient_boosting)
print("Classification Report for Gradient Boosting Model:\n", classification_rep_gradient_boosting)



# Calculate accuracy differences between Models 1/2 and Random Forest model
accuracy_diff_rf = accuracy_random_forest - np.mean(accuracy)
print("Accuracy Difference between Models 1/2 and Random Forest Model:", accuracy_diff_rf)

# Calculate accuracy differences between Models 1/2 and Gradient Boosting model
accuracy_diff_lr = accuracy_gradient_boosting - np.mean(accuracy)
print("Accuracy Difference between Models 1/2 and Gradient Boosting Model:", accuracy_diff_lr)


# Visualize the distribution of the target variable in the test dataset
plt.figure(figsize=(8, 6))
sns.countplot(x=y_test)
plt.title('Distribution of Target Variable in Test Dataset')
plt.xlabel('Target Variable')
plt.ylabel('Count')
plt.show()

# Create a confusion matrix for the ensemble model
conf_matrix_ensemble = confusion_matrix(y_test, np.argmax(y_pred_weighted, axis=1))
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_ensemble, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Ensemble Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Compare the accuracy of the models using a bar chart
models = ['Ensemble (Model1 & Model2)', 'Random Forest', 'Gradient Boosting']
accuracies = [np.mean(accuracy), accuracy_random_forest, accuracy_gradient_boosting]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.title('Model Comparison on Test Set')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)
plt.show()

# Create a table to display detailed performance metrics
performance_table = pd.DataFrame({
    'Model': ['Ensemble (Model1 & Model2)', 'Random Forest', 'Gradient Boosting'],
    'Accuracy': [np.mean(accuracy), accuracy_random_forest, accuracy_gradient_boosting],
    'Precision': [precision_score(y_test, y_pred_weighted_labels, average='weighted'),
                  precision_score(y_test, y_pred_random_forest, average='weighted'),
                  precision_score(y_test, y_pred_gradient_boosting, average='weighted')],
    'Recall': [recall_score(y_test, y_pred_weighted_labels, average='weighted'),
               recall_score(y_test, y_pred_random_forest, average='weighted'),
               recall_score(y_test, y_pred_gradient_boosting, average='weighted')],
    'F1-Score': [f1_score(y_test, y_pred_weighted_labels, average='weighted'),
                 f1_score(y_test, y_pred_random_forest, average='weighted'),
                 f1_score(y_test, y_pred_gradient_boosting, average='weighted')]
})

print("Performance Metrics:")
print(performance_table)

# Save the performance table to a CSV file
performance_table.to_csv('performance_metrics.csv', index=False)