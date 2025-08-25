import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import joblib  # For saving the trained model
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping

# Define parent directory (replace with the correct parent dir path)
current_dir = os.getcwd()  # Get the current working directory
parent_dir = os.path.dirname(current_dir)  # Go one step back (parent directory)

# Define paths to dataset and save location for the model
dataset_path = os.path.join(parent_dir, 'data', 'PCOS_3080_DATASET_with_stage.csv')
model_save_path = os.path.join(parent_dir, 'models', 'BILSTM_Models')

# Load the dataset
data = pd.read_csv(dataset_path)

# Clean column names by stripping extra spaces (if any)
data.columns = data.columns.str.strip()

# Handle missing values in features (e.g., replace empty strings or spaces with NaN)
data = data.replace(r'^\s*$', pd.NA, regex=True)

# Convert relevant columns to numeric, forcing errors to NaN
numeric_columns = ['Cycle(R/I)', 'Cycle length(days)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH',
                   'AMH(ng/mL)', 'RBS(mg/dl)', 'Follicle No. (L)', 'Follicle No. (R)',
                   'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)',
                   'Waist/Hip Ratio', 'BMI', 'PCOS Stage']

# Select only the relevant columns for training
data = data[numeric_columns]

# Convert all numeric columns to proper numeric values
for col in numeric_columns[:-1]:  # Exclude 'PCOS Stage' for this
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Remove rows with any NaN values after conversion
data = data.dropna()

# Ensure 'PCOS Stage' exists and encode it
if "PCOS Stage" in data.columns:
    # Check the unique values before encoding
    print("Unique values in 'PCOS Stage':", data["PCOS Stage"].unique())
    
    # Encode 'PCOS Stage' to numeric values
    label_encoder = LabelEncoder()
    data["PCOS Stage"] = label_encoder.fit_transform(data["PCOS Stage"])
    
    # Check the classes learned by the encoder
    print("Classes learned by the encoder:", label_encoder.classes_)
else:
    raise ValueError("PCOS Stage column is missing in the dataset.")

# Split features and target
X = data.drop(columns=["PCOS Stage"])  # Features
y = data["PCOS Stage"]  # Target variable with encoded PCOS stages

# Normalize the data using StandardScaler (instead of (X - X.mean()) / X.std())
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into 70% training and 30% validation/test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define training percentages to test on 70% training data (40%, 50%, 60%, 70%, 80%)
training_percentages = [40, 50, 60, 70, 80]

for percent in training_percentages:
    print(f"\nTraining with {percent}% of the training data...\n")
    
    # Split the 70% training data into smaller portions for training
    X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, test_size=(100 - percent) / 100, random_state=42)
    
    # Reshape data for LSTM (adding an extra dimension for the sequence length)
    X_train_sub = np.expand_dims(X_train_sub, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # Build the BiLSTM model with more layers, batch normalization, and tuning
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train_sub.shape[1], 1)))  # Increased units
    model.add(Dropout(0.4))  # Adjusted dropout
    model.add(BatchNormalization())  # Added Batch Normalization
    model.add(LSTM(64, return_sequences=False))  # Added another LSTM layer
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))  # Increased dense layer size
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # For multi-class classification
    
    # Compile the model with an adjusted learning rate
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # EarlyStopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit(X_train_sub, y_train_sub, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    
    # Evaluate the model on the test set
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    
    # Calculate accuracy on the test set
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Compute and print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    
    # Plot training & validation accuracy values
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy - {percent}% Training Data')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    
    # Plot training & validation loss values
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - {percent}% Training Data')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    # Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {percent}% Training Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Save the trained model and label encoder for the current training percentage
    model_save_file = os.path.join(model_save_path, f'bilstm_model_{percent}pct.h5')
    model.save(model_save_file)  # Save the model as a .h5 file
    joblib.dump(label_encoder, os.path.join(model_save_path, f'label_encoder_{percent}pct.pkl'))  # Save the encoder
    
    print(f"Model trained and saved successfully at {model_save_file} and label_encoder_{percent}pct.pkl.")
