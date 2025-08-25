import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import joblib  # For saving the trained model
import matplotlib.pyplot as plt
import seaborn as sns

# Define parent directory (replace with the correct parent dir path)
current_dir = os.getcwd()  # Get the current working directory
parent_dir = os.path.dirname(current_dir)  # Go one step back (parent directory)

# Define paths to dataset and save location for the model
dataset_path = os.path.join(parent_dir, 'data', 'PCOS_3080_DATASET_with_stage.csv')
model_save_path = os.path.join(parent_dir, 'models', 'RF_BILSTM_Models')

# Load the dataset
data = pd.read_csv(dataset_path)

# Clean column names by stripping extra spaces (if any)
data.columns = data.columns.str.strip()

# Handle missing values in features (e.g., replace empty strings or spaces with NaN)
data = data.replace(r'^\s*$', pd.NA, regex=True)

# Select only the columns you want to use for training
selected_columns = [
    'Cycle(R/I)', 'Cycle length(days)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH',
    'AMH(ng/mL)', 'RBS(mg/dl)', 'Follicle No. (L)', 'Follicle No. (R)',
    'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)',
    'Waist/Hip Ratio', 'BMI', 'PCOS Stage'
]

data = data[selected_columns]

# Convert relevant columns to numeric, forcing errors to NaN
numeric_columns = selected_columns[:-1]  # Exclude 'PCOS Stage'

# Convert all numeric columns to proper numeric values
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Remove rows with any NaN values after conversion
data = data.dropna()

# Ensure 'PCOS Stage' exists and encode it
if "PCOS Stage" in data.columns:
    # Encode 'PCOS Stage' to numeric values
    label_encoder = LabelEncoder()
    data["PCOS Stage"] = label_encoder.fit_transform(data["PCOS Stage"])
else:
    raise ValueError("PCOS Stage column is missing in the dataset.")

# Split features and target
X = data.drop(columns=["PCOS Stage"])  # Features
y = data["PCOS Stage"]  # Target variable with encoded PCOS stages

# Normalize the data (important for neural networks)
X = (X - X.mean()) / X.std()

# Train-test split (70% training, 30% testing)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# **Step 1: Train the Random Forest Model with various percentages of training data (40%, 50%, 60%, 70%, 80%)**
training_percentages = [40, 50, 60, 70, 80]

for percent in training_percentages:
    # Select the percentage of training data
    train_size = int(percent / 100 * len(X_train_full))  # Calculate the number of samples based on percentage
    X_train = X_train_full[:train_size]
    y_train = y_train_full[:train_size]
    
    # **Step 2: Train the Random Forest Model**
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Extract Random Forest predictions (probabilities)
    rf_train_probs = rf_model.predict_proba(X_train)  # Get the probability for each class
    rf_test_probs = rf_model.predict_proba(X_test)

    # **Step 3: Build the Combined RF + BiLSTM Model**
    # Define the input layers for both RF output and original features
    input_features = Input(shape=(X_train.shape[1],))  # Original features
    input_rf_output = Input(shape=(rf_train_probs.shape[1],))  # RF probabilities

    # RF part (no training required, just pass through)
    rf_output = Dense(32, activation='relu')(input_rf_output)

    # Combine both features (original + RF probabilities)
    combined = tf.keras.layers.concatenate([input_features, rf_output])

    # Reshape to be 3D for LSTM (add the timestep dimension)
    combined_reshaped = tf.keras.layers.Reshape((1, combined.shape[1]))(combined)

    # Add BiLSTM layers
    x = Bidirectional(LSTM(64, return_sequences=True))(combined_reshaped)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.5)(x)

    # Dense layers for classification
    x = Dense(32, activation='relu')(x)
    output = Dense(len(label_encoder.classes_), activation='softmax')(x)  # For multi-class classification

    # Build the combined model
    combined_model = Model(inputs=[input_features, input_rf_output], outputs=output)

    # Compile the model
    combined_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # **Step 4: Train the Combined Model**
    history = combined_model.fit([X_train, rf_train_probs], y_train, epochs=20, batch_size=32, validation_data=([X_test, rf_test_probs], y_test))

    # **Step 5: Evaluate the Combined Model**
    y_pred = np.argmax(combined_model.predict([X_test, rf_test_probs]), axis=-1)

    # Calculate accuracy on the test set
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy for {percent}% training data: {accuracy:.4f}")

    # Print detailed classification report
    print(f"Classification Report for {percent}% training data:\n", classification_report(y_test, y_pred))

    # Compute and print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for {percent}% training data:\n", cm)

    # Plot training & validation accuracy values
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Combined RF + BiLSTM Model Accuracy ({percent}% Training Data)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Combined RF + BiLSTM Model Loss ({percent}% Training Data)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix ({percent}% Training Data)')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Save the model for the current training percentage
    model_save_file = os.path.join(model_save_path, f'rf_bilstm_model_{percent}pct.h5')
    combined_model.save(model_save_file)  # Save the combined model
    print(f"Model saved at {model_save_file}")

    # Save the label encoder for the current training percentage
    label_encoder_save_file = os.path.join(model_save_path, f'label_encoder_{percent}pct.pkl')
    joblib.dump(label_encoder, label_encoder_save_file)
    print(f"Label encoder saved at {label_encoder_save_file}")
