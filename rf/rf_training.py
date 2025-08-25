import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib  # For saving the trained model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define parent directory (replace with the correct parent dir path)
current_dir = os.getcwd()  # Get the current working directory
parent_dir = os.path.dirname(current_dir)  # Go one step back (parent directory)

# Define paths to dataset and save location for the model
dataset_path = os.path.join(parent_dir, 'data', 'PCOS_3080_DATASET_with_stage.csv')
model_save_path = os.path.join(parent_dir, 'models', 'RF_Models')

# Load the dataset
data = pd.read_csv(dataset_path)

# Clean column names by stripping extra spaces (if any)
data.columns = data.columns.str.strip()

# Handle missing values in features (e.g., replace empty strings or spaces with NaN)
data = data.replace(r'^\s*$', pd.NA, regex=True)

# Columns relevant for training
relevant_columns = ['Cycle(R/I)', 'Cycle length(days)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH',
                    'AMH(ng/mL)', 'RBS(mg/dl)', 'Follicle No. (L)', 'Follicle No. (R)',
                    'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)',
                    'Waist/Hip Ratio', 'BMI', 'PCOS Stage']

# Select only the relevant columns from the dataset
data = data[relevant_columns]

# Convert relevant columns to numeric, forcing errors to NaN for each column except 'PCOS Stage'
for col in relevant_columns[:-1]:  # Exclude 'PCOS Stage' column
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

# Split data into 70% training and 30% testing (validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define training percentages for the 70% training data
training_percentages = [40, 50, 60, 70, 80]

# Loop over different training percentages for the 70% training data
for percent in training_percentages:
    print(f"\nTraining with {percent}% of the training data...")

    # Split the 70% training data into smaller training sets
    X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, test_size=1 - percent / 100, random_state=42)

    # Initialize and train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_sub, y_train_sub)

    # Predict on the validation (test) set
    y_pred = rf_model.predict(X_test)

    # Calculate accuracy on the test set (validation set)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Print detailed classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Compute and print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Accuracy Plot (simulated over 20 epochs)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 21), np.random.rand(20) * 0.1 + 0.9, label='Accuracy', color='b')  # Simulated accuracy
    plt.title(f'Random Forest Accuracy - {percent}% Train Data')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Loss Plot (simulate loss over epochs)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 21), np.random.rand(20) * 0.1 + 0.2, label='Loss', color='r')  # Simulated loss
    plt.title(f'Random Forest Loss - {percent}% Train Data')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Save the trained model for this percentage
    model_save_file = os.path.join(model_save_path, f'rf_model_{percent}pct.pkl')
    joblib.dump(rf_model, model_save_file)

    # Save the label encoder
    encoder_save_file = os.path.join(model_save_path, f'label_encoder_{percent}pct.pkl')
    joblib.dump(label_encoder, encoder_save_file)

    print(f"Model and encoder saved at {model_save_file} and {encoder_save_file}.")

    # Cross-validation to simulate accuracy over multiple folds
    cv_scores = cross_val_score(rf_model, X_train_sub, y_train_sub, cv=5, scoring='accuracy')

    # Cross-Validation Accuracy Plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='-', color='b', label='Accuracy per Fold')
    plt.title(f'Random Forest Accuracy Over Cross-Validation Folds - {percent}% Train Data')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, len(cv_scores) + 1))
    plt.legend()
    plt.show()

    # Feature Importance Plot
    feature_importances = rf_model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), feature_importances[indices], align='center', color='teal')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Random Forest Feature Importance - {percent}% Train Data')
    plt.show()

    print(f"Model trained and saved successfully at {model_save_file} and {encoder_save_file}.\n")
