import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Updated function to load images and assign labels for 4 stages
def load_data(img_dir):
    images = []
    labels = []
    stage_labels = {
        'normal': 0,
        'mild': 1,
        'moderate': 2,
        'severe': 3
    }
    
    for stage, label in stage_labels.items():
        folder = os.path.join(img_dir, stage)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            
            # Try reading the image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Failed to load image {img_path}. Skipping...")
                continue  # Skip this image if it failed to load
            
            img = cv2.resize(img, (128, 128))  # Resize to fit model input
            img = img / 255.0  # Normalize the image
            img = np.expand_dims(img, axis=-1)  # Add a channel dimension for grayscale
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Feature extraction using DCNN for BiLSTM
def extract_features_dcnn(input_shape=(128, 128, 1)):
    model = models.Sequential()
    
    # DCNN Layers (Deeper architecture)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Flatten())
    
    return model

# BiLSTM model
def build_bilstm_model(input_shape=(128, 128, 1)):
    dcnn_model = extract_features_dcnn(input_shape)  # Feature extraction using DCNN
    dcnn_model.trainable = False  # Freeze the DCNN layers
    
    model = models.Sequential()
    
    # Use the DCNN base model
    model.add(dcnn_model)
    
    # Add BiLSTM layer for sequence processing
    model.add(layers.Reshape((-1, dcnn_model.output_shape[1])))  # Reshape output to feed into LSTM
    model.add(layers.Bidirectional(layers.LSTM(64)))  # BiLSTM layer
    
    # Dense layer for classification
    model.add(layers.Dense(4, activation='softmax'))  # 4 classes: normal, mild, moderate, severe
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Train the BiLSTM model
def train_bilstm(img_dir, model_save_path, training_percentages):
    X, y = load_data(img_dir)
    
    # Split the data into training and validation sets (70-30 split)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Augment the training data
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    datagen.fit(X_train)

    # Loop over different training percentages
    for percent in training_percentages:
        print(f"\nTraining with {percent}% of data...")

        # Select a subset of the training data
        num_samples = int(len(X_train) * percent / 100)
        X_train_subset = X_train[:num_samples]
        y_train_subset = y_train[:num_samples]

        # Build and train the BiLSTM model
        model = build_bilstm_model()

        # Train the model with data augmentation
        history = model.fit(datagen.flow(X_train_subset, y_train_subset, batch_size=32), epochs=20, validation_data=(X_val, y_val))

        # Evaluate on the validation data (or a separate test set if available)
        loss, accuracy = model.evaluate(X_val, y_val, verbose=1)
        print(f"Validation Loss: {loss}")
        print(f"Validation Accuracy: {accuracy}")
        
        # Save the model
        model_save_file = os.path.join(model_save_path, f'dcnn+bilstm_model_{percent}pct.h5')
        model.save(model_save_file)

        # Plot the loss and accuracy graphs
        plot_metrics(history)

        # Optionally: print detailed performance metrics (precision, recall, F1 score, etc.)
        y_pred = np.argmax(model.predict(X_val), axis=-1)  # Predict the class indices
        print("Classification Report:\n", classification_report(y_val, y_pred))
        
        # Compute and print confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        plot_confusion_matrix(cm)

# Plot accuracy and loss
def plot_metrics(history):
    # Plot accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Mild', 'Moderate', 'Severe'], yticklabels=['Normal', 'Mild', 'Moderate', 'Severe'])
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == '__main__':
    current_dir = os.getcwd()  # Get the current working directory
    parent_dir = os.path.dirname(current_dir)  # Go one step back (parent directory)

    train_bilstm(
        os.path.join(parent_dir, 'images', 'train_70%_data'),  # Make sure to point to the right folder containing the 4 stages
        os.path.join(parent_dir, 'models', 'DCNN_BILSTM_Models'),  # Save models in models directory
        training_percentages=[40, 50, 60, 70, 80]  # Train with different data percentages
    )
