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

# CBAM Block Definition
class CBAM(layers.Layer):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = self._build_channel_attention(channels, reduction_ratio)
        self.spatial_attention = self._build_spatial_attention()

    def _build_channel_attention(self, channels, reduction_ratio):
        return models.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Reshape((1, 1, channels)),
            layers.Dense(channels // reduction_ratio, activation='relu'),
            layers.Dense(channels, activation='sigmoid')
        ])

    def _build_spatial_attention(self):
        return models.Sequential([
            layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')
        ])

    def call(self, inputs):
        # Apply Channel Attention
        channel_attention = self.channel_attention(inputs)
        x = layers.Multiply()([inputs, channel_attention])

        # Apply Spatial Attention
        spatial_attention = self.spatial_attention(x)
        x = layers.Multiply()([x, spatial_attention])

        return x

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

def build_dcnn_attention_bilstm_model():
    model = models.Sequential()

    # Add more convolutional layers to make it deeper with CBAM integration
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
    model.add(CBAM(32))  # Add CBAM after the convolutional layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(CBAM(64))  # Add CBAM after the convolutional layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(CBAM(128))  # Add CBAM after the convolutional layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(CBAM(256))  # Add CBAM after the convolutional layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(CBAM(512))  # Add CBAM after the convolutional layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten the output of the convolutional layers
    model.add(layers.Flatten())
    
    # Reshape the output of the CNN layers to feed into the BiLSTM layer
    model.add(layers.Reshape((1, -1)))  # Reshape for BiLSTM
    
    # Add Bidirectional LSTM
    model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=False)))  # BiLSTM
    
    # Add a dense layer with more neurons to increase complexity
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))  # Add dropout for regularization
    
    # Output layer for 4-class classification
    model.add(layers.Dense(4, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_and_evaluate(img_dir, model_save_path, training_percentages):
    X, y = load_data(img_dir)
    
    # Split the data into training and validation sets (70-30 split)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Prepare to store results
    results = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "specificity": [],
        "sensitivity": [],
        "loss": []
    }

    # Loop over different training percentages
    for percent in training_percentages:
        print(f"\nTraining with {percent}% of the data...\n")
        
        # Select a subset of the training data
        num_samples = int(len(X_train) * percent / 100)
        X_train_subset = X_train[:num_samples]
        y_train_subset = y_train[:num_samples]

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
        datagen.fit(X_train_subset)

        # Build and train the Deep CNN model with CBAM and BiLSTM
        model = build_dcnn_attention_bilstm_model()

        # Train the model with data augmentation
        history = model.fit(datagen.flow(X_train_subset, y_train_subset, batch_size=32), epochs=30, validation_data=(X_val, y_val))

        # Evaluate on the validation data
        y_pred = np.argmax(model.predict(X_val), axis=-1)

        # Calculate performance metrics
        report = classification_report(y_val, y_pred, output_dict=True)
        cm = confusion_matrix(y_val, y_pred)
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)

        # Sensitivity and Specificity
        sensitivity = report['macro avg']['recall']
        specificity = report['macro avg']['precision']

        # Store results
        results["accuracy"].append(accuracy)
        results["precision"].append(report['macro avg']['precision'])
        results["recall"].append(report['macro avg']['recall'])
        results["f1_score"].append(report['macro avg']['f1-score'])
        results["specificity"].append(specificity)
        results["sensitivity"].append(sensitivity)
        results["loss"].append(loss)

        # Save model after each training
        model.save(os.path.join(model_save_path, f"dcnn_attention_bilstm_model_{percent}pct.h5"))
        
        # Plot accuracy and loss graphs
        plot_metrics(history)

        # Plot confusion matrix
        plot_confusion_matrix(cm, [0, 1, 2, 3], f"Confusion Matrix ({percent}% data)")

        # Print Classification Report
        print(f"Classification Report for {percent}% data:\n", classification_report(y_val, y_pred))

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

def plot_confusion_matrix(cm, labels, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == '__main__':
    current_dir = os.getcwd()  # Get the current working directory
    parent_dir = os.path.dirname(current_dir)  # Go one step back (parent directory)

    train_and_evaluate(
        os.path.join(parent_dir, 'images', 'train_70%_data'),  # Make sure to point to the right folder containing the 4 stages
        os.path.join(parent_dir, 'models', 'DCNN_BILSTM_Attentions_Models'),  # Save the models in the models directory
        training_percentages=[40, 50, 60, 70, 80]  # Train with different data percentages
    )

