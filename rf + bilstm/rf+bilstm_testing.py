import os
import pandas as pd
import numpy as np
import joblib
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define parent directory (replace with the correct parent dir path)
current_dir = os.getcwd()  # Get the current working directory
parent_dir = os.path.dirname(current_dir)  # Go one step back (parent directory)

# Define paths to the saved model and label encoder
combined_model_save_path = os.path.join(parent_dir, 'models', 'RF_BILSTM_Models', 'rf_bilstm_model_40pct.h5')
encoder_save_path = os.path.join(parent_dir, 'models', 'RF_BILSTM_Models','label_encoder_40pct.pkl')

# Load the trained combined model and label encoder
combined_model = load_model(combined_model_save_path)
label_encoder = joblib.load(encoder_save_path)

# Input fields corresponding to the features used in training (updated list)
input_fields = [
    'Cycle(R/I)', 'Cycle length(days)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 
    'AMH(ng/mL)', 'RBS(mg/dl)', 'Follicle No. (L)', 'Follicle No. (R)',
    'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)',
    'Waist/Hip Ratio', 'BMI'
]

# Function to generate the report
def generate_report():
    input_data = []
    
    # Collect inputs from the user via the UI
    for field in input_fields:
        try:
            value = float(entry_dict[field].get())
            input_data.append(value)
        except ValueError:
            messagebox.showerror("Error", f"Invalid input for {field}. Please enter a valid number.")
            return
    
    # Convert to a numpy array and reshape for prediction
    user_input = np.array(input_data).reshape(1, -1)
    
    # Normalize the input data (same as during training)
    user_input_normalized = (user_input - np.mean(user_input, axis=0)) / np.std(user_input, axis=0)
    
    # Preprocess the input data for the BiLSTM model (reshape as required)
    user_input_reshaped = np.expand_dims(user_input_normalized, axis=1)  # Reshape for LSTM input
    
    # Predict the PCOS stage using the trained combined model (RF + BiLSTM)
    predicted_stage = combined_model.predict(user_input_reshaped)
    
    # Convert the output into a class label (based on the model's output layer)
    predicted_stage_label = np.argmax(predicted_stage, axis=1)
    
    # Decode the predicted stage
    predicted_stage_label = label_encoder.inverse_transform(predicted_stage_label)
    
    # Prepare the report text
    report_text = f"""
    Predicted PCOS Stage: {predicted_stage_label[0]}
    """
    
    # Display the report in a popup
    show_report_popup(report_text)

# Function to show the report popup
def show_report_popup(report_text):
    # Create a new top-level window for the report popup
    report_popup = tk.Toplevel(root)
    report_popup.title("PCOS Stage Prediction Report")
    report_popup.geometry("450x300")  # Set the size of the popup window

    # Set a dark background and white text
    report_popup.configure(bg="#2e2e2e")

    # Label to display the report
    report_label = tk.Label(report_popup, text=report_text, font=("Helvetica", 12), fg="#FFFFFF", bg="#2e2e2e", justify="left", padx=10, pady=10)
    report_label.pack(pady=20, padx=20)

    # Close button
    btn_close = tk.Button(report_popup, text="Close", font=("Helvetica", 12), bg="#4CAF50", fg="#FFFFFF", command=report_popup.destroy)
    btn_close.pack(pady=10)

# Create the main window
root = tk.Tk()
root.title("PCOS Stage Prediction")

# Maximize the window on startup
root.state('zoomed')

# Set dark theme colors
root.configure(bg="#2e2e2e")

# Define the font and style
font_style = ("Helvetica", 14)  # Increased font size for better readability
highlight_color = "#4CAF50"  # Button color for contrast
light_text_color = "#FFFFFF"
dark_bg = "#2e2e2e"
input_bg = "#555555"

# Create a title label for the application
title_label = tk.Label(root, text="PCOS Stage Prediction System", font=("Helvetica", 24), fg=highlight_color, bg=dark_bg)
title_label.grid(row=0, column=0, columnspan=3, pady=20, padx=20)

# Create frames for sections (Left, Center, Right)
frame_details_left = tk.Frame(root, bg=dark_bg)
frame_details_left.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

frame_details_center = tk.Frame(root, bg=dark_bg)
frame_details_center.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")

frame_details_right = tk.Frame(root, bg=dark_bg)
frame_details_right.grid(row=1, column=2, padx=20, pady=20, sticky="nsew")

# Set the grid weights to make frames expand equally
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)

# Dictionary to store entry widgets for each input field
entry_dict = {}

# Split the input fields into three parts (5-5-4)
input_fields_left = input_fields[:5]
input_fields_center = input_fields[5:10]
input_fields_right = input_fields[10:]

# Define the smaller font size for input fields
font_style = ("Helvetica", 12)  # Smaller font size for input fields
highlight_color = "#4CAF50"  # Button color for contrast
light_text_color = "#FFFFFF"
dark_bg = "#2e2e2e"
input_bg = "#555555"

# Patient Details Frame (Left side)
for idx, field in enumerate(input_fields_left):
    label = tk.Label(frame_details_left, text=field + ":", font=font_style, bg=dark_bg, fg=light_text_color)
    label.grid(row=idx, column=0, padx=15, pady=15, sticky="w")
    
    entry = tk.Entry(frame_details_left, font=font_style, bg=input_bg, fg=light_text_color, relief="solid", width=30)
    entry.grid(row=idx, column=1, padx=15, pady=15)
    
    entry_dict[field] = entry  # Store the entry widget in the dictionary

# Patient Details Frame (Center side)
for idx, field in enumerate(input_fields_center):
    label = tk.Label(frame_details_center, text=field + ":", font=font_style, bg=dark_bg, fg=light_text_color)
    label.grid(row=idx, column=0, padx=15, pady=15, sticky="w")
    
    entry = tk.Entry(frame_details_center, font=font_style, bg=input_bg, fg=light_text_color, relief="solid", width=30)
    entry.grid(row=idx, column=1, padx=15, pady=15)
    
    entry_dict[field] = entry  # Store the entry widget in the dictionary

# Patient Details Frame (Right side)
for idx, field in enumerate(input_fields_right):
    label = tk.Label(frame_details_right, text=field + ":", font=font_style, bg=dark_bg, fg=light_text_color)
    label.grid(row=idx, column=0, padx=15, pady=15, sticky="w")
    
    entry = tk.Entry(frame_details_right, font=font_style, bg=input_bg, fg=light_text_color, relief="solid", width=30)
    entry.grid(row=idx, column=1, padx=15, pady=15)
    
    entry_dict[field] = entry  # Store the entry widget in the dictionary

# Continue Button to generate the report (Centered across the whole window)
btn_continue = tk.Button(root, text="Generate Report", command=generate_report, font=("Helvetica", 14), bg=highlight_color, fg=light_text_color, relief="flat", width=20)
btn_continue.grid(row=2, column=0, columnspan=3, pady=30)

# Run the application
root.mainloop()
