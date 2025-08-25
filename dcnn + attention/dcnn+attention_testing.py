import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.utils import get_custom_objects
from cbam import CBAM  # Import CBAM class from cbam.py

# Global variable to hold the model loaded once
model = None

def load_model_once(model_path):
    global model
    if model is None:
        model = load_model(model_path, custom_objects={'CBAM': CBAM})
        print("Model loaded successfully.")
    return model

def load_and_predict(img_path, model):
    try:
        # Load and preprocess the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Grayscale images
        if img is None:
            raise ValueError("Invalid image path")
        img = cv2.resize(img, (128, 128))  # Resize to match model input
        img = img / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make the prediction
        prediction = model.predict(img)
        class_idx = np.argmax(prediction)  # Get the index of the highest probability
        confidence = np.max(prediction)  # Confidence of the prediction

        stages = ['Normal', 'Mild', 'Moderate', 'Severe']
        predicted_stage = stages[class_idx]
        
        return predicted_stage, confidence
    except Exception as e:
        return f"Error: {str(e)}", 0.0

def select_image():
    # Open a file dialog to select an image
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if img_path:
        # Display the selected image in the GUI
        show_image(img_path)
        img_path_var.set(img_path)  # Store the image path

def show_image(img_path):
    # Open the image using PIL
    img = Image.open(img_path)
    img = img.resize((200, 200))  # Resize to fit in the window
    img = ImageTk.PhotoImage(img)
    
    # Create a label to display the image
    image_label.config(image=img)
    image_label.image = img  # Keep a reference to the image object

def generate_report():
    patient_name = entry_name.get()
    patient_age = entry_age.get()
    patient_contact = entry_contact.get()
    patient_address = entry_address.get()
    
    img_path = img_path_var.get()
    
    if not patient_name or not patient_age or not patient_contact or not patient_address or not img_path:
        messagebox.showerror("Error", "Please fill in all details and select an image.")
        return
    
    # Load model once
    # Load the model
    current_dir = os.getcwd()  # Get the current working directory
    parent_dir = os.path.dirname(current_dir)  # Go one step back (parent directory)
    
    model_path = os.path.join(parent_dir, 'models', 'DCNN_Attention_Models', 'dcnn+attention_model_40pct.h5')
    model = load_model_once(model_path)

    # Load and predict the image
    predicted_stage, confidence = load_and_predict(img_path, model)
    
    # Prepare the patient report
    report_text = f"""
    Patient Name: {patient_name}
    Age: {patient_age}
    Contact: {patient_contact}
    Address: {patient_address}
    
    Predicted Stage: {predicted_stage}
    Confidence: {confidence*100:.2f}%
    """
    
    # Display the report in a popup
    show_report_popup(report_text)

def show_report_popup(report_text):
    # Create a new top-level window for the report popup
    report_popup = tk.Toplevel(root)
    report_popup.title("Patient Report")
    report_popup.geometry("450x400")  # Set the size of the popup window

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
title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=20)

# Create frames for sections
frame_details = tk.Frame(root, bg=dark_bg)
frame_details.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

frame_image = tk.Frame(root, bg=dark_bg)
frame_image.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")

# Set the grid weights to make frames expand equally
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Patient Details Frame (Left side)
label_name = tk.Label(frame_details, text="Patient Name:", font=font_style, bg=dark_bg, fg=light_text_color)
label_name.grid(row=0, column=0, padx=15, pady=15, sticky="w")
entry_name = tk.Entry(frame_details, font=font_style, bg=input_bg, fg=light_text_color, relief="solid", width=30)
entry_name.grid(row=0, column=1, padx=15, pady=15)

label_age = tk.Label(frame_details, text="Age:", font=font_style, bg=dark_bg, fg=light_text_color)
label_age.grid(row=1, column=0, padx=15, pady=15, sticky="w")
entry_age = tk.Entry(frame_details, font=font_style, bg=input_bg, fg=light_text_color, relief="solid", width=30)
entry_age.grid(row=1, column=1, padx=15, pady=15)

label_contact = tk.Label(frame_details, text="Contact:", font=font_style, bg=dark_bg, fg=light_text_color)
label_contact.grid(row=2, column=0, padx=15, pady=15, sticky="w")
entry_contact = tk.Entry(frame_details, font=font_style, bg=input_bg, fg=light_text_color, relief="solid", width=30)
entry_contact.grid(row=2, column=1, padx=15, pady=15)

label_address = tk.Label(frame_details, text="Address:", font=font_style, bg=dark_bg, fg=light_text_color)
label_address.grid(row=3, column=0, padx=15, pady=15, sticky="w")
entry_address = tk.Entry(frame_details, font=font_style, bg=input_bg, fg=light_text_color, relief="solid", width=30)
entry_address.grid(row=3, column=1, padx=15, pady=15)

# Image Selection Section (Right side)
btn_select = tk.Button(frame_image, text="Select Image", command=select_image, font=("Helvetica", 14), bg=highlight_color, fg=light_text_color, relief="flat", width=20)
btn_select.pack(pady=20)

image_label = tk.Label(frame_image, bg=dark_bg)
image_label.pack(pady=10)

# Variable to store the image path
img_path_var = tk.StringVar()

# Continue Button to generate the report (Centered across the whole window)
btn_continue = tk.Button(root, text="Generate Report", command=generate_report, font=("Helvetica", 14), bg=highlight_color, fg=light_text_color, relief="flat", width=20)
btn_continue.grid(row=2, column=0, columnspan=2, pady=30)

# Run the application
root.mainloop()
