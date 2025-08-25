import pandas as pd
import os

# Define parent directory (replace with the correct parent dir path)
parent_dir = os.getcwd()  # Current directory for demonstration

# Define paths to dataset and save location for the model
dataset_path = os.path.join(parent_dir, 'data', 'PCOS_3080_DATASET.csv')

# Load the dataset
data = pd.read_csv(dataset_path)

# Clean column names by stripping any leading/trailing spaces
data.columns = data.columns.str.strip()

# Convert relevant columns to numeric, forcing errors to NaN
data['BMI'] = pd.to_numeric(data['BMI'], errors='coerce')
data['FSH/LH'] = pd.to_numeric(data['FSH/LH'], errors='coerce')
data['AMH(ng/mL)'] = pd.to_numeric(data['AMH(ng/mL)'], errors='coerce')
data['RBS(mg/dl)'] = pd.to_numeric(data['RBS(mg/dl)'], errors='coerce')

# Drop rows with NaN values in any of the columns (this ensures no NaN values remain)
data.dropna(subset=['BMI', 'FSH/LH', 'AMH(ng/mL)', 'RBS(mg/dl)'], inplace=True)

# Define a function to categorize PCOS stage based on multiple features
def categorize_pcos_stage(row):
    # Define conditions for Normal, Mild, Moderate, Severe based on multiple features
    
    # Normal: BMI < 20, FSH/LH < 2, AMH < 5, RBS < 100
    if row['BMI'] < 20 and row['FSH/LH'] < 2 and row['AMH(ng/mL)'] < 5 and row['RBS(mg/dl)'] < 100:
        return 'Normal'

    # Mild: 20 <= BMI < 24, 2 <= FSH/LH < 2.5, 5 <= AMH < 10, RBS < 110
    elif 20 <= row['BMI'] < 24 and 2 <= row['FSH/LH'] < 2.5 and 5 <= row['AMH(ng/mL)'] < 10 and row['RBS(mg/dl)'] < 110:
        return 'Mild'

    # Moderate: 24 <= BMI < 28, 2.5 <= FSH/LH < 3.5, 10 <= AMH < 20, RBS < 120
    elif 24 <= row['BMI'] < 28 and 2.5 <= row['FSH/LH'] < 3.5 and 10 <= row['AMH(ng/mL)'] < 20 and row['RBS(mg/dl)'] < 120:
        return 'Moderate'

    # Severe: BMI >= 28, FSH/LH >= 3.5, AMH >= 20, RBS >= 120
    elif row['BMI'] >= 28 or row['FSH/LH'] >= 3.5 or row['AMH(ng/mL)'] >= 20 or row['RBS(mg/dl)'] >= 120:
        return 'Severe'

    # Adding other cases to cover all combinations (with boundary conditions)
    else:
        # Catch remaining cases based on certain thresholds
        if row['BMI'] < 20:
            if row['FSH/LH'] < 2:
                return 'Normal'
            elif row['FSH/LH'] < 2.5:
                return 'Mild'
            else:
                return 'Moderate'

        elif 20 <= row['BMI'] < 24:
            if row['FSH/LH'] < 2:
                return 'Mild'
            elif row['FSH/LH'] < 3.5:
                return 'Moderate'
            else:
                return 'Severe'

        elif 24 <= row['BMI'] < 28:
            if row['FSH/LH'] < 2.5:
                return 'Moderate'
            elif row['FSH/LH'] < 3.5:
                return 'Severe'
            else:
                return 'Severe'

        else:
            return 'Severe'

# Apply this function to create the 'PCOS Stage' column
data['PCOS Stage'] = data.apply(categorize_pcos_stage, axis=1)

# Remove the 'PCOS (Y/N)' column if it exists
if 'PCOS (Y/N)' in data.columns:
    data.drop(columns=['PCOS (Y/N)'], inplace=True)

# Check the first few rows of the dataset to confirm
print(data[['Patient file no.', 'BMI', 'FSH/LH', 'AMH(ng/mL)', 'RBS(mg/dl)', 'PCOS Stage']].head())

# Save the dataset with the new 'PCOS Stage' column
data.to_csv('PCOS_3080_DATASET_with_stage.csv', index=False)

print("PCOS Stage has been added and 'PCOS (Y/N)' column has been removed successfully!")
