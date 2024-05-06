import os
import pandas as pd
import json

# Path to the dataset folder containing images
dataset_folder = 'dataset'

# List all files in the dataset folder
image_files = os.listdir(dataset_folder)

# Extract IDs from filenames (assuming filenames are the IDs)
ids = sorted([os.path.splitext(file)[0] for file in image_files])  # Sort IDs

# Save IDs to ids.txt file
with open('ids.txt', 'w') as file:
    file.write('\n'.join(ids))

# Load the attendance log from the JSON file
with open('attendance.json', 'r') as f:
    attendance_log = json.load(f)

# Load detected faces from detected_faces.txt file
with open('detected_faces.txt', 'r') as f:
    detected_faces = [line.strip() for line in f]

# Initialize an empty list to store attendance data
attendance_data = []

# Iterate through the IDs and attendance log
for id in ids:
    if id in attendance_log:
        # Get the last seen timestamp for the ID
        last_seen_timestamp = attendance_log[id].get("last_seen")
        # Check if the ID exists in the detected faces
        attendance_status = 'Present' if id in detected_faces else 'Absent'
        # Append the attendance record to the list
        attendance_data.append({'ID': id, 'Date': last_seen_timestamp, 'Attendance': attendance_status})

# Convert the attendance data to a Pandas DataFrame
df = pd.DataFrame(attendance_data)

print(df)  # Add this line to check the structure of your DataFrame

# Sort DataFrame by 'ID' column if it exists
if 'ID' in df.columns:
    df = df.sort_values(by='ID')
    # Write the attendance data to an Excel file, overwriting any existing data
    with pd.ExcelWriter('attendance_log.xlsx', engine='xlsxwriter', mode='w') as writer:
        df.to_excel(writer, index=False)

        # Set the column widths manually
        worksheet = writer.sheets['Sheet1']
        worksheet.set_column('A:A', 20)  # Adjust the width of column 'A' to 20
        worksheet.set_column('B:B', 20)  # Adjust the width of column 'B' to 20
        worksheet.set_column('C:C', 20)  # Adjust the width of column 'C' to 20
else:
    print("DataFrame does not contain 'ID' column.")
