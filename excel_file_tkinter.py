import os
import pandas as pd
import json
import tkinter as tk
from tkinter import filedialog, messagebox

def generate_attendance_log():
    # Path to the dataset folder containing images
    dataset_folder = entry_dataset_folder.get()

    if not dataset_folder:
        messagebox.showerror("Error", "Please select dataset folder.")
        return

    # List all files in the dataset folder
    image_files = os.listdir(dataset_folder)

    # Extract IDs from filenames (assuming filenames are the IDs)
    ids = sorted([os.path.splitext(file)[0] for file in image_files])  # Sort IDs

    # Save IDs to ids.txt file
    with open('ids.txt', 'w') as file:
        file.write('\n'.join(ids))

    # Load the attendance log from the JSON file
    try:
        with open('attendance.json', 'r') as f:
            attendance_log = json.load(f)
    except FileNotFoundError:
        attendance_log = {}

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

    if attendance_data:
        # Convert the attendance data to a Pandas DataFrame
        df = pd.DataFrame(attendance_data)

        # Sort DataFrame by 'ID' column
        df = df.sort_values(by='ID')

        # Write the attendance data to an Excel file, overwriting any existing data
        output_file = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                    filetypes=[("Excel files", "*.xlsx")],
                                                    title="Save Attendance Log As")
        if output_file:
            with pd.ExcelWriter(output_file, engine='xlsxwriter', mode='w') as writer:
                df.to_excel(writer, index=False)

                # Set the column widths manually
                worksheet = writer.sheets['Sheet1']
                worksheet.set_column('A:A', 20)  # Adjust the width of column 'A' to 20
                worksheet.set_column('B:B', 20)  # Adjust the width of column 'B' to 20
                worksheet.set_column('C:C', 20)  # Adjust the width of column 'C' to 20
            messagebox.showinfo("Success", "Attendance log generated successfully.")
        else:
            messagebox.showerror("Error", "Please select output file path.")
    else:
        messagebox.showwarning("Warning", "No attendance data available.")

# Create a Tkinter window
root = tk.Tk()
root.title("Attendance Log Generator")

# Label and Entry for Dataset Folder
label_dataset_folder = tk.Label(root, text="Dataset Folder:")
label_dataset_folder.grid(row=0, column=0, padx=5, pady=5, sticky="w")
entry_dataset_folder = tk.Entry(root, width=50)
entry_dataset_folder.grid(row=0, column=1, padx=5, pady=5, sticky="we")
entry_dataset_folder.insert(0, os.getcwd())  # Set default folder to current directory

def browse_dataset_folder():
    folder_path = filedialog.askdirectory()
    entry_dataset_folder.delete(0, tk.END)
    entry_dataset_folder.insert(0, folder_path)

button_browse_dataset = tk.Button(root, text="Browse", command=browse_dataset_folder)
button_browse_dataset.grid(row=0, column=2, padx=5, pady=5)

# Button to generate attendance log
button_generate_log = tk.Button(root, text="Generate Attendance Log", command=generate_attendance_log)
button_generate_log.grid(row=1, column=0, columnspan=3, pady=10)

# Run the Tkinter event loop
root.mainloop()
