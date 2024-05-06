import cv2
import os
import numpy as np
import face_recognition
import json
from datetime import datetime
import tkinter as tk
from tkinter import filedialog


def detect_faces_and_recognize():
    # Create a directory for storing the embeddings
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")

    # Load IDs from the filenames in the dataset directory
    dataset_ids = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir("dataset")]

    # Initialize attendance dictionary
    attendance = {id: {"last_seen": "", "dates": []} for id in dataset_ids}

    # Load the images
    image_paths = [os.path.join("dataset", f) for f in os.listdir("dataset")]
    embeddings = []

    for image_path in image_paths:
        image = cv2.imread(image_path)

        # Resize the image to a larger size to improve face detection in low-quality images
        image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # Convert the image to RGB format (required by face_recognition)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face locations in the image
        face_locations = face_recognition.face_locations(image_rgb)

        # Extract face encodings
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

        print(f"Found {len(face_encodings)} face(s) in {image_path}")
        for i, face_encoding in enumerate(face_encodings):
            print(f"Embedding {i + 1}: {face_encoding}")
            embeddings.append(face_encoding)

            # Recognize the face
            matches = face_recognition.compare_faces(embeddings, face_encoding)
            id = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                id = dataset_ids[match_index]

                # Update attendance information
                now = datetime.now()
                date_string = now.strftime("%Y-%m-%d %H:%M:%S")
                attendance[id]["last_seen"] = date_string
                attendance[id]["dates"].append(date_string)

    # Check if embeddings are collected
    if len(embeddings) == 0:
        print("No face embeddings were collected.")
    else:
        print(f"{len(embeddings)} face embeddings were collected.")

        # Save the embeddings
        with open("embeddings/embeddings.txt", "w") as file:
            for embedding in embeddings:
                embedding_str = ','.join(map(str, embedding))
                file.write(embedding_str + "\n")

    # Save attendance information to JSON file
    with open("attendance.json", "w") as json_file:
        json.dump(attendance, json_file, indent=4)

    print("Attendance information saved successfully.")


def select_and_detect():
    file_path = filedialog.askdirectory()
    if file_path:
        os.chdir(file_path)
        detect_faces_and_recognize()


def tkinter_gui():
    root = tk.Tk()
    root.title("Face Recognition")

    detect_button = tk.Button(root, text="Select Folder and Detect Faces", command=select_and_detect)
    detect_button.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    tkinter_gui()
