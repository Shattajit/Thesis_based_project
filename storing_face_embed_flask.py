from flask import Flask, render_template, request
import cv2
import os
import numpy as np
import face_recognition
import json
from datetime import datetime

app = Flask(__name__)

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load IDs from the filenames in the dataset directory
def load_dataset_ids():
    return [os.path.splitext(os.path.basename(f))[0] for f in os.listdir("dataset")]

# Load embeddings from the embeddings file
def load_embeddings():
    embeddings_file = "embeddings/embeddings.txt"
    embeddings = []
    if os.path.exists(embeddings_file):
        with open(embeddings_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                embedding = [float(x) for x in line.strip().split(',')]
                embeddings.append(embedding)
    return embeddings

# Load attendance information from the JSON file
def load_attendance():
    attendance_file = "attendance.json"
    attendance = {}
    if os.path.exists(attendance_file):
        with open(attendance_file, "r") as json_file:
            attendance = json.load(json_file)
    return attendance

# Function to save attendance information to the JSON file
def save_attendance(attendance):
    with open("attendance.json", "w") as json_file:
        json.dump(attendance, json_file, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    output_folder = "dataset"
    person_name = request.form['person_name']
    person_id = request.form['person_id']
    num_images = 5
    ip_camera_url = 'https://192.168.1.10:8080/shot.jpg'  # Replace with your IP camera address
    capture_images(output_folder, person_name, num_images, ip_camera_url)
    return "Images Captured Successfully!"

@app.route('/recognize')
def recognize():
    # Create a directory for storing the embeddings
    create_directory("embeddings")

    # Load the dataset IDs
    dataset_ids = load_dataset_ids()

    # Initialize attendance dictionary
    attendance = {id: {"last_seen": "", "dates": []} for id in dataset_ids}

    # Load the embeddings
    embeddings = load_embeddings()

    # Load the images
    image_paths = [os.path.join("dataset", f) for f in os.listdir("dataset")]

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
            print(f"Embedding {i+1}: {face_encoding}")
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
    save_attendance(attendance)

    return render_template('attendance.html', attendance=attendance)

if __name__ == "__main__":
    app.run(debug=True)
