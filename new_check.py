import os
import cv2
import numpy as np
from retinaface import RetinaFace
import face_recognition


# Function to detect faces using RetinaFace
def detect_faces_retinaface(image_path):
    # Load RetinaFace model locally
    retinaface = RetinaFace.build_model()
    image = cv2.imread(image_path)

    # Convert the image to the expected format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)

    faces = retinaface(image)
    return faces


# Function to encode face using ArcFace
# Function to encode face using ArcFace
def encode_face_arcface(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        encoding = face_encodings[0]
        return encoding
    else:
        return None


# Rest of the code remains the same...


# Function to compare face embeddings
def compare_faces(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return False
    distance = np.linalg.norm(embedding1 - embedding2)
    threshold = 10.0  # You may adjust the threshold based on your requirement
    return distance < threshold

# Path to the dataset and collected_faces folders
dataset_folder = "dataset"
collected_faces_folder = "collected_faces"

# Iterate through images in the dataset folder
for dataset_image_name in os.listdir(dataset_folder):
    dataset_image_path = os.path.join(dataset_folder, dataset_image_name)
    dataset_faces = detect_faces_retinaface(dataset_image_path)
    dataset_encodings = [encode_face_arcface(dataset_image_path) for _ in dataset_faces]

    # Iterate through images in the collected_faces folder
    for collected_image_name in os.listdir(collected_faces_folder):
        collected_image_path = os.path.join(collected_faces_folder, collected_image_name)
        collected_faces = detect_faces_retinaface(collected_image_path)
        collected_encodings = [encode_face_arcface(collected_image_path) for _ in collected_faces]

        # Compare face embeddings
        for dataset_encoding in dataset_encodings:
            for collected_encoding in collected_encodings:
                if compare_faces(dataset_encoding, collected_encoding):
                    print(f"Dataset image {dataset_image_name} matches with collected_faces image {collected_image_name}")
                    break
            else:
                continue
            break
