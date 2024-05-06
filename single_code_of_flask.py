from flask import Flask, render_template, request, jsonify
import cv2
import base64
import numpy as np
import os
import json
import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1
import face_recognition
from mtcnn import MTCNN
from datetime import datetime

app = Flask(__name__)


# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Function to capture a single image from IP camera
def capture_image(ip_camera_url):
    capture = cv2.VideoCapture(ip_camera_url)
    if not capture.isOpened():
        print("Error: Unable to connect to the IP camera. Exiting.")
        exit()
    ret, frame = capture.read()
    capture.release()
    return frame


# Function to identify the best image from captured frames
def find_best_image(frames):
    best_image = None
    best_confidence = 0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for frame in frames:
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                confidence = w * h
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_image = frame.copy()

    return best_image


# Function to clear the content of the ids.txt file
def clear_ids_file():
    with open("ids.txt", "w") as ids_file:
        ids_file.write("")  # Write an empty string to clear the file


# Function to capture images from IP camera
def capture_images(output_folder, person_name, num_images, ip_camera_url):
    create_directory(output_folder)
    captured_frames = []

    # Clear the content of the ids.txt file
    clear_ids_file()

    for count in range(num_images):
        print(f"Capturing image {count + 1}...")
        frame = capture_image(ip_camera_url)
        captured_frames.append(frame)

    # Find the best image among the captured frames
    best_image = find_best_image(captured_frames)

    # Save the best image
    if best_image is not None:
        filename = os.path.join(output_folder, f"{person_name}_best.jpg")
        cv2.imwrite(filename, best_image)
        print(f"Best image saved as '{filename}'")

        # Rename the best image
        person_id = request.form['person_id']
        new_filename = f"{person_id}.jpg"
        new_filepath = os.path.join(output_folder, new_filename)
        os.rename(filename, new_filepath)
        print(f"Best image renamed and saved as '{new_filename}'")
        # Append the ID to the ids.txt file
        with open("ids.txt", "a") as ids_file:
            ids_file.write(f"{person_id}\n")


# Function to detect faces using MTCNN
def detect_faces_mtcnn(image):
    boxes, _ = mtcnn.detect(image)
    return boxes


# Function to extract face from image using bounding box coordinates
def extract_face(image, box):
    x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
    face = image[y1:y2, x1:x2]
    return face if face.size != 0 else None


# Function to generate face embeddings using FaceNet
def generate_face_embeddings_facenet(image_list):
    embeddings_list = []
    for filename, image in image_list:
        boxes = detect_faces_mtcnn(image)
        if boxes is None:
            print(f"No faces detected in image: {filename}.")
            continue

        print(f"Detected {len(boxes)} faces in image: {filename}.")

        embeddings = []
        for i, box in enumerate(boxes):
            # Extract face using bounding box coordinates
            face = extract_face(image, box)

            if face is None:
                print(f"Failed to extract face in image: {filename}.")
                continue

            # Resize the face to match the input size of the model (160x160)
            aligned_face_resized = cv2.resize(face, (160, 160))

            # Convert aligned face to RGB and normalize pixel values
            aligned_face_resized = (aligned_face_resized / 255.).astype(np.float32)

            # Convert face array to PyTorch tensor
            face_tensor = torch.tensor(aligned_face_resized.transpose(2, 0, 1), dtype=torch.float32)

            # Generate face embedding
            with torch.no_grad():
                embedding = facenet(face_tensor.unsqueeze(0)).detach().numpy()
            embeddings.append(embedding)

        embeddings_list.append((filename, embeddings))

    return embeddings_list


# Function to compare face embeddings and determine similarity
def compare_face_embeddings(embedding1, embedding2):
    similarity_score = cosine_similarity(embedding1, embedding2)
    return similarity_score


# Function to clear the detected_faces folder
def clear_detected_faces_folder():
    detected_faces_folder = "detected_faces"
    for file_name in os.listdir(detected_faces_folder):
        file_path = os.path.join(detected_faces_folder, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")


# Function to update detected_faces.txt file with the IDs
def update_detected_faces_file(image_ids):
    with open("detected_faces.txt", "w") as file:  # Use 'w' mode to overwrite existing content
        for image_id in image_ids:
            image_id_without_extension = os.path.splitext(image_id)[0]  # Get ID without extension
            file.write(image_id_without_extension + '\n')  # Add a new line after each ID


# Function to save the detected faces to a folder
def save_detected_faces(image, boxes, save_folder, filename):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    filename_without_extension = os.path.splitext(filename)[0]  # Get filename without extension

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        face = image[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(save_folder, f"{filename_without_extension}.jpg"), face)


# Function to compare faces between dataset and collected_faces images with adaptive thresholding
def compare_faces_with_adaptive_threshold(dataset_images, collected_faces_images):
    match_info = {}  # Dictionary to store match information

    # Clear previous detected faces
    clear_detected_faces_folder()

    for dataset_name, dataset_img in dataset_images:
        dataset_embeddings_facenet = generate_face_embeddings_facenet([(dataset_name, dataset_img)])

        for collected_name, collected_img in collected_faces_images:
            collected_embeddings_facenet = generate_face_embeddings_facenet([(collected_name, collected_img)])

            for d_name, d_embeddings in dataset_embeddings_facenet:
                for c_name, c_embeddings in collected_embeddings_facenet:
                    match_scores = compare_face_embeddings(d_embeddings, c_embeddings)

                    # Compare similarity and perform adaptive thresholding
                    if match_scores > 0.7:
                        if d_name in match_info:
                            match_info[d_name].append((c_name, match_scores))
                        else:
                            match_info[d_name] = [(c_name, match_scores)]

    return match_info


# Loading the dataset
def load_dataset(dataset_folder):
    dataset_images = []
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            filepath = os.path.join(dataset_folder, filename)
            image = cv2.imread(filepath)
            dataset_images.append((filename, image))
    return dataset_images


# Function to find the most recent image file in a folder
def most_recent_file(folder):
    list_of_files = os.listdir(folder)
    full_path = [os.path.join(folder, file) for file in list_of_files]
    latest_file = max(full_path, key=os.path.getctime)
    return latest_file


# Path to the folder containing dataset images
dataset_folder = "dataset"

# Initialize MTCNN and FaceNet
mtcnn = MTCNN(keep_all=True)
facenet = InceptionResnetV1(pretrained='vggface2').eval()


# Route to detect faces in a video stream
@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    # Clear previous detected faces
    clear_detected_faces_folder()

    # Load the video file
    video_file = request.files['file']
    video_file.save("video.mp4")
    video_capture = cv2.VideoCapture("video.mp4")

    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces in the frame
        boxes, _ = mtcnn.detect(frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save the frame with detected faces
        cv2.imwrite(f"detected_faces/detected_faces_{frame_count}.jpg", frame)
        frame_count += 1

    video_capture.release()
    return "Faces detected and saved successfully!"


# Route to upload a video file
@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['file']
    file.save("video.mp4")
    return "Video uploaded successfully!"


# Route to identify a person from a video
@app.route('/identify_person', methods=['POST'])
def identify_person():
    # Load the video file
    video_file = request.files['file']
    video_file.save("video.mp4")
    video_capture = cv2.VideoCapture("video.mp4")

    # Capture frames from the video
    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)

    # Find the best image from the captured frames
    best_image = find_best_image(frames)

    if best_image is not None:
        cv2.imwrite("detected_faces/best_frame.jpg", best_image)
        print("Best image saved successfully!")
    else:
        print("Failed to save best image!")

    return "Person identified and image saved successfully!"


# Route to upload a dataset
@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    file = request.files['file']
    dataset_folder = "dataset"
    file.save(os.path.join(dataset_folder, file.filename))
    return "Dataset uploaded successfully!"


# Route to upload images to compare
@app.route('/compare_images', methods=['POST'])
def compare_images():
    # Load the dataset
    dataset_images = load_dataset(dataset_folder)

    # Load the collected faces
    collected_faces_images = [("collected_face.jpg", cv2.imread("detected_faces/collected_face.jpg"))]

    # Compare the faces
    match_info = compare_faces_with_adaptive_threshold(dataset_images, collected_faces_images)

    # Update detected_faces.txt with the IDs
    if match_info:
        matched_ids = match_info.keys()
        update_detected_faces_file(matched_ids)

    return jsonify(match_info)


# Route to capture and save images from IP camera
@app.route('/capture_images', methods=['POST'])
def capture_images_from_ip_camera():
    ip_camera_url = request.form['ip_camera_url']
    output_folder = "dataset"
    person_name = request.form['person_name']
    person_id = request.form['person_id']
    num_images = int(request.form['num_images'])
    capture_images(output_folder, person_name, num_images, ip_camera_url)
    return "Images captured successfully!"


if __name__ == '__main__':
    app.run(debug=True)
