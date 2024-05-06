from flask import Flask, render_template, request, jsonify
import cv2
import os
import numpy as np
import base64
from mtcnn import MTCNN
import json
import torch
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN as FaceNetMTCNN
from facenet_pytorch import InceptionResnetV1

app = Flask(__name__)

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to capture a single image from IP camera
def capture_image(ip_camera_url):
    capture = cv2.VideoCapture(ip_camera_url)
    if not capture.isOpened():
        return {"error": "Unable to connect to the IP camera."}, 400
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
    detector = MTCNN(min_face_size=20, scale_factor=0.709)
    boxes = detector.detect_faces(image)
    return boxes

# Function to extract face from image using bounding box coordinates
def extract_face(image, box):
    x1, y1, width, height = box['box']
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]
    return face

# Function to generate face embeddings using FaceNet
def generate_face_embeddings_facenet(image_list):
    face_embeddings = []
    facenet = InceptionResnetV1(pretrained='vggface2').eval()

    for filename, image in image_list:
        boxes = detect_faces_mtcnn(image)
        if boxes:
            for box in boxes:
                face = extract_face(image, box)
                aligned_face_resized = cv2.resize(face, (160, 160))
                aligned_face_resized = (aligned_face_resized / 255.).astype(np.float32)
                face_tensor = torch.tensor(aligned_face_resized.transpose(2, 0, 1), dtype=torch.float32)
                with torch.no_grad():
                    embedding = facenet(face_tensor.unsqueeze(0)).detach().numpy()
                face_embeddings.append((filename, embedding))
        else:
            print(f"No faces detected in {filename}")

    return face_embeddings

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
        x1, y1, width, height = box['box']
        x2, y2 = x1 + width, y1 + height
        face = image[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(save_folder, f"{filename_without_extension}_{i}.jpg"), face)

# Function to compare faces between dataset and collected_faces images with adaptive thresholding
def compare_faces_with_adaptive_threshold(dataset_images, collected_faces_images):
    match_info = {}  # Dictionary to store match information

    # Clear previous detected faces
    clear_detected_faces_folder()

    for dataset_name, dataset_img in dataset_images:
        dataset_embeddings_facenet = generate_face_embeddings_facenet([(dataset_name, dataset_img)])

        if not dataset_embeddings_facenet:
            print(f"No embeddings generated for {dataset_name}.")
            continue

        for collected_name, collected_img in collected_faces_images:
            collected_embeddings_facenet = generate_face_embeddings_facenet([(collected_name, collected_img)])

            if not collected_embeddings_facenet:
                print(f"No embeddings generated for {collected_name}.")
                continue

            similarity_scores = []
            for emb1 in dataset_embeddings_facenet:
                for emb2 in collected_embeddings_facenet:
                    similarity_score = compare_face_embeddings(emb1[1], emb2[1])
                    similarity_scores.append(similarity_score)

            # Set a dynamic threshold based on the similarity scores
            threshold = np.mean(similarity_scores) - 2 * np.std(similarity_scores)
            if threshold < 0.1:
                threshold = 0.1

            matches = []
            for emb1 in dataset_embeddings_facenet:
                for emb2 in collected_embeddings_facenet:
                    similarity_score = compare_face_embeddings(emb1[1], emb2[1])
                    if similarity_score > threshold:
                        matches.append((emb1[0], emb2[0], similarity_score[0][0]))

            match_info[collected_name] = matches
            print(f"Similarity scores for {collected_name}: {matches}")

            # Save the detected faces
            if matches:
                boxes = detect_faces_mtcnn(collected_img)
                if boxes:
                    save_detected_faces(collected_img, boxes, "detected_faces", collected_name)

    # Update detected_faces.txt with the IDs of the collected faces
    update_detected_faces_file(match_info.keys())

    return match_info

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/capture", methods=["POST"])
def capture():
    output_folder = "dataset_images"
    person_name = request.form['person_name']
    person_id = request.form['person_id']
    num_images = int(request.form['num_images'])
    ip_camera_url = request.form['ip_camera_url']

    capture_images(output_folder, person_name, num_images, ip_camera_url)

    return jsonify({"message": f"{num_images} images captured for {person_name}."})

@app.route("/identify")
def identify():
    dataset_folder = "dataset_images"
    collected_faces_folder = "collected_faces"
    dataset_images = []
    collected_faces_images = []

    # Load images from the dataset folder
    for file_name in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file_name)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            dataset_images.append((file_name, image))

    # Load images from the collected_faces folder
    for file_name in os.listdir(collected_faces_folder):
        file_path = os.path.join(collected_faces_folder, file_name)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            collected_faces_images.append((file_name, image))

    match_info = compare_faces_with_adaptive_threshold(dataset_images, collected_faces_images)

    return jsonify(match_info)

if __name__ == "__main__":
    app.run(debug=True)
