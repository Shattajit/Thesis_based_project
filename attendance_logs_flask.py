from flask import Flask, render_template, request
import cv2
import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1

app = Flask(__name__)

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# Initialize FaceNet for face recognition
facenet = InceptionResnetV1(pretrained='vggface2').eval()

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

        if not dataset_embeddings_facenet:
            print(f"No embeddings generated for {dataset_name}.")
            continue

        for collected_name, collected_img in collected_faces_images:
            collected_embeddings_facenet = generate_face_embeddings_facenet([(collected_name, collected_img)])

            if not collected_embeddings_facenet:
                print(f"No embeddings generated for {collected_name}.")
                continue

            for dataset_embedding_facenet in dataset_embeddings_facenet[0][1]:
                for collected_embedding_facenet in collected_embeddings_facenet[0][1]:
                    threshold_facenet = compare_face_embeddings(dataset_embedding_facenet, collected_embedding_facenet)

                    # Use adaptive thresholding
                    if threshold_facenet >= get_adaptive_threshold(dataset_img):
                        if dataset_name not in match_info:
                            match_info[dataset_name] = []
                        match_info[dataset_name].append(collected_name)

                        # Save the detected face
                        save_detected_faces(dataset_img, detect_faces_mtcnn(dataset_img), "detected_faces", dataset_name)

    # Update detected_faces.txt file with the IDs
    update_detected_faces_file(match_info.keys())

    return match_info

# Function to calculate adaptive threshold for an image
def get_adaptive_threshold(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate adaptive threshold using Gaussian adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Calculate the percentage of white pixels in the thresholded image
    white_pixels_ratio = np.sum(adaptive_thresh == 255) / (adaptive_thresh.shape[0] * adaptive_thresh.shape[1])

    # Return the adaptive threshold value based on the percentage of white pixels
    # Adjust this threshold value according to your requirements
    if white_pixels_ratio > 0.1:  # Example threshold, you can adjust this
        return 0.6
    else:
        return 0.7

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    # Load dataset and collected_faces images (assuming the dataset and collected_faces directories exist)
    dataset_images = []
    for filename in os.listdir("dataset"):
        img_path = os.path.join("dataset", filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                dataset_images.append((filename, img))
            else:
                print(f"Failed to load image: {img_path}")

    collected_faces_images = []
    for filename in os.listdir("collected_faces"):
        img_path = os.path.join("collected_faces", filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                collected_faces_images.append((filename, img))
            else:
                print(f"Failed to load image: {img_path}")

    # Compare faces between dataset and collected_faces images with adaptive threshold
    match_info_adaptive_threshold = compare_faces_with_adaptive_threshold(dataset_images, collected_faces_images)

    # Print match information
    for dataset_image, matched_collected_images in match_info_adaptive_threshold.items():
        print(f"Dataset Image: {dataset_image} matched with Collected Faces: {matched_collected_images}")

    return "Face comparison completed successfully!"

if __name__ == "__main__":
    app.run(debug=True)
