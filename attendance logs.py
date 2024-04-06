import cv2
import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MTCNN for face alignment
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# Initialize FaceNet for face recognition
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Function to detect faces using cascade classifier
def detect_faces_cascade(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Function to generate face embeddings using FaceNet
def generate_face_embeddings(image_list, image_type):
    embeddings_list = []
    for filename, image in image_list:
        faces = detect_faces_cascade(image)
        if len(faces) == 0:
            print(f"No faces detected in {image_type} image: {filename}.")
            continue

        print(f"Detected {len(faces)} faces in {image_type} image: {filename}.")

        embeddings = []
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face using bounding box coordinates
            face = image[y:y+h, x:x+w]

            # Resize the face to match the input size of the model (160x160)
            aligned_face_resized = cv2.resize(face, (160, 160))

            # Convert aligned face to RGB and normalize pixel values
            aligned_face_resized = (aligned_face_resized / 255.).astype(np.float32)

            # Convert face array to PyTorch tensor
            face_tensor = torch.tensor(aligned_face_resized.transpose(2, 0, 1), dtype=torch.float32)

            # Generate face embedding
            with torch.no_grad():
                embedding = facenet(face_tensor.unsqueeze(0)).detach().numpy()
            print(f"Generated embedding for face {i + 1} in {image_type} image: {filename}.")
            embeddings.append((embedding, (x, y, w, h)))  # Store the embedding along with its bounding box

        embeddings_list.append((filename, embeddings))

    return embeddings_list

# Function to compare face embeddings and determine similarity
def compare_face_embeddings(embedding1, embedding2):
    similarity_score = cosine_similarity(embedding1, embedding2)
    return similarity_score

# Function to compare faces between dataset and collected_faces images
def compare_faces(dataset_images, collected_faces_images):
    matched_pairs = []
    for dataset_name, dataset_img in dataset_images:
        dataset_embeddings = generate_face_embeddings([(dataset_name, dataset_img)], "dataset")
        if not dataset_embeddings:
            print(f"No embeddings generated for {dataset_name}.")
            continue
        for collected_name, collected_img in collected_faces_images:
            collected_embeddings = generate_face_embeddings([(collected_name, collected_img)], "collected_faces")
            if not collected_embeddings:
                print(f"No embeddings generated for {collected_name}.")
                continue
            for dataset_embedding, dataset_bbox in dataset_embeddings[0][1]:
                for collected_embedding, collected_bbox in collected_embeddings[0][1]:
                    similarity_score = compare_face_embeddings(dataset_embedding, collected_embedding)
                    threshold = 0.5  # Adjust threshold as needed
                    if similarity_score > threshold:
                        matched_pairs.append((dataset_name, collected_name, similarity_score, threshold, dataset_bbox, collected_bbox))
                        break
    return matched_pairs

# Load dataset and collected_faces images
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

# Compare faces between dataset and collected_faces images
matched_pairs = compare_faces(dataset_images, collected_faces_images)

# Print matched pairs
print("Matched pairs:")
for dataset_name, collected_name, similarity_score, threshold, dataset_bbox, collected_bbox in matched_pairs:
    print(f"Dataset image '{dataset_name}' matched with collected_faces image '{collected_name}' with similarity score: {similarity_score}, threshold: {threshold}")
    print("Bounding Box (Dataset):", dataset_bbox)
    print("Bounding Box (Collected):", collected_bbox)
