import cv2
import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize MTCNN for face detection and alignment
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# Initialize FaceNet for face recognition
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Function to generate face embeddings using FaceNet
def generate_face_embeddings(image_list, image_type):
    embeddings_list = []
    for filename, image in image_list:
        # Detect and align faces using MTCNN
        detections = mtcnn.detect(image, landmarks=True)

        if detections is None or len(detections) == 0:
            print(f"No faces detected in {image_type} image: {filename}.")
            continue

        # Extract bounding boxes from the detections
        boxes = detections[0]
        print(f"Detected {len(boxes)} faces in {image_type} image: {filename}.")

        # Generate face embeddings for each detected face
        embeddings = []
        for i, box in enumerate(boxes):
            # Extract face using bounding box coordinates
            x1, y1, x2, y2 = box.astype(np.int32)
            face = image[y1:y2, x1:x2]

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
            embeddings.append(embedding)

        embeddings_list.append((filename, embeddings))

    return embeddings_list


# Function to compare face embeddings and determine similarity
def compare_face_embeddings(embedding1, embedding2):
    # Use cosine similarity to compare embeddings
    similarity_score = cosine_similarity(embedding1, embedding2)
    return similarity_score

# Function to compare faces between dataset and collected_faces images
def compare_faces(dataset_images, collected_faces_images):
    matched_pairs = []
    # Clear the previous content in detected_faces.txt
    with open("detected_faces.txt", "w") as ids_file:
        # Clear the previous images in the detected_faces folder
        for filename in os.listdir("detected_faces"):
            file_path = os.path.join("detected_faces", filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        for dataset_name, dataset_img in dataset_images:
            dataset_id = dataset_name.split("_")[0]  # Assuming the ID is at the beginning of the filename before underscore
            dataset_id_only = dataset_id.split(".")[0]  # Remove extension
            dataset_embeddings = generate_face_embeddings([(dataset_name, dataset_img)], "dataset")
            if not dataset_embeddings:
                print(f"No embeddings generated for {dataset_name}.")
                continue
            for collected_name, collected_img in collected_faces_images:
                collected_embeddings = generate_face_embeddings([(collected_name, collected_img)], "collected_faces")
                if not collected_embeddings:
                    print(f"No embeddings generated for {collected_name}.")
                    continue
                # Compare each pair of embeddings to determine similarity
                for dataset_embedding in dataset_embeddings[0][1]:
                    for collected_embedding in collected_embeddings[0][1]:
                        similarity_score = compare_face_embeddings(dataset_embedding, collected_embedding)
                        # Set a threshold for similarity score
                        if similarity_score > 0.7:  # Adjust threshold as needed
                            matched_pairs.append((dataset_name, collected_name, similarity_score))
                            # Save the matched image from dataset folder into detected_faces folder
                            cv2.imwrite(os.path.join("detected_faces", dataset_name), dataset_img)
                            ids_file.write(f"{dataset_id_only}\n")  # Write ID to detected_faces.txt without extension
                            break  # Move to the next dataset image
    return matched_pairs

# Load dataset and collected_faces images
dataset_images = []
for filename in os.listdir("dataset"):
    img_path = os.path.join("dataset", filename)
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            dataset_images.append((filename, img))

collected_faces_images = []
for filename in os.listdir("collected_faces"):
    img_path = os.path.join("collected_faces", filename)
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            collected_faces_images.append((filename, img))

# Compare faces between dataset and collected_faces images
matched_pairs = compare_faces(dataset_images, collected_faces_images)

# Print matched pairs
print("Matched pairs:")
for dataset_name, collected_name, similarity_score in matched_pairs:
    print(f"Dataset image '{dataset_name}' matched with collected_faces image '{collected_name}' with similarity score: {similarity_score}")
