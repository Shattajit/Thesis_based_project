import cv2
import os
from mtcnn import MTCNN

# Create an instance of the MTCNN detector with adjusted parameters
detector = MTCNN(min_face_size=20, scale_factor=0.709)

# Function to detect faces using MTCNN and apply additional filters
def detect_faces(image):
    try:
        # Convert the image to RGB (MTCNN expects RGB images)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        faces = detector.detect_faces(rgb_image)

        # Filter out faces based on various criteria
        valid_faces = []
        for face in faces:
            # Check if all facial landmarks are present
            if all(key in face['keypoints'] for key in ('left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right')):
                # Filter out faces based on confidence score
                if face['confidence'] > 0.5:  # Adjust this threshold as needed
                    # Filter out faces based on size and aspect ratio
                    x, y, w, h = face['box']
                    aspect_ratio = w / h
                    # Adjust these thresholds as needed
                    if 0.5 < aspect_ratio < 2.0 and w > 20 and h > 20:
                        valid_faces.append((x, y, w, h))

        return valid_faces

    except Exception as e:
        print("Error:", e)
        return []

# Function to save detected faces with a portion of the neck as JPG files
def save_faces(image, faces, output_folder):
    # Delete all files in the output folder
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Save new detected faces as JPG files
    for i, (x, y, w, h) in enumerate(faces):
        # Extend the bounding box vertically to include a portion of the neck
        neck_height = int(h * 0.5)  # Adjust this value as needed
        extended_y = max(0, y - neck_height)
        extended_h = min(image.shape[0] - extended_y, h + neck_height)

        # Crop the detected face region with neck
        face_with_neck = image[extended_y:extended_y + extended_h, x:x + w]

        # Generate a unique filename for the face
        filename = os.path.join(output_folder, f"face_{i + 1}.jpg")

        # Save the face with neck as a JPG file
        cv2.imwrite(filename, face_with_neck)

# Load the image
img = cv2.imread("best_captured_image_enhanced.jpg")

# Create the output folder if it doesn't exist
output_folder = "collected_faces"
os.makedirs(output_folder, exist_ok=True)

# Detect faces in the image using MTCNN and apply additional filters
faces = detect_faces(img)

if faces:
    # Save detected faces with a portion of the neck as JPG files
    save_faces(img, faces, output_folder)

    print(f"{len(faces)} faces detected and saved to '{output_folder}' folder.")
else:
    print("No faces detected.")
