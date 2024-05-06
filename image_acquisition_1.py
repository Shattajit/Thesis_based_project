import cv2
import os
import numpy as np
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# Create an instance of the MTCNN detector with adjusted parameters
detector = MTCNN(min_face_size=20, scale_factor=0.709)

# Initialize FaceNet for feature extraction
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Define the preprocessing transformations
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

def capture_image(ip_camera_url):
    cap = cv2.VideoCapture(ip_camera_url)
    if not cap.isOpened():
        print("Error: Unable to connect to the IP camera. Exiting.")
        exit()
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture image.")
        exit()
    cap.release()
    return frame

def evaluate_quality(image):
    # Calculate clarity score as a measure of image quality
    clarity_score = calculate_clarity(image)
    return clarity_score

def calculate_clarity(image):
    # Calculate clarity score based on Laplacian variance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clarity_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return clarity_score

# Function to detect faces, neck, ears, and chest using MTCNN and apply additional filters
def detect_faces(image):
    try:
        # Convert the image to RGB (MTCNN expects RGB images)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        detections = detector.detect_faces(rgb_image)

        # Filter out faces based on various criteria
        valid_faces = []
        for detection in detections:
            face_box = detection['box']
            keypoints = detection['keypoints']

            # Check if all facial landmarks are present
            if all(key in keypoints for key in ('left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right')):
                # Filter out faces based on confidence score
                if detection['confidence'] > 0.5:  # Adjust this threshold as needed
                    # Extract coordinates of facial keypoints
                    left_eye = keypoints['left_eye']
                    right_eye = keypoints['right_eye']
                    nose = keypoints['nose']
                    mouth_left = keypoints['mouth_left']
                    mouth_right = keypoints['mouth_right']

                    # Calculate additional keypoints for neck, ears, and chest
                    neck = ((nose[0] + left_eye[0] + right_eye[0]) // 3, (nose[1] + left_eye[1] + right_eye[1]) // 3)
                    left_ear = ((left_eye[0] + mouth_left[0]) // 2, (left_eye[1] + mouth_left[1]) // 2)
                    right_ear = ((right_eye[0] + mouth_right[0]) // 2, (right_eye[1] + mouth_right[1]) // 2)
                    chest = (nose[0], mouth_right[1])  # Adjust as needed

                    # Filter out faces based on size and aspect ratio
                    x, y, w, h = face_box
                    aspect_ratio = w / h
                    # Adjust these thresholds as needed
                    if 0.5 < aspect_ratio < 2.0 and w > 20 and h > 20:
                        valid_faces.append({'box': face_box, 'keypoints': keypoints, 'neck': neck, 'left_ear': left_ear,
                                            'right_ear': right_ear, 'chest': chest})

        return valid_faces

    except Exception as e:
        print("Error:", e)
        return []

# Function to save detected faces with a portion of the neck and chest as JPG files
def save_faces(image, faces, output_folder):
    # Delete all files in the output folder
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Save new detected faces as JPG files
    for i, face in enumerate(faces):
        # Extract bounding box coordinates
        x, y, w, h = face['box']

        # Extend the bounding box vertically to include a portion of the neck and chest
        neck_height = int(h * 0.5)  # Adjust this value as needed
        chest_height = int(h * 1.5)  # Adjust this value as needed
        extended_y = max(0, y - neck_height)
        extended_h = min(image.shape[0] - extended_y, h + chest_height)

        # Extend the bounding box horizontally to capture more chest region
        extended_w = int(w * 1.2)  # Adjust this value as needed
        extended_x = max(0, x - (extended_w - w) // 2)
        extended_w = min(image.shape[1] - extended_x, extended_w)

        # Crop the detected face region with neck and chest
        face_with_neck_and_chest = image[extended_y:extended_y + extended_h, extended_x:extended_x + extended_w]

        # Generate a unique filename for the face
        filename = os.path.join(output_folder, f"face_{i + 1}.jpg")

        # Save the face with neck and chest as a JPG file
        cv2.imwrite(filename, face_with_neck_and_chest)


if __name__ == "__main__":
    ip_camera_url = input("Enter IP camera URL: ")
    num_images = 5
    captured_images = []

    for _ in range(num_images):
        image = capture_image(ip_camera_url)
        captured_images.append(image)
        cv2.imshow("Captured Image", image)
        cv2.waitKey(1000)

    quality_scores = [evaluate_quality(image) for image in captured_images]
    best_index = quality_scores.index(max(quality_scores))
    best_image = captured_images[best_index]

    cv2.imwrite("best_captured_image_enhanced.jpg", best_image)
    print("Best image saved as 'best_captured_image_enhanced.jpg'")

    cv2.imshow("Best Image", best_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Load the image
    img = best_image

    # Create the output folder if it doesn't exist
    output_folder = "collected_faces"
    os.makedirs(output_folder, exist_ok=True)

    # Detect faces, neck, ears, and chest in the image using MTCNN and apply additional filters
    faces = detect_faces(img)

    if faces:
        # Save detected faces with a portion of the neck and chest as JPG files
        save_faces(img, faces, output_folder)

        print(f"{len(faces)} faces detected and saved to '{output_folder}' folder.")
    else:
        print("No faces detected.")
