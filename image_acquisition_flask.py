from flask import Flask, request, render_template, jsonify
import cv2
import os
from mtcnn import MTCNN

app = Flask(__name__)

# Create an instance of the MTCNN detector with adjusted parameters
detector = MTCNN(min_face_size=20, scale_factor=0.709)

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


@app.route('/detect_faces', methods=['POST'])
def detect_faces_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = "uploaded_image.jpg"
        file.save(filename)

        # Load the image
        img = cv2.imread(filename)

        # Create the output folder if it doesn't exist
        output_folder = "collected_faces"
        os.makedirs(output_folder, exist_ok=True)

        # Detect faces, neck, ears, and chest in the image using MTCNN and apply additional filters
        faces = detect_faces(img)

        if faces:
            # Save detected faces with a portion of the neck and chest as JPG files
            save_faces(img, faces, output_folder)

            return jsonify({"message": f"{len(faces)} faces detected and saved to '{output_folder}' folder."}), 200
        else:
            return jsonify({"message": "No faces detected."}), 200

    return jsonify({"error": "Something went wrong"}), 400

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
