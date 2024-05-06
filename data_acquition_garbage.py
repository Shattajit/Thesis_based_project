from flask import Flask, request, jsonify
import cv2
import base64
import numpy as np

app = Flask(__name__)

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

@app.route('/')
def index():
    return open('index.html').read()

@app.route('/capture_image', methods=['POST'])
def capture_image_route():
    ip_camera_url = request.json['ip_camera_url']
    num_images = 5
    captured_images = []

    for _ in range(num_images):
        image = capture_image(ip_camera_url)
        captured_images.append(image)
        cv2.imshow("Captured Image", image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()  # Close OpenCV window after capturing image

    quality_scores = [evaluate_quality(image) for image in captured_images]
    best_index = quality_scores.index(max(quality_scores))
    best_image = captured_images[best_index]

    cv2.imwrite("best_captured_image_enhanced.jpg", best_image)

    _, buffer = cv2.imencode('.jpg', best_image)
    best_image_encoded = base64.b64encode(buffer).decode()

    return best_image_encoded

if __name__ == "__main__":
    app.run(debug=True)
