import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

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

def show_best_image():
    quality_scores = [evaluate_quality(image) for image in captured_images]
    best_index = quality_scores.index(max(quality_scores))
    best_image = captured_images[best_index]
    cv2.imwrite("best_captured_image_enhanced.jpg", best_image)
    print("Best image saved as 'best_captured_image_enhanced.jpg'")
    best_image = cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(best_image)
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo

def capture_and_display():
    image = capture_image(ip_camera_url)
    captured_images.append(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo

if __name__ == "__main__":
    ip_camera_url = input("Enter IP camera URL: ")
    num_images = 5
    captured_images = []

    root = tk.Tk()
    root.title("IP Camera Image Capture")

    canvas = tk.Canvas(root, width=640, height=480)
    canvas.pack()

    capture_button = tk.Button(root, text="Capture Image", command=capture_and_display)
    capture_button.pack(pady=10)

    best_button = tk.Button(root, text="Show Best Image", command=show_best_image)
    best_button.pack(pady=10)

    root.mainloop()
