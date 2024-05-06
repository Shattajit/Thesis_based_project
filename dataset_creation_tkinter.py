import cv2
import os
import tkinter as tk
from PIL import Image, ImageTk
import urllib.request
import numpy as np

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to capture a single image from IP camera
def capture_image(ip_camera_url):
    img_resp = urllib.request.urlopen(ip_camera_url)
    img_array = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(img_array, -1)
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

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Captured Images")

    for count in range(num_images):
        print(f"Capturing image {count + 1}...")
        frame = capture_image(ip_camera_url)
        captured_frames.append(frame)

        # Display the captured frame using Tkinter
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        panel = tk.Label(root, image=image)
        panel.image = image
        panel.pack()

        # Wait for 1 second before capturing the next frame
        root.after(1000, panel.pack_forget)

    # Find the best image among the captured frames
    best_image = find_best_image(captured_frames)

    # Save the best image
    if best_image is not None:
        filename = os.path.join(output_folder, f"{person_name}_best.jpg")
        cv2.imwrite(filename, best_image)
        print(f"Best image saved as '{filename}'")

        # Rename the best image
        person_id = input("Enter the ID for the person: ")
        new_filename = f"{person_id}.jpg"
        new_filepath = os.path.join(output_folder, new_filename)
        os.rename(filename, new_filepath)
        print(f"Best image renamed and saved as '{new_filename}'")
        # Append the ID to the ids.txt file
        with open("ids.txt", "a") as ids_file:
            ids_file.write(f"{person_id}\n")

    root.mainloop()

# Example usage:
output_folder = "dataset"
person_name = "opp"  # Change this to the name of the person being captured
num_images = 5  # Change this to the number of images to capture for each person
ip_camera_url = input("Enter the IP camera URL: ")  # Replace with your IP camera address
capture_images(output_folder, person_name, num_images, ip_camera_url)
