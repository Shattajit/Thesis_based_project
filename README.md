# Smart Attendance System Using Machine Learning And Deep Learning

## Overview

This project implements a face recognition system that detects faces in images, compares them with a predefined dataset, and logs attendance. It uses MTCNN for face detection, FaceNet (InceptionResnetV1) for facial feature extraction, and cosine similarity for comparing embeddings. The system logs the attendance in an `attendance.json` file and generates a report in `attendance_log.xlsx`. The detected faces are also saved, and their details are logged in `detected_faces.txt`.

## Features

- **Face Detection**: Utilizes MTCNN to detect multiple faces in images.
  
- **Face Recognition**: Uses FaceNet to extract facial embeddings and compares them using cosine similarity.
  
- **Attendance Logging**: Tracks and records attendance information, marking individuals as 'Present' or 'Absent'.
  
- **Dynamic Thresholding**: Employs adaptive thresholding to improve face matching accuracy based on image quality.
  
- **Report Generation**: Exports attendance data to an Excel file (`attendance_log.xlsx`).
  
- **Detected Faces Saving**: Detected faces are saved in the `detected_faces` folder for further verification.

  

## Technologies Used

### Frontend:

- **React**: Interactive user interface for image uploading and results display.

### Backend:
  
- **Python**: Core logic for face detection, recognition, and attendance tracking.
  
- **OpenCV**: Image preprocessing and handling.
  
- **Facenet-Pytorch**: Pretrained models for generating face embeddings.
  
- **MTCNN**: For accurate face detection.
  
- **Pandas**: For data handling and generating attendance logs.
  
- **XlsxWriter**: For exporting data into Excel files.

- **Flask/Django**: For backend API services, handling image uploads, and database management.

## Getting Started

### Prerequisites

To run the project locally, ensure you have the following installed:

- Python 3.x and package manager **pip**
  
- **Node.js** and **npm** (for React frontend)
  
- **OpenCV** (`pip install opencv-python`)

- **Facenet-Pytorch** (`pip install facenet-pytorch`)

- **MTCNN** (`pip install mtcnn`)

- **Pandas**, **XlsxWriter**, and other dependencies

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <project-directory>

2. **Install python dependencies**

3. **Run the application**

   The system will detect faces from the collected images, compare them with the dataset, log the attendance, and generate the report.

   ```bash
   python app.py
   

## Output

1. **Detected Faces**: Faces detected from the input images are stored in the `detected_faces` folder.
   
2. **Attendance Log**: Attendance information is stored in `attendance.json` and exported to an Excel file, `attendance_log.xlsx`.

3. **Detected Faces Log**: A list of detected face IDs is saved in `detected_faces.txt`.
  
4. **Detected Faces**: A list of all dataset image IDs is saved in `ids.txt`.



## Functionality Breakdown

### Face Detection

- **MTCNN** detects multiple faces in images from the `collected_faces` folder.

### Face Embedding
  
- **FaceNet (InceptionResnetV1)** generates facial embeddings for detected faces.
  
- Embeddings are compared to those of known individuals in the `dataset` folder using cosine similarity.


### Attendance Tracking
  
- Attendance is recorded based on face recognition results.
  
- The system updates attendance status as 'Present' or 'Absent' and logs this information.


### Adaptive Thresholding

- The system uses adaptive thresholding to dynamically adjust face matching accuracy based on image quality.

### Report Generation
- Attendance data is exported to `attendance_log.xlsx` with columns for ID, last seen timestamp, and attendance status.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

