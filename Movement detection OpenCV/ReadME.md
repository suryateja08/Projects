# OpenCV

### 1. Movement Detection
**File:** `detect_motion.py`  
**Video:** `vtest.avi`

This project detects and highlights movement in a video stream using frame differencing and contour detection.

#### Key Features:
- **Frame Differencing:** Compares consecutive video frames to identify areas of change.
- **Thresholding:** Converts the differences into a binary image to isolate moving areas.
- **Morphological Transformations:** Utilizes gradient operations to refine the detected motion.
- **Contour Detection:** Identifies and bounds regions of movement with rectangular boxes.
- **Real-Time Feedback:** Displays the status of motion detection in the video feed.

#### Technologies Used:
- **OpenCV**: For image processing and video handling.
- **NumPy**: For array operations and morphological kernel creation.

### 2. Face and Feature Detection
**File:** `face_detection.py`

This project employs Haar cascades to detect faces, eyes, and smiles in a real-time video feed from a webcam.

#### Key Features:
- **Face Detection:** Identifies human faces using Haar cascade classifiers.
- **Eye Detection:** Locates eyes within the detected face regions.
- **Smile Detection:** Detects smiles as additional features within face regions.
- **Real-Time Analysis:** Continuously processes video frames from the webcam for instant feedback.

#### Technologies Used:
- **OpenCV**: For implementing Haar cascades and real-time video capture.
- **Pre-Trained Models**: Haar cascade XML files for face, eye, and smile detection.
- **NumPy**: For efficient numerical operations.

## Installation
To run these projects, ensure the following dependencies are installed:

1. Python 3.x
2. OpenCV library
3. NumPy library

Install dependencies using pip:
```bash
pip install opencv-python numpy
```

## Usage

### 1. Movement Detection
- Place `vtest.avi` in the same directory as `detect_motion.py`.
- Run the script:
  ```bash
  python detect_motion.py
  ```

### 2. Face and Feature Detection
- Run the `face_detection.py` script:
  ```bash
  python face_detection.py
  ```
- Ensure your webcam is connected and accessible.

## Acknowledgments
- OpenCV Documentation and Tutorials: https://docs.opencv.org
- Haar Cascade XML files provided by OpenCV.
