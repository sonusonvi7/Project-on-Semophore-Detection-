# Semophore_Detection

1. 🪖 Real-Time Semaphore Decoder with Pose Detection

This project uses **Jetson Nano**, **MediaPipe**, and **OpenCV** to recognize **naval semaphore signals** using human body poses in real-time. It decodes the hand-held flag positions (based on shoulder–wrist angles) into alphabetic letters.

---

## 🚀 Live Demo

> Watch the decoded message appear as you hold semaphore poses in front of the camera. Letters are appended automatically when a pose is held steadily for 1.5 seconds.

---

## 🛠️ Tech Stack

- **Jetson Nano Developer Kit (4GB)**
- Python 3.x
- [MediaPipe](https://github.com/google/mediapipe) for pose detection
- [OpenCV](https://opencv.org/) for camera and visual rendering
- Flask (for optional web-based control via `app.py`)

---

## 📂 Folder Structure

---

## ⚙️ How It Works

1. **Pose Estimation**: MediaPipe tracks left/right shoulders and wrists.
2. **Angle Calculation**: Calculates signed angles between shoulders and wrists.
3. **Angle Matching**: Compares angle pair with a pre-defined semaphore dictionary.
4. **Hold Timer**: Adds a letter only if the pose is steady for >1.5 seconds.
5. **Display**: Shows decoded message, current angles, and frame via OpenCV.

---

## 🧠 Algorithm

- Uses **angle matching algorithm**:
  - Computes signed angle from shoulder to wrist.
  - Matches with `(left_angle, right_angle)` in a dictionary.
  - Uses `deque` buffer to ensure stability before accepting a letter.

---

## 🎮 Controls

- Press `q` to quit the live video feed.
- Use `app.py` to start/stop detection via a Flask-based frontend.

---

## 🖥️ Running the Code

### 1. Install Requirements

```bash
1.pip install opencv-python mediapipe flask numpy

2.Start Detection
python Navy.py

3.cap = cv2.VideoCapture(1)  # Change index from 0 to 1 if you are using a seconday web cam

4.Start Web Interface
python app.py

✨ Future Add-ons
Browser-based visualization (via Flask streaming)

Accuracy improvement with dynamic calibration

Support for full word and sentence formation

Save decoded messages to a log file

🔗 Connect
Linkedin : Sonvi Assis Noronha
https://www.linkedin.com/in/sonvi-assis-noronha-96531524b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app
Email:sonusonvi7@gmail.com
```
