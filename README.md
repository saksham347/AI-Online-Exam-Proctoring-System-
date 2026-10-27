# AI-Online-Exam-Proctoring-System-
AI-powered online exam proctoring system using Python, Flask, OpenCV, MediaPipe, and YOLO for real-time face detection, multiple-person detection, and mobile phone detection to help monitor exam activities.
# AI Online Exam Proctoring System

An AI-powered web application designed to monitor online examinations using computer vision and real-time webcam analysis. The system helps identify suspicious activities during exams and provides a more reliable online examination monitoring experience.

## Features

- **Face Detection:** Detects faces through the webcam during an examination.
- **Multiple-Person Detection:** Identifies when more than one person appears in the camera frame.
- **Mobile Phone Detection:** Uses YOLO object detection to identify mobile phones.
- **Real-Time Monitoring:** Processes webcam frames using a Flask backend.
- **Suspicious Activity Alerts:** Displays warnings for detected exam violations.
- **Violation Logging:** Records suspicious activity events with timestamps.
- **Exam Interface:** Provides an interactive examination interface with questions and a timer.
- **Video Recording:** Supports webcam recording with microphone audio and an on-screen monitoring overlay.

## Technologies Used

- Python
- Flask
- OpenCV
- MediaPipe
- YOLO (Ultralytics)
- HTML
- CSS
- JavaScript

## Project Purpose

The goal of this project is to explore how artificial intelligence and computer vision can support online examination monitoring by detecting faces, multiple people, mobile phones, and other potentially suspicious activities.

## Getting Started

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the required dependencies from `requirements.txt`.
4. Run the Flask application using `python app.py`.
5. Open `http://127.0.0.1:5000` in your browser and allow webcam and microphone access when prompted.

## Future Improvements

- Student identity verification using facial recognition.
- Improved mobile phone detection accuracy.
- Enhanced examination reports and violation analytics.
- More robust identity matching and anti-spoofing checks.

## Disclaimer

This project is intended for educational and research purposes. Detection results may not always be accurate, and automated alerts should not be treated as definitive proof of cheating.
