from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# ==============================
# MediaPipe Face Detection
# ==============================
mp_face_detection = mp.solutions.face_detection

face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# ==============================
# YOLO Phone Detection
# ==============================
phone_model = YOLO("yolo11n.pt")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/exam")
def exam():
    return render_template("exam.html")


@app.route("/detect", methods=["POST"])
def detect():

    try:
        file = request.files["frame"]

        # Browser image -> OpenCV
        image_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        # ==================================
        # FACE DETECTION
        # ==================================

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_detector.process(rgb_frame)

        face_count = 0

        if face_results.detections:
            face_count = len(face_results.detections)

        # ==================================
        # PHONE DETECTION
        # ==================================

        phone_detected = False

        yolo_results = phone_model(
            frame,
            verbose=False,
            conf=0.20
        )

        for result in yolo_results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])

                class_name = phone_model.names[class_id]

                if class_name == "cell phone":
                    phone_detected = True
                    break

            if phone_detected:
                break

        # ==================================
        # DETERMINE STATUS
        # ==================================

        if phone_detected:

            status = "phone_detected"

        elif face_count == 0:

            status = "no_face"

        elif face_count > 1:

            status = "multiple_faces"

        else:

            status = "normal"

        # ==================================
        # SEND RESULT TO BROWSER
        # ==================================

        return jsonify({
            "face_count": face_count,
            "phone_detected": phone_detected,
            "status": status
        })

    except Exception as e:

        print("Detection Error:", e)

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)