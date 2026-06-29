"""
ExerciseAI
Single-file Flask application

Modes
-----
1. Squat
2. Curl

Features
--------
✔ Live MJPEG Streaming
✔ Squat Classification
✔ Curl Classification
✔ Rep Counter
✔ Latest Classification Memory
✔ Start / Stop Camera
✔ Status API
"""

import cv2
import mediapipe as mp
import numpy as np

import threading
import time

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request
)

# ---------------------------------------
# Flask
# ---------------------------------------

app = Flask(__name__)

# ---------------------------------------
# Angle Calculation
# ---------------------------------------

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = (
        np.arctan2(c[1]-b[1], c[0]-b[0])
        -
        np.arctan2(a[1]-b[1], a[0]-b[0])
    )

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360-angle

    return angle


# ---------------------------------------
# MediaPipe
# ---------------------------------------

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------------------
# Camera Class
# ---------------------------------------

class VideoCamera:

    def __init__(self):

        self.cap = None

        self.running = False

        self.thread = None

        self.lock = threading.Lock()

        self.frame = None

    # ----------------------------

    def start(self):

        if self.running:
            return

        self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():

            raise RuntimeError("Cannot open camera")

        self.running = True

        self.thread = threading.Thread(
            target=self.update,
            daemon=True
        )

        self.thread.start()
        print("Camera thread launched")

    # ----------------------------

    def update(self):

        print("Update thread started")

        while self.running:

            success, frame = self.cap.read()

            print("Reading...", success)

            if success:

                with self.lock:
                    self.frame = frame.copy()

            time.sleep(0.01)

        print("Update thread stopped")
    # ----------------------------

    def get_frame(self):

        with self.lock:

            if self.frame is None:

                print("Frame still None")

                return None

            print("Returning frame")

            return self.frame.copy()
    # ----------------------------

    def stop(self):

        self.running = False

        if self.thread:

            self.thread.join(timeout=1)

        if self.cap:

            self.cap.release()

        self.frame = None

# ---------------------------------------
# Camera Object
# ---------------------------------------

camera = VideoCamera()

# ---------------------------------------
# Current Mode
# ---------------------------------------

current_mode = "squat"

# ---------------------------------------
# Shared State
# ---------------------------------------

state = {

    "camera": False,

    "mode": "SQUAT",

    "classification": "Standing",

    "angle": 0,

    "curl": {

        "angles": [],

        "completed": False,

        "reps": 0
    },

    "squat": {

        "langles": [],

        "rangles": [],

        "completed": False,

        "reps": 0
    }

}

state_lock = threading.Lock()
# ---------------------------------------
# Frame Processing
# ---------------------------------------

def process_frame(frame):

    global current_mode

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if not results.pose_landmarks:

        cv2.putText(
            image,
            "No Pose Detected",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        return image

    landmarks = results.pose_landmarks.landmark

    # =====================================================
    #                SQUAT MODE
    # =====================================================

    if current_mode == "squat":

        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        ]

        left_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        ]

        left_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        ]


        right_hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        ]

        right_knee = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
        ]

        right_ankle = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
        ]


        left_angle = calculate_angle(
            left_hip,
            left_knee,
            left_ankle
        )

        right_angle = calculate_angle(
            right_hip,
            right_knee,
            right_ankle
        )

        squat_type = "Standing"

        display_angle = (left_angle + right_angle) / 2

        with state_lock:

            if left_angle >= 120 and right_angle >= 120:

                squat_type = "Standing"

            elif left_angle < 120 and right_angle < 120:

                squat_type = "Squatting"

                state["squat"]["langles"].append(left_angle)

                state["squat"]["rangles"].append(right_angle)
                        # ----------------------------------------
            # Your original classification logic
            # ----------------------------------------

            if squat_type == "Standing":
                if (
                    state["squat"]["langles"]
                    and
                    state["squat"]["rangles"]
                ):

                    min_langle = min(state["squat"]["langles"])

                    min_rangle = min(state["squat"]["rangles"])

                    avg_angle = (min_langle + min_rangle) / 2

                    display_angle = avg_angle

                    if avg_angle < 90:

                        squat_type = "Deep Squat"

                    elif avg_angle < 120:

                        squat_type = "Medium Squat"

                    else:

                        squat_type = "Shallow Squat"

                    # Count every completed squat only once
                    if not state["squat"]["completed"]:

                        state["squat"]["reps"] += 1

                        state["squat"]["completed"] = True

                    state["squat"]["langles"].clear()

                    state["squat"]["rangles"].clear()

            # ----------------------------------------
            # Reset when user starts next squat
            # ----------------------------------------

            if squat_type == "Squatting":

                state["squat"]["completed"] = False

            # ----------------------------------------
            # Save latest values
            # ----------------------------------------

            state["mode"] = "SQUAT"

            if squat_type in (
                "Deep Squat",
                "Medium Squat",
                "Shallow Squat"
            ):
                state["classification"] = squat_type

            state["angle"] = int(display_angle)

        cv2.putText(
            image,
            f"Status : {squat_type}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        cv2.putText(
            image,
            f"Angle : {int(display_angle)}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        # =====================================================
    #                CURL MODE
    # =====================================================

    elif current_mode == "curl":

        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ]

        left_elbow = [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        ]

        left_wrist = [
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        ]

        left_angle = calculate_angle(
            left_shoulder,
            left_elbow,
            left_wrist
        )

        display_angle = left_angle

        curl_type = "Arm Extended"

        with state_lock:

            if left_angle < 120:

                curl_type = "Arm Flexing"

                state["curl"]["angles"].append(left_angle)

            else:

                curl_type = "Arm Extended"

            if curl_type == "Arm Extended":

                if state["curl"]["angles"]:

                    min_angle = min(state["curl"]["angles"])

                    display_angle = min_angle

                    if min_angle < 90:

                        curl_type = "Arm Flexed"

                    else:

                        curl_type = "Arm Partially Flexed"

                    # Count every completed curl only once
                    if not state["curl"]["completed"]:

                        state["curl"]["reps"] += 1

                        state["curl"]["completed"] = True
                    state["curl"]["angles"].clear()

            if curl_type == "Arm Flexing":

                state["curl"]["completed"] = False

            state["mode"] = "CURL"

            if curl_type in (
                "Arm Flexed",
                "Arm Partially Flexed"
            ):
                state["classification"] = curl_type

            state["angle"] = int(display_angle)
        cv2.putText(
            image,
            f"Status : {curl_type}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        cv2.putText(
            image,
            f"Angle : {int(display_angle)}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )
    mp.solutions.drawing_utils.draw_landmarks(
    image,
    results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    mp.solutions.drawing_styles.get_default_pose_landmarks_style()
)
    return image
# ---------------------------------------
# Draw Overlay
# ---------------------------------------

def draw_overlay(image):

    cv2.rectangle(
        image,
        (0, 0),
        (430, 150),
        (0, 0, 0),
        -1
    )

    with state_lock:

        if state["mode"] == "SQUAT":
            reps = state["squat"]["reps"]
        else:
            reps = state["curl"]["reps"]

        mode = state["mode"]
        classification = state["classification"]
        angle = state["angle"]

    cv2.putText(
        image,
        f"MODE : {mode}",
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    cv2.putText(
        image,
        f"TYPE : {classification}",
        (20,65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    cv2.putText(
        image,
        f"ANGLE : {angle}",
        (20,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        image,
        f"REPS : {reps}",
        (20,135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )

    return image


# ---------------------------------------
# Frame Generator
# ---------------------------------------

def gen_frames():

    while True:

        if not camera.running:

            blank = np.zeros((480,640,3),dtype=np.uint8)

            cv2.putText(
                blank,
                "Camera Stopped",
                (150,240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2
            )

            ret, buffer = cv2.imencode(".jpg", blank)

            frame = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + frame +
                b'\r\n'
            )

            time.sleep(0.1)

            continue

        frame = camera.get_frame()

        if frame is None:
            print("Frame is None")
            continue

        frame = process_frame(frame)

        if frame is None:
            print("process_frame returned None")
            continue

        frame = draw_overlay(frame)

        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )
# ---------------------------------------
# Flask Routes
# ---------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------
# Video Stream
# ---------------------------------------

@app.route("/video_feed")
def video_feed():

    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------
# Live Status
# ---------------------------------------

@app.route("/status")
def status():

    with state_lock:

        if state["mode"] == "SQUAT":
            reps = state["squat"]["reps"]
        else:
            reps = state["curl"]["reps"]

        return jsonify({

            "camera": camera.running,

            "mode": state["mode"],

            "classification": state["classification"],

            "angle": state["angle"],

            "reps": reps

        })


# ---------------------------------------
# Change Exercise Mode
# ---------------------------------------

@app.route("/set_mode")
def set_mode():

    global current_mode

    mode = request.args.get("mode", "squat")

    if mode in ("squat", "curl"):

        current_mode = mode

        with state_lock:

            state["mode"] = mode.upper()

            state["classification"] = "Waiting..."

            state["angle"] = 0

    return jsonify({"success": True})


# ---------------------------------------
# Start Camera
# ---------------------------------------

@app.route("/start")
def start_camera():

    if not camera.running:

        camera.start()

    with state_lock:
        state["camera"] = True

    return jsonify({"success": True})


# ---------------------------------------
# Stop Camera
# ---------------------------------------

@app.route("/stop")
def stop_camera():

    if camera.running:

        camera.stop()

    with state_lock:
        state["camera"] = False

    return jsonify({"success": True})


# ---------------------------------------
# Reset Squat Counter
# ---------------------------------------

@app.route("/reset_squat")
def reset_squat():

    with state_lock:

        state["squat"]["langles"].clear()
        state["squat"]["rangles"].clear()
        state["squat"]["completed"] = False
        state["squat"]["reps"] = 0

        if current_mode == "squat":
            state["classification"] = "Standing"
            state["angle"] = 0

    return jsonify({"success": True})


# ---------------------------------------
# Reset Curl Counter
# ---------------------------------------

@app.route("/reset_curl")
def reset_curl():

    with state_lock:

        state["curl"]["angles"].clear()
        state["curl"]["completed"] = False
        state["curl"]["reps"] = 0

        if current_mode == "curl":
            state["classification"] = "Arm Extended"
            state["angle"] = 0

    return jsonify({"success": True})



# ---------------------------------------
# Run Flask
# ---------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False
    )