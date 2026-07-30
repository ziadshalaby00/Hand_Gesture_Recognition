# =============================================================================
# =============================================================================
# This file is the main file responsible for running the mediapipe model, 
# detecting the hand, drawing the frame, retrieving information, 
# updating the frame, and sending it to the server, which then streams it to the web. 
# It also runs the streaming server and sends data to the file connected to socket io 
# so that it can send the data to the web.
# =============================================================================
# =============================================================================

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

from video_server import app, set_frame
from robot_3d import send_data

# ==============
# MediaPipe Hand Landmarker Initialization
# ==============
base_options = python.BaseOptions(model_asset_path='./hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)


# ==============
# Hand Skeleton Connection Definitions
# ==============
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (9,10), (10,11), (11,12),
    (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20),
    (5,9), (9,13), (13,17)
]


# ==============
# Camera Capture Initialization
# ==============
cap = cv2.VideoCapture(0)


# ==============
# Helper Functions: Drawing and Gesture Analysis
# ==============
def draw_manual(image, result):
    if not result.hand_landmarks: return image
    h, w, _ = image.shape
    for landmarks in result.hand_landmarks:
        for landmark in landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 8, (0, 255, 0), -1)
        for start_idx, end_idx in HAND_CONNECTIONS:
            start, end = landmarks[start_idx], landmarks[end_idx]
            pt1 = (int(start.x * w), int(start.y * h))
            pt2 = (int(end.x * w), int(end.y * h))
            cv2.line(image, pt1, pt2, (255, 0, 0), 3)
    return image


def get_fingers_status(landmarks):
    tips_ids, pip_ids = [8, 12, 16, 20], [6, 10, 14, 18]
    fingers_open = [1 if landmarks[tip].y < landmarks[pip].y else 0 for tip, pip in zip(tips_ids, pip_ids)]
    fingers_open.append(1 if landmarks[4].x > landmarks[3].x else 0)
    return fingers_open


def get_palm_center(landmarks, w, h):
    # landmarks[0] = wrist
    x = landmarks[0].x * w
    y = landmarks[0].y * h
    return x, y


# ==============
# Main Loop Configuration and State Variables
# ==============
action_map = {
    0: "sit",
    1: "Walking",
    2: "run",
    3: "dance",
    4: "punch",
    5: "stop"
}

history = []
MAX_HISTORY = 5
last_x = None
last_time = 0
latest_frame = None

# ==============
# Performance Metrics
# ==============
prev_frame_time = time.perf_counter()
pipeline_fps = 0
inference_avg = 0.0
processing_avg = 0.0

# ==============
# Main Frame Processing Loop
# ==============
def update_frame():
    global last_time, latest_frame, last_x, history
    global prev_frame_time, pipeline_fps, inference_avg, processing_avg

    while True:
        loop_start = time.perf_counter()
        
        ret, frame = cap.read()
        if not ret:
            continue

        # ==============
        # Camera Preprocessing and MediaPipe Inference
        # ==============
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        inference_start = time.perf_counter()
        
        timestamp_ms = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)
        
        inference_ms = (
            time.perf_counter() - inference_start
        ) * 1000

        # ==============
        # Default State Values
        # ==============
        hand_present = "No"
        status_text = "Closed"
        fingers_count = 0
        robot_action = "sit"
        direction = None

        # ==============
        # Hand Detection Reset Logic
        # ==============
        if not result.hand_landmarks:
            last_x = None
            history.clear()
        else:
            hand_present = "Yes"

            h, w, _ = frame.shape
            landmarks = result.hand_landmarks[0]

            # ==============
            # Finger Status Calculation
            # ==============
            fingers_list = get_fingers_status(landmarks)
            fingers_count = sum(fingers_list)
            status_text = "Open" if fingers_count >= 3 else "Closed"

            # ==============
            # Palm Center Position Extraction
            # ==============
            current_x, _ = get_palm_center(landmarks, w, h)

            # ==============
            # Position Smoothing with History Buffer
            # ==============
            history.append(current_x)
            if len(history) > MAX_HISTORY:
                history.pop(0)

            avg_x = sum(history) / len(history)

            # ==============
            # Horizontal Movement Direction Detection
            # ==============
            if last_x is not None:
                diff = avg_x - last_x

                if abs(diff) > 5:
                    direction = "walk-right" if diff > 0 else "walk-left"

            last_x = avg_x

            # ==============
            # Visual Overlay Drawing
            # ==============
            frame = draw_manual(frame, result)

        # ==============
        # Robot Action Mapping Based on Gesture
        # ==============
        robot_action = action_map.get(fingers_count, "sit")

        # override only when full hand open
        if fingers_count == 5 and direction:
            robot_action = direction

        # ==============
        # Frame Transmission to Video Server
        # ==============
        set_frame(frame)
        
        
        current_time = time.perf_counter()
        fps = 1 / (current_time - prev_frame_time)
        pipeline_fps = pipeline_fps * 0.9 + fps * 0.1
        
        prev_frame_time = current_time
        
        processing_ms = (
            time.perf_counter() - loop_start
        ) * 1000
        
     
        inference_avg = inference_avg * 0.9 + inference_ms * 0.1
        processing_avg = processing_avg * 0.9 + processing_ms * 0.1

        # ==============
        # Hand Data Transmission to Robot Client
        # ==============
        last_time = send_data(
            {
                "hand_present": hand_present,
                "status_text": status_text,
                "fingers_count": fingers_count,
                "robot_action": robot_action,

                "pipeline_fps": round(pipeline_fps, 1),
                "inference_ms": round(inference_avg, 2),
                "processing_ms": round(processing_avg, 2),
            },
            last_time,
            20
        )


# ==============
# FastAPI Server Runner Utility
# ==============
import threading
import uvicorn

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=5000)


# ==============
# Application Entry Point
# ==============
if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    update_frame()