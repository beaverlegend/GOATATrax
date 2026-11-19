import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Skeleton connections (tuple = faster, never changes)
CONNECTIONS = (
    (11, 12),              # shoulders
    (11, 13), (13, 15),    # left arm
    (12, 14), (14, 16),    # right arm
    (11, 23), (12, 24),    # torso sides
    (23, 24),              # hips
    (23, 25), (25, 27), (27, 29), (27, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # right leg
)

# GOATA lower body landmarks (teal)
LOWER_BODY = {23, 24, 25, 26, 27, 28, 29, 30, 31, 32}

# Head control landmarks (yellow)
HEAD_CONTROL = {0, 7, 8, 9, 10}

# Only draw these landmarks (skip irrelevant ones)
RELEVANT = LOWER_BODY | HEAD_CONTROL

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/pose_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO
)

cap = cv2.VideoCapture(0)

# Get original frame dimensions once (outside loop)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster detection
        small = cv2.resize(frame, (640, 480))

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        )

        # VIDEO mode needs a timestamp
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # Draw skeleton lines first (behind dots)
            # Landmarks are 0.0-1.0 so multiply by original w/h
            for start_idx, end_idx in CONNECTIONS:
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Only draw relevant landmarks
            for idx, landmark in enumerate(landmarks):
                if idx not in RELEVANT:
                    continue

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                if idx in LOWER_BODY:
                    color = (255, 200, 0)   # teal
                elif idx in HEAD_CONTROL:
                    color = (0, 255, 255)   # yellow
                
                cv2.circle(frame, (x, y), 5, color, -1)

        cv2.imshow("GOATATrax - Pose Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()