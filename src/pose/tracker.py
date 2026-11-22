"""
GOATATrax pose tracker.

Usage (run from anywhere):
    python src/pose/tracker.py                  # live webcam
    python src/pose/tracker.py clips/jump.mp4   # video file (e.g. filmed on your phone)

Controls:
    q      quit
    space  pause / resume (video files only)
"""
import argparse
from pathlib import Path

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Find the model relative to THIS file, so the script works no matter
# which folder you run it from. tracker.py -> pose -> src -> GOATATrax
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker.task"

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

# Longest side (in pixels) of the image we hand to MediaPipe.
# Smaller = faster detection. We keep the aspect ratio so a tall
# portrait phone video doesn't get squashed.
DETECT_SIZE = 640

# Longest side of the preview window, so a 4K phone clip fits on screen.
DISPLAY_SIZE = 960


def scale_to_fit(frame, max_side):
    """Shrink a frame so its longest side is max_side, keeping its shape."""
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame  # already small enough, don't upscale
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def draw_pose(frame, landmarks):
    """Draw the skeleton and GOATA landmarks onto the frame (in place)."""
    # Use the frame's real size. Landmarks are 0.0-1.0 fractions of it.
    h, w = frame.shape[:2]

    # Skeleton lines first, so the dots sit on top
    for start_idx, end_idx in CONNECTIONS:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Dots for the landmarks GOATA cares about
    for idx, landmark in enumerate(landmarks):
        if idx not in RELEVANT:
            continue
        x, y = int(landmark.x * w), int(landmark.y * h)
        color = (255, 200, 0) if idx in LOWER_BODY else (0, 255, 255)  # teal / yellow
        cv2.circle(frame, (x, y), 5, color, -1)


def run(source):
    """Track pose on a webcam (source=0) or a video file (source=path)."""
    is_file = isinstance(source, str)
    if is_file and not Path(source).exists():
        raise SystemExit(f"Video not found: {source}")
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")

    # Frames per second. Some webcams report 0, so fall back to 30.
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if is_file:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Loaded {source}: {total} frames at {fps:.0f} fps")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
    )

    frame_index = 0
    paused = False

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break  # end of video (or camera disconnected)

                # Detect on a smaller copy for speed
                small = scale_to_fit(frame, DETECT_SIZE)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
                )

                # VIDEO mode needs a timestamp that always goes up.
                # Derive it from the frame count so it works the same for
                # webcams and video files (frame 30 -> 1000 ms at 30fps).
                timestamp_ms = int(frame_index * 1000 / fps)
                frame_index += 1
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    draw_pose(frame, result.pose_landmarks[0])

                display = scale_to_fit(frame, DISPLAY_SIZE)

            cv2.imshow("GOATATrax - Pose Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and is_file:
                paused = not paused

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="GOATATrax pose tracker")
    parser.add_argument(
        "video",
        nargs="?",
        help="path to a video file; leave out to use the webcam",
    )
    args = parser.parse_args()
    run(args.video if args.video else 0)


if __name__ == "__main__":
    main()
