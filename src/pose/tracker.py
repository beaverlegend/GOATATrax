"""
GOATATrax pose tracker.

Usage (run from anywhere):
    python src/pose/tracker.py                  # live webcam
    python src/pose/tracker.py clips/jump.mp4   # video file (e.g. filmed on your phone)

Controls:
    q      quit
    space  pause / resume (video files only)
    m      show / hide the measurement drawing (takes effect on the next frame)
"""
import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Find the model relative to THIS file, so the script works no matter
# which folder you run it from. tracker.py -> pose -> src -> GOATATrax
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker.task"

# When you run "python src/pose/tracker.py", Python only looks for imports
# inside src/pose. Adding src/ to the search path lets us import our own
# goata package that lives next door in src/goata.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from goata import scorer  # noqa: E402  (import must come after the path fix)
from goata.scorer import score_frame  # noqa: E402

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


PASS_COLOR = (0, 200, 0)    # green (OpenCV colors are BGR, not RGB)
FAIL_COLOR = (0, 0, 255)    # red
TEXT_COLOR = (255, 255, 255)
CHECK_LABELS = {"heel_away": "Heel away", "iabh": "IABH", "bow_load": "Bow load"}


def draw_scores(frame, scores):
    """Draw the GOATA score panel in the top-left corner (in place).

    scores is the dict from score_frame(), or None if no pose was found.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = 22

    # Build the list of (text, color) lines first so we know how tall
    # the background box needs to be.
    if scores is None:
        lines = [("No pose detected", FAIL_COLOR)]
    else:
        lines = [(f"GOATA overall: {scores['overall']}", TEXT_COLOR)]
        if scores["view"] == "side":
            lines.append(("SIDE VIEW - don't trust", FAIL_COLOR))
        else:
            lines.append((f"View: {scores['view']}", TEXT_COLOR))
        for side in ("left", "right"):
            leg = scores[side]
            lines.append((f"{side.upper()}  composite {leg['composite']}", TEXT_COLOR))
            for check, label in CHECK_LABELS.items():
                result = leg[check]
                verdict = "PASS" if result["pass"] else "FAIL"
                color = PASS_COLOR if result["pass"] else FAIL_COLOR
                lines.append((f"  {label}: {result['score']} {verdict}", color))

    # Semi-transparent dark box so the text is readable over any video:
    # draw a solid box on a copy, then blend the copy 60/40 with the frame.
    box_w, box_h = 260, line_h * len(lines) + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)

    for i, (text, color) in enumerate(lines):
        y = 24 + i * line_h
        cv2.putText(frame, text, (10, y), font, 0.55, color, 1, cv2.LINE_AA)


BOW_LINE_COLOR = (0, 255, 255)   # yellow: straight hip -> ankle line
BOW_COLOR = (255, 0, 255)        # magenta: knee's distance from that line
HEEL_COLOR = (255, 255, 0)       # cyan: heel's sideways offset from ankle
IABH_COLOR = (0, 165, 255)       # orange: ankle height above toe


def to_px(point):
    """OpenCV drawing needs whole-number pixel positions."""
    return int(round(point[0])), int(round(point[1]))


def draw_measurements(frame, landmarks):
    """Draw exactly what the scorer measures, on each leg (in place).

    Uses the scorer's own measure_ functions, so the numbers on screen are
    the same numbers that get turned into scores.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    _, outward_dirs = scorer.detect_view(landmarks, w, h)

    for side in ("left", "right"):
        outward = outward_dirs[side]
        pts = {name: scorer.get_coords(landmarks, idx, w, h)
               for name, idx in scorer.LANDMARKS[side].items()}
        hip, knee, ankle, heel, toe = (pts[k] for k in ("hip", "knee", "ankle", "heel", "toe"))

        # BOW LOAD: straight line hip -> ankle, then the shortest (perpendicular)
        # path from the knee to that line. "closest" is the foot of that path:
        # move along the line by t, where t comes from the dot product.
        a, b, p = (np.array(v) for v in (hip, ankle, knee))
        t = np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-6)
        closest = a + t * (b - a)
        cv2.line(frame, to_px(hip), to_px(ankle), BOW_LINE_COLOR, 2)
        cv2.line(frame, to_px(knee), to_px(closest), BOW_COLOR, 3)

        # HEEL AWAY: sideways (x only) gap between ankle and heel, drawn at heel height
        cv2.arrowedLine(frame, to_px((ankle[0], heel[1])), to_px(heel), HEEL_COLOR, 2, tipLength=0.4)

        # IABH: straight down from the ankle to the toe's height
        cv2.line(frame, to_px(ankle), to_px((ankle[0], toe[1])), IABH_COLOR, 2)
        cv2.line(frame, to_px((ankle[0] - 8, toe[1])), to_px((ankle[0] + 8, toe[1])), IABH_COLOR, 2)

        # Raw values as % of leg length (+ means "good direction")
        values = [
            (f"bow {scorer.measure_bow_load(pts, outward) * 100:+.1f}%", BOW_COLOR, knee),
            (f"heel {scorer.measure_heel_away(pts, outward) * 100:+.1f}%", HEEL_COLOR, heel),
            (f"iabh {scorer.measure_iabh(pts, outward) * 100:+.1f}%", IABH_COLOR, ankle),
        ]
        for text, color, anchor in values:
            # Put each leg's labels on its outer side so they don't overlap
            (tw, _), _ = cv2.getTextSize(text, font, 0.5, 1)
            x = anchor[0] - tw - 15 if outward < 0 else anchor[0] + 15
            cv2.putText(frame, text, to_px((x, anchor[1])), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, to_px((x, anchor[1])), font, 0.5, color, 1, cv2.LINE_AA)

        # Mark which leg MediaPipe thinks is which
        cv2.putText(frame, side[0].upper(), to_px((hip[0] - 5, hip[1] - 10)),
                    font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


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
    show_measurements = True  # press m to hide/show the measurement drawing

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

                scores = None
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    draw_pose(frame, landmarks)
                    h, w = frame.shape[:2]
                    scores = score_frame(landmarks, w, h)

                display = scale_to_fit(frame, DISPLAY_SIZE)
                # Draw text AFTER shrinking, so it stays readable
                # (text drawn on a 4K frame would shrink to a tiny size).
                if show_measurements and result.pose_landmarks:
                    draw_measurements(display, result.pose_landmarks[0])
                draw_scores(display, scores)

            cv2.imshow("GOATATrax - Pose Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and is_file:
                paused = not paused
            if key == ord("m"):
                show_measurements = not show_measurements

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
