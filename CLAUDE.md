# GOATATrax — Project Context for Claude Code

## What This Project Is

GOATATrax is a computer vision tool that analyzes athlete movement using GOATA biomechanical principles to predict non-contact ACL tear probability in basketball players.

A user uploads a basketball clip. The tool tracks their body using pose estimation, scores how well they move against GOATA's seven movement laws, and outputs an ACL injury risk probability.

**Target audience:** coaches, athletes, physios, scouts — anyone who cares about movement quality and injury prevention.

---

## The GOATA Framework

GOATA (Greatest of All Time Actions) is a biomechanics system that reverse-engineers natural human movement by studying pain-free elite athletes, toddlers, and indigenous populations. It identifies seven universal movement laws that protect joints and prevent injury.

The principles GOATATrax measures:
- **Columns** — joints stacked and aligned vertically
- **Back-chain dominance** — posterior chain driving movement, not front chain
- **Bow load** — correct force loading before explosive movement
- **Heel flick** — proper heel mechanics during gait and running
- **Head control** — head level, crown forward, spine stacked
- **Inside ankle bone high (IABH)** — arch not collapsing
- **The recoil** — energy return through the kinetic chain

---

## Tech Stack

- **Language:** Python 3.15
- **Pose estimation:** MediaPipe Pose Landmarker 1.2.3 (33 keypoints)
- **Video processing:** OpenCV 5.0.0
- **ML (planned):** TBD — injury prediction model
- **Web app (planned):** TBD

---

## Project Structure

```
GOATATrax/
├── src/
│   ├── pose/
│   │   └── tracker.py        # pose estimation — CURRENT WORKING FILE
│   ├── goata/
│   │   └── scorer.py         # GOATA checks — NOT YET BUILT
│   └── ml/
│       └── predictor.py      # ACL prediction model — NOT YET BUILT
├── models/
│   └── pose_landmarker.task  # MediaPipe model file
├── tests/
├── requirements.txt
├── README.md
└── main.py                   # entry point — currently empty
```

---

## Current Build Status

| Feature | Status |
|---|---|
| Real-time pose tracking (webcam) | ✅ Working |
| Skeleton overlay with connections | ✅ Working |
| Color-coded landmarks (teal = lower body, yellow = head) | ✅ Working |
| VIDEO mode for smooth tracking | ✅ Working |
| Frame resize optimization (640x480 for detection) | ✅ Working |
| Video file upload mode | 🔜 Next to build |
| GOATA compliance scoring | 🔜 After video upload |
| ACL injury prediction model | 🔜 Later |
| Web app with upload UI | 🔜 Later |

---

## Key Implementation Details

### Pose Tracking (src/pose/tracker.py)
- Uses MediaPipe 1.2.3 new API (NOT the old mp.solutions API)
- Requires `models/pose_landmarker.task` model file
- Runs in `VisionRunningMode.VIDEO` for temporal smoothing
- Resizes frames to 640x480 before detection, draws on original frame size
- Uses `detect_for_video()` with timestamp for smooth tracking

### Landmark Color Coding
- **Teal (255, 200, 0)** — GOATA lower body landmarks: hips (23,24), knees (25,26), ankles (27,28), heels (29,30), foot index (31,32)
- **Yellow (0, 255, 255)** — Head control landmarks: nose (0), ears (7,8), mouth (9,10)
- **Gray (200, 200, 200)** — Everything else

### GOATA-Relevant Landmarks
```python
LOWER_BODY = {23, 24, 25, 26, 27, 28, 29, 30, 31, 32}  # hips to feet
HEAD_CONTROL = {0, 7, 8, 9, 10}                          # nose, ears, mouth
RELEVANT = LOWER_BODY | HEAD_CONTROL                      # union for drawing
```

### Skeleton Connections (tuple for performance)
```python
CONNECTIONS = (
    (11, 12),              # shoulders
    (11, 13), (13, 15),    # left arm
    (12, 14), (14, 16),    # right arm
    (11, 23), (12, 24),    # torso sides
    (23, 24),              # hips
    (23, 25), (25, 27), (27, 29), (27, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # right leg
)
```

---

## Build Order Going Forward

1. **Video file upload mode** — swap webcam input for a video file path
2. **GOATA scorer** — calculate joint angles, score each principle
3. **Score overlay** — display live GOATA score on video
4. **ACL prediction model** — train on injury history data
5. **Web app** — drag and drop upload, visual output, shareable results

---

## Important Notes

- Kerry is a first-year university student (FARM program, University of Waterloo), Python background but rusty on libraries
- Keep explanations clear — prioritize understanding over speed
- This is both a portfolio project and something Kerry wants to actually ship
- The project is also connected to UWAGGS (UWaterloo sports analytics club)
- GitHub repo: https://github.com/beaverlegend/GOATATrax
