"""
GOATA biomechanics scorer.

Takes the 33 pose landmarks MediaPipe finds in ONE frame and scores three
GOATA movement checks for each leg:

    heel_away  - heel sits outward of the ankle
    iabh       - inside ankle bone high (the arch is not collapsing)
    bow_load   - knee bows outward from the hip-ankle line

Each check gets a score from 0 to 100 and a pass/fail. The three scores are
combined into a weighted composite per leg, and the two legs are averaged
into an overall score.

Works from the FRONT or the BACK: each frame, detect_view() works out which
way the athlete is facing, so "outward" always means away from the body.
Side-on views can't be scored reliably and are flagged as "side".

Usage:
    from goata.scorer import score_frame
    result = score_frame(result.pose_landmarks[0], w, h)

    # Override some settings (anything you leave out keeps its default):
    result = score_frame(landmarks, w, h, config={"bow_load": {"pass_score": 70}})
"""
import numpy as np

# ---------------------------------------------------------------------------
# Landmark indices (from the MediaPipe Pose Landmarker docs).
# "left" and "right" mean the ATHLETE'S left and right, not the image's.
# ---------------------------------------------------------------------------
LANDMARKS = {
    "left": {"hip": 23, "knee": 25, "ankle": 27, "heel": 29, "toe": 31},
    "right": {"hip": 24, "knee": 26, "ankle": 28, "heel": 30, "toe": 32},
}

# If the hips are closer together than this (as a fraction of leg length),
# the athlete is probably turned side-on to the camera and the left/right
# measurements stop meaning anything.
SIDE_VIEW_HIP_RATIO = 0.08

# ---------------------------------------------------------------------------
# Settings. Every number here is a STARTING GUESS to be tuned on real clips.
#
# The measurements are divided by leg length (hip-to-ankle distance), so
# they mean the same thing whether the athlete is near the camera or far
# away. 0.04 means "4% of the athlete's leg length".
#
#   zero_at    - a measurement at or below this scores 0
#   full_at    - a measurement at or above this scores 100
#                (anything in between is scaled in a straight line)
#   pass_score - the score needed to pass
#   weight     - how much this check counts toward the composite
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "heel_away": {"zero_at": 0.00, "full_at": 0.08, "pass_score": 60, "weight": 0.25},
    "iabh": {"zero_at": 0.03, "full_at": 0.10, "pass_score": 60, "weight": 0.375},
    "bow_load": {"zero_at": 0.00, "full_at": 0.06, "pass_score": 60, "weight": 0.375},
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def calculate_angle(a, b, c):
    """Angle at point b (in degrees) formed by points a-b-c.

    Not used by the three checks below yet, but needed for future
    angle-based checks such as knee bend.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def get_coords(landmarks, idx, w, h):
    """Turn one landmark's 0.0-1.0 position into pixel coordinates.

    We use pixels because normalized x and y have different scales when the
    frame isn't square. On a 1920x1080 frame, 0.1 in x is 192 px but 0.1 in
    y is only 108 px. Distances only make sense once both are in pixels.
    """
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def signed_distance_to_line(p, a, b):
    """Perpendicular distance from point p to the line through a and b.

    Uses the 2D cross product: cross(b - a, p - a) / |b - a|.
    The SIGN tells you which side of the line p is on. For a line pointing
    DOWN the frame (like hip -> ankle), positive = toward image left.

    We write the cross product out by hand (x1*y2 - y1*x2) because NumPy 2
    deprecates np.cross on 2D vectors.
    """
    line = np.array(b) - np.array(a)
    to_point = np.array(p) - np.array(a)
    cross = line[0] * to_point[1] - line[1] * to_point[0]
    length = np.linalg.norm(line)
    if length == 0:
        return 0.0  # hip and ankle on the same pixel; no line to measure from
    return cross / length


def detect_view(landmarks, w, h):
    """Work out which way the athlete faces, and which way is "outward" per leg.

    Trick: MediaPipe labels hips by the ATHLETE'S left/right.
      - From BEHIND, their left hip shows up on the image's left  (left x < right x).
      - From the FRONT, it's mirrored: their left hip is on image right.
    Either way, "outward" for a leg is simply "away from the other hip",
    so we don't need to know the view to get the direction right.

    Returns (view, outward) where view is "back", "front" or "side" and
    outward = {"left": -1 or +1, "right": -1 or +1} in image x.
    """
    left_hip = np.array(get_coords(landmarks, LANDMARKS["left"]["hip"], w, h))
    right_hip = np.array(get_coords(landmarks, LANDMARKS["right"]["hip"], w, h))
    left_ankle = np.array(get_coords(landmarks, LANDMARKS["left"]["ankle"], w, h))
    right_ankle = np.array(get_coords(landmarks, LANDMARKS["right"]["ankle"], w, h))

    hip_gap_x = left_hip[0] - right_hip[0]
    left_is_outward_positive = 1 if hip_gap_x > 0 else -1
    outward = {"left": left_is_outward_positive, "right": -left_is_outward_positive}

    avg_leg = (np.linalg.norm(left_hip - left_ankle) + np.linalg.norm(right_hip - right_ankle)) / 2
    if abs(hip_gap_x) < SIDE_VIEW_HIP_RATIO * max(avg_leg, 1e-6):
        view = "side"
    else:
        view = "back" if hip_gap_x < 0 else "front"
    return view, outward


def to_score(value, zero_at, full_at):
    """Scale a measurement onto 0-100 in a straight line, clamped at both ends."""
    fraction = (value - zero_at) / (full_at - zero_at)
    return int(round(np.clip(fraction, 0.0, 1.0) * 100))


# ---------------------------------------------------------------------------
# The checks.
#
# Each one takes:
#   pts     - dict of this leg's points in pixels, e.g. pts["knee"] = (x, y)
#   outward - -1 or +1: which image-x direction points AWAY from the body
#             for this leg (from detect_view)
# and returns ONE number: the measurement as a fraction of leg length.
# Bigger = better for every check, so they can all share to_score().
# ---------------------------------------------------------------------------
def leg_length(pts):
    """Hip-to-ankle distance in pixels, used to make measurements size-independent."""
    return max(np.linalg.norm(np.array(pts["hip"]) - np.array(pts["ankle"])), 1e-6)


def measure_heel_away(pts, outward):
    """How high the heel is lifted above the ankle.
    
    During a heel flick, the heel kicks UP behind the body.
    ankle.y - heel.y: positive = heel is above ankle (good).
    In MediaPipe, smaller y = higher on screen.
    """
    vertical_lift = pts["ankle"][1] - pts["heel"][1]
    return vertical_lift / leg_length(pts)


def measure_iabh(pts, outward):
    """How high the ankle sits above the toe (foot index).

    Image y grows DOWNWARD, so "ankle above toe" means ankle.y < toe.y,
    which is why we take toe - ankle. If the arch collapses, the ankle drops
    toward the toe's height and this number shrinks toward 0.
    """
    height = pts["toe"][1] - pts["ankle"][1]
    return height / leg_length(pts)


def measure_bow_load(pts, outward):
    """How far the knee bows OUTWARD from the straight hip-to-ankle line.

    Positive = knee outside the line (bow loaded, good).
    Negative = knee caving inward (valgus), the classic ACL risk position.
    """
    distance = signed_distance_to_line(pts["knee"], pts["hip"], pts["ankle"])
    # signed_distance_to_line gives positive = image LEFT for a downward line.
    # Image left is outward only when outward == -1, so multiply by -outward.
    return -outward * distance / leg_length(pts)


# To add a new check: write a measure_ function above, add it here, and add
# its settings to DEFAULT_CONFIG. Nothing else needs to change.
CHECKS = {
    "heel_away": measure_heel_away,
    "iabh": measure_iabh,
    "bow_load": measure_bow_load,
}


# ---------------------------------------------------------------------------
# Putting it all together
# ---------------------------------------------------------------------------
def merge_config(overrides):
    """Start from DEFAULT_CONFIG and replace only the values you passed in."""
    config = {name: dict(settings) for name, settings in DEFAULT_CONFIG.items()}
    for name, settings in (overrides or {}).items():
        config.setdefault(name, {}).update(settings)
    return config


def score_leg(landmarks, side, outward, w, h, config):
    """Run every check on one leg and add the weighted composite."""
    pts = {name: get_coords(landmarks, idx, w, h) for name, idx in LANDMARKS[side].items()}

    results = {}
    weighted_total = 0.0
    weight_sum = 0.0
    for name, measure in CHECKS.items():
        settings = config[name]
        score = to_score(measure(pts, outward), settings["zero_at"], settings["full_at"])
        # bool() so we return a plain Python True/False, not a numpy bool
        results[name] = {"score": score, "pass": bool(score >= settings["pass_score"])}
        weighted_total += score * settings["weight"]
        weight_sum += settings["weight"]

    # Dividing by weight_sum means the weights don't have to add up to exactly 1
    results["composite"] = int(round(weighted_total / weight_sum))
    return results


def score_frame(landmarks, w, h, config=None):
    """Score one frame.

    landmarks - result.pose_landmarks[0] from the MediaPipe Pose Landmarker
    w, h      - frame width and height in pixels
    config    - optional overrides for DEFAULT_CONFIG

    Returns a dict like:
        {"left":  {"heel_away": {"score": 85, "pass": True}, ..., "composite": 72},
         "right": {...},
         "overall": 71,
         "view": "back"}      # "back", "front" or "side" (side = don't trust it)
    """
    config = merge_config(config)
    view, outward = detect_view(landmarks, w, h)
    left = score_leg(landmarks, "left", outward["left"], w, h, config)
    right = score_leg(landmarks, "right", outward["right"], w, h, config)
    return {
        "left": left,
        "right": right,
        "overall": int(round((left["composite"] + right["composite"]) / 2)),
        "view": view,
    }


# ---------------------------------------------------------------------------
# Quick self-test with made-up landmarks:  python src/goata/scorer.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pprint import pprint
    from types import SimpleNamespace

    # Back view of a standing athlete on a 1000x1000 frame.
    # Left leg is on image LEFT. Both knees bow out, heels point out,
    # ankles sit well above the toes.
    fake = [SimpleNamespace(x=0.5, y=0.5) for _ in range(33)]
    positions = {
        23: (0.45, 0.50), 24: (0.55, 0.50),  # hips
        25: (0.42, 0.70), 26: (0.58, 0.70),  # knees, bowed outward
        27: (0.44, 0.90), 28: (0.56, 0.90),  # ankles
        29: (0.43, 0.93), 30: (0.57, 0.93),  # heels, slightly outward
        31: (0.44, 0.96), 32: (0.56, 0.96),  # toes, below the ankles
    }
    for idx, (x, y) in positions.items():
        fake[idx] = SimpleNamespace(x=x, y=y)

    print("Good posture:")
    pprint(score_frame(fake, 1000, 1000), sort_dicts=False)

    # Now cave the LEFT knee inward (toward image right) and check it fails
    fake[25] = SimpleNamespace(x=0.47, y=0.70)
    print("\nLeft knee caving in:")
    pprint(score_frame(fake, 1000, 1000), sort_dicts=False)
