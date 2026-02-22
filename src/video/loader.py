import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import deque

# =========================
# MODELS
# =========================
person_detector = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt")

# =========================
# CONSTANTS
# =========================
WRIST_SPEED_THRESH = 18
HIP_DROP_THRESH = 20
STRIKE_COOLDOWN = 0.8
STANCE_HISTORY = 15
BUCKET_SIZE = 10

# =========================
# GLOBAL STATE
# =========================
FIGHTERS = {
    "A": {"bbox": None, "center": None},
    "B": {"bbox": None, "center": None},
}

PREV_POSE = {"A": None, "B": None}
LAST_STRIKE_TIME = {"A": None, "B": None}
STRIKE_COUNT = {"A": 0, "B": 0}
STRIKE_BUCKETS = {"A": {}, "B": {}}

STANCE_BUFFER = {
    "A": deque(maxlen=STANCE_HISTORY),
    "B": deque(maxlen=STANCE_HISTORY),
}

# =========================
# RESET
# =========================
def reset_state():
    global FIGHTERS, PREV_POSE, LAST_STRIKE_TIME, STRIKE_COUNT, STRIKE_BUCKETS, STANCE_BUFFER
    FIGHTERS = {
        "A": {"bbox": None, "center": None},
        "B": {"bbox": None, "center": None},
    }
    PREV_POSE = {"A": None, "B": None}
    LAST_STRIKE_TIME = {"A": None, "B": None}
    STRIKE_COUNT = {"A": 0, "B": 0}
    STRIKE_BUCKETS = {"A": {}, "B": {}}
    STANCE_BUFFER = {
        "A": deque(maxlen=STANCE_HISTORY),
        "B": deque(maxlen=STANCE_HISTORY),
    }

# =========================
# VIDEO
# =========================
def load_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video loaded | FPS={fps:.2f} | Resolution={w}x{h}")
    return cap, fps, (w, h)

# =========================
# ROI
# =========================
def get_octagon_roi(frame):
    h, w, _ = frame.shape
    top, bottom = int(0.28*h), int(0.82*h)
    left, right = int(0.18*w), int(0.82*w)
    return frame[top:bottom, left:right], (left, top, right, bottom)

# =========================
# DETECTION
# =========================
def detect_people(roi):
    results = person_detector(roi, conf=0.4, classes=[0], verbose=False)
    boxes = []
    for r in results:
        for b in r.boxes:
            boxes.append(tuple(map(int, b.xyxy[0])))
    return boxes

# =========================
# IDENTITY MEMORY (CORE FIX)
# =========================
def bbox_center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

def assign_fighters(detections):
    global FIGHTERS

    if len(detections) < 2:
        return FIGHTERS

    centers = [(b, bbox_center(b)) for b in detections]

    # Initial assignment
    if FIGHTERS["A"]["bbox"] is None:
        centers = sorted(centers, key=lambda x: x[1][0])  # left-right once
        FIGHTERS["A"]["bbox"], FIGHTERS["A"]["center"] = centers[0]
        FIGHTERS["B"]["bbox"], FIGHTERS["B"]["center"] = centers[1]
        return FIGHTERS

    # Distance-based reassignment
    assigned = {}
    for label in ["A", "B"]:
        prev_center = FIGHTERS[label]["center"]
        dists = [(b, c, np.linalg.norm(np.array(c) - np.array(prev_center))) for b, c in centers]
        chosen = min(dists, key=lambda x: x[2])
        assigned[label] = chosen

    # Ensure no duplicate assignment
    if assigned["A"][0] == assigned["B"][0]:
        other = [c for c in centers if c[0] != assigned["A"][0]][0]
        assigned["B"] = other

    for label in ["A", "B"]:
        FIGHTERS[label]["bbox"] = assigned[label][0]
        FIGHTERS[label]["center"] = assigned[label][1]

    return FIGHTERS

# =========================
# POSE
# =========================
def estimate_pose(roi, bbox):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    crop = roi[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    res = pose_model(crop, conf=0.3, verbose=False)
    for r in res:
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            return r.keypoints.xy.cpu().numpy()[0]
    return None

def draw_pose(roi, bbox, pose, color):
    if bbox is None or pose is None:
        return
    x1, y1, _, _ = bbox
    for x, y in pose:
        cv2.circle(roi, (int(x1+x), int(y1+y)), 3, color, -1)

# =========================
# STANCE
# =========================
def estimate_stance(pose, opp_bbox):
    if pose is None or opp_bbox is None:
        return None
    left_ankle = pose[15]
    right_ankle = pose[16]
    opp_x = (opp_bbox[0] + opp_bbox[2]) / 2
    return "ORTHODOX" if abs(left_ankle[0] - opp_x) < abs(right_ankle[0] - opp_x) else "SOUTHPAW"

def stable_stance(label, stance):
    if stance:
        STANCE_BUFFER[label].append(stance)
    if not STANCE_BUFFER[label]:
        return "UNKNOWN"
    return max(set(STANCE_BUFFER[label]), key=STANCE_BUFFER[label].count)

# =========================
# STRIKE LOGIC (FINAL FIX)
# =========================
def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def is_standing(prev_pose, curr_pose):
    return abs(curr_pose[11][1] - prev_pose[11][1]) < HIP_DROP_THRESH

def wrist_in_bbox(w, b):
    return b[0] < w[0] < b[2] and b[1] < w[1] < b[3]

def valid_strike(prev_pose, curr_pose, opp_bbox):
    opp_center = (
        (opp_bbox[0] + opp_bbox[2]) / 2,
        (opp_bbox[1] + opp_bbox[3]) / 2
    )
    for idx in [9, 10]:
        prev_w = prev_pose[idx]
        curr_w = curr_pose[idx]
        speed = dist(prev_w, curr_w)
        if speed < WRIST_SPEED_THRESH:
            continue
        if dist(curr_w, opp_center) >= dist(prev_w, opp_center):
            continue
        if wrist_in_bbox(curr_w, opp_bbox):
            return True
    return False

# =========================
# MAIN PIPELINE
# =========================
def read_frames(cap, fps, size, output_path):
    reset_state()
    os.makedirs("outputs", exist_ok=True)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 9000)
    frame_id = 9000

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_id / fps
        roi, roi_box = get_octagon_roi(frame)
        cv2.rectangle(frame, roi_box[:2], roi_box[2:], (0,255,0), 2)

        detections = detect_people(roi)
        fighters = assign_fighters(detections)

        poseA = estimate_pose(roi, fighters["A"]["bbox"])
        poseB = estimate_pose(roi, fighters["B"]["bbox"])

        stanceA = stable_stance("A", estimate_stance(poseA, fighters["B"]["bbox"]))
        stanceB = stable_stance("B", estimate_stance(poseB, fighters["A"]["bbox"]))

        for label, pose, prev, opp in [
            ("A", poseA, PREV_POSE["A"], fighters["B"]["bbox"]),
            ("B", poseB, PREV_POSE["B"], fighters["A"]["bbox"]),
        ]:
            if pose is None or prev is None or opp is None:
                continue
            if not is_standing(prev, pose):
                continue
            if valid_strike(prev, pose, opp):
                last = LAST_STRIKE_TIME[label]
                if last is None or t-last > STRIKE_COOLDOWN:
                    STRIKE_COUNT[label] += 1
                    LAST_STRIKE_TIME[label] = t
                    bucket = BUCKET_SIZE * int(t//BUCKET_SIZE)
                    STRIKE_BUCKETS[label][bucket] = STRIKE_BUCKETS[label].get(bucket,0)+1

        PREV_POSE["A"] = poseA
        PREV_POSE["B"] = poseB

        # Draw ROI info
        if fighters["A"]["bbox"]:
            cv2.rectangle(roi, fighters["A"]["bbox"][:2], fighters["A"]["bbox"][2:], (255,0,0), 2)
            draw_pose(roi, fighters["A"]["bbox"], poseA, (0,0,255))
        if fighters["B"]["bbox"]:
            cv2.rectangle(roi, fighters["B"]["bbox"][:2], fighters["B"]["bbox"][2:], (0,255,255), 2)
            draw_pose(roi, fighters["B"]["bbox"], poseB, (255,0,255))

        # HUD
        cv2.putText(frame, f"A | {stanceA} | Strikes: {STRIKE_COUNT['A']}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        cv2.putText(frame, f"B | {stanceB} | Strikes: {STRIKE_COUNT['B']}",
                    (frame.shape[1]-430,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        writer.write(frame)
        frame_id += 1

    writer.release()
    cap.release()
    generate_momentum_graph()
    print("Processing complete.")

# =========================
# GRAPH
# =========================
def generate_momentum_graph():
    times = sorted(set(STRIKE_BUCKETS["A"]) | set(STRIKE_BUCKETS["B"]))
    a = [STRIKE_BUCKETS["A"].get(t,0) for t in times]
    b = [STRIKE_BUCKETS["B"].get(t,0) for t in times]
    plt.figure(figsize=(10,5))
    plt.plot(times, a, label="Fighter A")
    plt.plot(times, b, label="Fighter B")
    plt.xlabel("Time (s)")
    plt.ylabel("Strikes / 10s")
    plt.title("Fight Momentum")
    plt.legend()
    plt.grid()
    plt.savefig("outputs/momentum_graph.png")
    plt.close()
