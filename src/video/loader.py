# =============================================================
# fight-vision | src/video/loader.py
#
# Core pipeline: loads a fight video and processes it frame by
# frame to detect fighters, estimate pose, validate strikes,
# track momentum over time, and export annotated output.
#
# Pipeline order per frame:
#   1. Read frame from video
#   2. Crop octagon ROI
#   3. Detect fighters (YOLOv8)
#   4. Assign persistent fighter identities (centroid tracking)
#   5. Estimate pose keypoints per fighter (YOLOv8-Pose)
#   6. Determine and smooth stance (Orthodox / Southpaw)
#   7. Validate strikes using multi-constraint logic
#   8. Enforce cooldown to prevent double-counting
#   9. Aggregate strikes into time buckets for momentum graph
#  10. Draw overlays and write annotated frame to output video
# =============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from ultralytics import YOLO

import config
from src.video.utils import (
    euclidean_distance,
    bbox_center,
    point_in_bbox,
    draw_hud,
)

# ── Load models once at startup ───────────────────────────────
# Loading inside the loop would reload a 6MB+ file every frame
# (~thousands of times). Loading here = once per run.
person_detector = YOLO(config.DETECTION_MODEL)
pose_model      = YOLO(config.POSE_MODEL)


# ── Global state ──────────────────────────────────────────────
# These variables carry information forward across frames.
# A video frame is stateless by itself — these give it memory.
#
# Design note: for a production system this state would live
# inside a FightAnalyzer class to allow multiple concurrent runs.

FIGHTERS = {
    "A": {"bbox": None, "center": None},
    "B": {"bbox": None, "center": None},
}

PREV_POSE        = {"A": None, "B": None}  # pose from previous frame (for velocity)
LAST_STRIKE_TIME = {"A": None, "B": None}  # timestamp of last valid strike per fighter
STRIKE_COUNT     = {"A": 0,    "B": 0}     # running total of validated strikes
STRIKE_BUCKETS   = {"A": {},   "B": {}}    # {time_bucket_start: count} for graph
STANCE_BUFFER    = {                        # sliding window for stance smoothing
    "A": deque(maxlen=config.STANCE_HISTORY),
    "B": deque(maxlen=config.STANCE_HISTORY),
}


def reset_state():
    """
    Reset all global state to initial values.

    Called at the start of read_frames() to ensure a clean slate
    when processing a new video — prevents carryover from previous runs.
    """
    global FIGHTERS, PREV_POSE, LAST_STRIKE_TIME, STRIKE_COUNT, STRIKE_BUCKETS, STANCE_BUFFER
    FIGHTERS = {
        "A": {"bbox": None, "center": None},
        "B": {"bbox": None, "center": None},
    }
    PREV_POSE        = {"A": None, "B": None}
    LAST_STRIKE_TIME = {"A": None, "B": None}
    STRIKE_COUNT     = {"A": 0,    "B": 0}
    STRIKE_BUCKETS   = {"A": {},   "B": {}}
    STANCE_BUFFER    = {
        "A": deque(maxlen=config.STANCE_HISTORY),
        "B": deque(maxlen=config.STANCE_HISTORY),
    }


# ── VIDEO I/O ─────────────────────────────────────────────────

def load_video(path):
    """
    Open a video file and read its metadata.

    Args:
        path (str): path to the video file (e.g. "data/raw/fight_01.mp4")

    Returns:
        cap  (cv2.VideoCapture): frame reader object
        fps  (float):            frames per second
        size (tuple):            (width, height) in pixels

    Raises:
        RuntimeError: if the file cannot be opened
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[fight-vision] Video loaded")
    print(f"  Path:       {path}")
    print(f"  FPS:        {fps:.2f}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Start frame: {config.START_FRAME}")

    return cap, fps, (width, height)


# ── ROI EXTRACTION ────────────────────────────────────────────

def get_octagon_roi(frame):
    """
    Crop the frame to the octagon (fighting area) region of interest.

    Why: broadcast frames include crowd, graphics overlays, and corner
    padding. Cropping to just the octagon reduces noise for YOLO
    detection and speeds up inference.

    Crop boundaries use percentages (not fixed pixels) so the ROI
    scales correctly across different video resolutions.

    Args:
        frame (np.ndarray): full video frame, shape (H, W, 3)

    Returns:
        roi     (np.ndarray): cropped octagon region
        roi_box (tuple):      (left, top, right, bottom) in frame coords
                               — used to draw the green ROI rectangle
    """
    h, w = frame.shape[:2]   # note: NumPy is (rows, cols) = (height, width)

    top    = int(config.ROI_TOP    * h)
    bottom = int(config.ROI_BOTTOM * h)
    left   = int(config.ROI_LEFT   * w)
    right  = int(config.ROI_RIGHT  * w)

    # NumPy array slicing: [row_start:row_end, col_start:col_end]
    roi = frame[top:bottom, left:right]

    return roi, (left, top, right, bottom)


# ── FIGHTER DETECTION ─────────────────────────────────────────

def detect_people(roi):
    """
    Run YOLOv8 person detection on the ROI.

    Args:
        roi (np.ndarray): cropped octagon region

    Returns:
        list of (x1, y1, x2, y2) tuples — one per detected person
        coordinates are in ROI space (not full frame space)
    """
    results = person_detector(
        roi,
        conf=config.DETECTION_CONF,   # ignore detections below 40% confidence
        classes=[0],                   # 0 = "person" in COCO dataset
        verbose=False,                 # suppress per-frame console output
    )

    boxes = []
    for r in results:
        for b in r.boxes:
            # b.xyxy[0] → tensor([x1, y1, x2, y2]) as floats → convert to int tuple
            boxes.append(tuple(map(int, b.xyxy[0])))

    return boxes


# ── FIGHTER IDENTITY TRACKING ─────────────────────────────────

def assign_fighters(detections):
    """
    Assign detected bounding boxes to persistent fighter identities A and B.

    Problem: YOLO returns an unordered list each frame. Without identity
    tracking, Fighter A and B could swap labels between frames, corrupting
    strike counts and stance data.

    Solution:
    - First frame: sort detections left-to-right. Leftmost = A.
    - Subsequent frames: match each detection to the nearest previous
      fighter position using Euclidean distance (centroid tracking).
    - Duplicate guard: if both fighters match the same detection
      (e.g. during a clinch), assign the remaining box to Fighter B.

    Args:
        detections (list): list of (x1, y1, x2, y2) bounding boxes

    Returns:
        FIGHTERS dict with updated bbox and center per fighter
    """
    global FIGHTERS

    if len(detections) < 2:
        # Not enough detections — keep previous assignment
        return FIGHTERS

    # Pair each bbox with its center point
    candidates = [(box, bbox_center(box)) for box in detections]

    # ── First frame: left-to-right initial assignment ──
    if FIGHTERS["A"]["bbox"] is None:
        candidates_sorted = sorted(candidates, key=lambda x: x[1][0])
        FIGHTERS["A"]["bbox"],   FIGHTERS["A"]["center"] = candidates_sorted[0]
        FIGHTERS["B"]["bbox"],   FIGHTERS["B"]["center"] = candidates_sorted[1]
        return FIGHTERS

    # ── Subsequent frames: nearest-centroid matching ──
    assigned = {}
    for label in ["A", "B"]:
        prev_center = FIGHTERS[label]["center"]

        # Compute distance from previous center to every new detection
        distances = [
            (box, center, euclidean_distance(center, prev_center))
            for box, center in candidates
        ]

        # Assign this fighter to the closest new detection
        best_match = min(distances, key=lambda x: x[2])
        assigned[label] = best_match

    # ── Duplicate guard: both assigned same box ──
    if assigned["A"][0] == assigned["B"][0]:
        # Give B the box that A didn't take
        other = [c for c in candidates if c[0] != assigned["A"][0]]
        if other:
            assigned["B"] = (*other[0], 0)  # (box, center, dummy_dist)

    # Update global state
    for label in ["A", "B"]:
        FIGHTERS[label]["bbox"]   = assigned[label][0]
        FIGHTERS[label]["center"] = assigned[label][1]

    return FIGHTERS


# ── POSE ESTIMATION ───────────────────────────────────────────

def estimate_pose(roi, bbox):
    """
    Run YOLOv8-Pose on a single fighter's cropped bounding box.

    Why crop first: the pose model is designed for single-person input.
    Passing both fighters at once produces ambiguous multi-person results.
    Cropping to one fighter's bbox gives clean, accurate keypoints.

    Args:
        roi  (np.ndarray): octagon ROI image
        bbox (tuple):      (x1, y1, x2, y2) of fighter in ROI space

    Returns:
        np.ndarray of shape (17, 2) — (x, y) per keypoint in crop space
        None if pose estimation fails or bbox is invalid
    """
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    crop = roi[y1:y2, x1:x2]   # crop to fighter only

    if crop.size == 0:
        # Invalid bbox produced an empty crop — skip this frame
        return None

    results = pose_model(crop, conf=config.POSE_CONF, verbose=False)

    for r in results:
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            # .cpu() — move tensor from GPU/device to CPU
            # .numpy() — convert PyTorch tensor to NumPy array
            # [0] — take the first (and only) person's keypoints
            return r.keypoints.xy.cpu().numpy()[0]   # shape: (17, 2)

    return None


# ── STANCE DETECTION ──────────────────────────────────────────

def estimate_stance(pose, opp_bbox):
    """
    Determine a fighter's stance from their ankle positions.

    Logic: the lead foot is the one closest to the opponent.
    - Left ankle closer to opponent → left foot is lead → ORTHODOX
    - Right ankle closer to opponent → right foot is lead → SOUTHPAW

    COCO keypoint indices:
        15 = left ankle
        16 = right ankle

    Args:
        pose:     (17, 2) keypoint array
        opp_bbox: (x1, y1, x2, y2) opponent bounding box

    Returns:
        "ORTHODOX", "SOUTHPAW", or None if inputs missing
    """
    if pose is None or opp_bbox is None:
        return None

    left_ankle  = pose[15]
    right_ankle = pose[16]
    opp_cx      = (opp_bbox[0] + opp_bbox[2]) / 2.0  # opponent center x

    dist_left  = abs(left_ankle[0]  - opp_cx)
    dist_right = abs(right_ankle[0] - opp_cx)

    return "ORTHODOX" if dist_left < dist_right else "SOUTHPAW"


def stable_stance(label, stance):
    """
    Smooth noisy per-frame stance predictions using a majority vote.

    Per-frame pose predictions can be wrong due to model noise, occlusion,
    or motion blur. Buffering the last N predictions and taking the majority
    vote gives a stable, trustworthy stance label.

    Args:
        label:  "A" or "B"
        stance: current frame's stance prediction (str or None)

    Returns:
        str — smoothed stance label
    """
    if stance:
        STANCE_BUFFER[label].append(stance)

    if not STANCE_BUFFER[label]:
        return "UNKNOWN"

    # Majority vote: whichever stance appears most in the buffer
    return max(set(STANCE_BUFFER[label]), key=STANCE_BUFFER[label].count)


# ── STRIKE VALIDATION ─────────────────────────────────────────

def is_standing(prev_pose, curr_pose):
    """
    Check if a fighter is in a standing posture between two frames.

    Uses the vertical movement of the left hip (keypoint 11).
    A large downward hip movement indicates a takedown or knockdown.
    Strikes are only counted when a fighter is standing.

    COCO keypoint 11 = left hip

    Args:
        prev_pose: (17, 2) pose from previous frame
        curr_pose: (17, 2) pose from current frame

    Returns:
        bool — True if fighter appears to be standing
    """
    hip_movement = abs(curr_pose[11][1] - prev_pose[11][1])
    return hip_movement < config.HIP_DROP_THRESH


def valid_strike(prev_pose, curr_pose, opp_bbox):
    """
    Determine whether a valid strike occurred between two frames.

    Checks both wrists (left=9, right=10) against 3 conditions:

    Condition 1 — Velocity:
        Wrist must move at least WRIST_SPEED_THRESH pixels between frames.
        Filters out passive guard movements and slow arm repositioning.

    Condition 2 — Direction:
        Wrist must be getting CLOSER to the opponent's center.
        If dist(curr_wrist, opp_center) >= dist(prev_wrist, opp_center)
        the wrist is moving away or laterally — not a strike.

    Condition 3 — Proximity (impact zone):
        Wrist must enter the opponent's bounding box.
        Ensures the punch actually reached the opponent, not just a swing.

    All 3 must pass for the same wrist → True (valid strike).
    If neither wrist passes → False.

    Args:
        prev_pose: (17, 2) pose from previous frame
        curr_pose: (17, 2) pose from current frame
        opp_bbox:  (x1, y1, x2, y2) opponent bounding box

    Returns:
        bool
    """
    opp_center = bbox_center(opp_bbox)

    for wrist_idx in [9, 10]:   # 9 = left wrist, 10 = right wrist
        prev_wrist = prev_pose[wrist_idx]
        curr_wrist = curr_pose[wrist_idx]

        # ── Condition 1: velocity ──
        speed = euclidean_distance(prev_wrist, curr_wrist)
        if speed < config.WRIST_SPEED_THRESH:
            continue   # too slow — skip this wrist

        # ── Condition 2: direction (must be moving toward opponent) ──
        dist_now  = euclidean_distance(curr_wrist, opp_center)
        dist_prev = euclidean_distance(prev_wrist, opp_center)
        if dist_now >= dist_prev:
            continue   # not getting closer — skip this wrist

        # ── Condition 3: proximity (wrist inside opponent bbox) ──
        if point_in_bbox(curr_wrist, opp_bbox):
            return True   # all 3 conditions passed → valid strike

    return False   # neither wrist passed all 3 conditions


# ── MAIN PROCESSING LOOP ──────────────────────────────────────

def read_frames(cap, fps, size, output_path=None):
    """
    Process a fight video frame by frame and produce:
    - An annotated output video with bounding boxes, pose, and HUD
    - A momentum graph (PNG) showing strikes per 10-second bucket

    Args:
        cap:         cv2.VideoCapture object (from load_video)
        fps:         float — video frame rate
        size:        (width, height) tuple
        output_path: str — path for output video (uses config default if None)
    """
    reset_state()
    os.makedirs("outputs", exist_ok=True)

    if output_path is None:
        output_path = config.OUTPUT_VIDEO

    # VideoWriter encodes and saves the annotated output video
    # fourcc = four-character code identifying the video codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # MPEG-4 codec → .mp4
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    # Optionally skip to a specific start frame
    if config.START_FRAME > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, config.START_FRAME)
        print(f"[fight-vision] Skipping to frame {config.START_FRAME} ({config.START_FRAME/fps:.1f}s)")

    frame_id = config.START_FRAME
    print(f"[fight-vision] Processing started...")

    while True:
        # ── Read next frame ──
        ret, frame = cap.read()
        if not ret:
            break   # video ended or read failed

        # Current time in seconds (used for cooldown and bucketing)
        t = frame_id / fps

        # ── Step 1: Crop octagon ROI ──
        roi, roi_box = get_octagon_roi(frame)
        cv2.rectangle(frame, roi_box[:2], roi_box[2:], (0, 255, 0), 2)

        # ── Step 2: Detect fighters ──
        detections = detect_people(roi)
        fighters   = assign_fighters(detections)

        # ── Step 3: Estimate pose for each fighter ──
        poseA = estimate_pose(roi, fighters["A"]["bbox"])
        poseB = estimate_pose(roi, fighters["B"]["bbox"])

        # ── Step 4: Determine stance (smoothed) ──
        stanceA = stable_stance("A", estimate_stance(poseA, fighters["B"]["bbox"]))
        stanceB = stable_stance("B", estimate_stance(poseB, fighters["A"]["bbox"]))

        # ── Step 5: Strike detection for both fighters ──
        # Iterating over both fighters avoids duplicating the if-block
        for label, curr_pose, opp_bbox in [
            ("A", poseA, fighters["B"]["bbox"]),
            ("B", poseB, fighters["A"]["bbox"]),
        ]:
            prev_pose = PREV_POSE[label]

            # Need both current and previous pose to compute velocity
            if curr_pose is None or prev_pose is None or opp_bbox is None:
                continue

            # Posture check: fighter must be standing
            if not is_standing(prev_pose, curr_pose):
                continue

            # Strike geometry check
            if valid_strike(prev_pose, curr_pose, opp_bbox):
                last_t = LAST_STRIKE_TIME[label]

                # Cooldown check: minimum time between two strikes
                if last_t is None or (t - last_t) > config.STRIKE_COOLDOWN:
                    STRIKE_COUNT[label]     += 1
                    LAST_STRIKE_TIME[label]  = t

                    # Assign to momentum bucket (floor division)
                    # e.g. t=23.5s → bucket = 10 * int(2.35) = 20
                    bucket = config.BUCKET_SIZE * int(t // config.BUCKET_SIZE)
                    STRIKE_BUCKETS[label][bucket] = STRIKE_BUCKETS[label].get(bucket, 0) + 1

        # ── Step 6: Update pose memory for next frame ──
        # CRITICAL: must happen after strike detection, before next frame
        PREV_POSE["A"] = poseA
        PREV_POSE["B"] = poseB

        # ── Step 7: Draw fighter overlays ──
        ox, oy = roi_box[0], roi_box[1]   # ROI origin offset in frame

        if fighters["A"]["bbox"]:
            x1, y1, x2, y2 = fighters["A"]["bbox"]
            cv2.rectangle(frame, (ox+x1, oy+y1), (ox+x2, oy+y2), (255, 0, 0), 2)
            if poseA is not None:
                for kx, ky in poseA:
                    cv2.circle(frame, (ox+x1+int(kx), oy+y1+int(ky)), 3, (0, 0, 255), -1)

        if fighters["B"]["bbox"]:
            x1, y1, x2, y2 = fighters["B"]["bbox"]
            cv2.rectangle(frame, (ox+x1, oy+y1), (ox+x2, oy+y2), (0, 255, 255), 2)
            if poseB is not None:
                for kx, ky in poseB:
                    cv2.circle(frame, (ox+x1+int(kx), oy+y1+int(ky)), 3, (255, 0, 255), -1)

        # ── Step 8: Draw HUD ──
        draw_hud(frame, STRIKE_COUNT["A"], STRIKE_COUNT["B"], stanceA, stanceB, t)

        # ── Step 9: Write annotated frame to output video ──
        writer.write(frame)
        frame_id += 1

        # Progress log every 300 frames (~10s at 30fps)
        if frame_id % 300 == 0:
            print(f"  Frame {frame_id} | t={t:.1f}s | Strikes A:{STRIKE_COUNT['A']} B:{STRIKE_COUNT['B']}")

    # ── Cleanup and finalize ──
    writer.release()
    cap.release()
    print(f"\n[fight-vision] Done. Output saved to: {output_path}")
    print(f"  Final strikes → Fighter A: {STRIKE_COUNT['A']}  Fighter B: {STRIKE_COUNT['B']}")

    generate_momentum_graph()


# ── MOMENTUM GRAPH ────────────────────────────────────────────

def generate_momentum_graph(output_path=None):
    """
    Plot and save a line graph showing strike momentum over time.

    X-axis: time in seconds (grouped into BUCKET_SIZE-second windows)
    Y-axis: number of strikes per window

    Gaps in the timeline (no strikes in a bucket) are filled with 0
    so both lines cover the same x-axis range.

    Args:
        output_path: str — where to save the PNG (uses config default if None)
    """
    if output_path is None:
        output_path = config.OUTPUT_GRAPH

    # Union of all time buckets from both fighters
    all_times = sorted(set(STRIKE_BUCKETS["A"]) | set(STRIKE_BUCKETS["B"]))

    if not all_times:
        print("[fight-vision] No strikes detected — skipping momentum graph.")
        return

    counts_a = [STRIKE_BUCKETS["A"].get(t, 0) for t in all_times]
    counts_b = [STRIKE_BUCKETS["B"].get(t, 0) for t in all_times]

    plt.figure(figsize=(12, 5))
    plt.plot(all_times, counts_a, label="Fighter A", color="#3B82F6", linewidth=2, marker="o", markersize=4)
    plt.plot(all_times, counts_b, label="Fighter B", color="#EF4444", linewidth=2, marker="o", markersize=4)

    plt.fill_between(all_times, counts_a, alpha=0.15, color="#3B82F6")
    plt.fill_between(all_times, counts_b, alpha=0.15, color="#EF4444")

    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel(f"Strikes per {config.BUCKET_SIZE}s window", fontsize=12)
    plt.title("Fight Momentum — Strike Frequency Over Time", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[fight-vision] Momentum graph saved to: {output_path}")
