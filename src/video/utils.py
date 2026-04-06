# =============================================================
# fight-vision | src/video/utils.py
# Pure helper functions — no side effects, no global state.
# Every function here takes inputs and returns outputs only.
# This makes them easy to test independently.
# =============================================================

import numpy as np
import cv2


# ── Geometry helpers ──────────────────────────────────────────

def euclidean_distance(point_a, point_b):
    """
    Straight-line pixel distance between two 2D points.

    Uses Pythagoras: sqrt((x2-x1)^2 + (y2-y1)^2)

    Args:
        point_a: (x, y) tuple or array
        point_b: (x, y) tuple or array

    Returns:
        float — distance in pixels
    """
    return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))


def bbox_center(bbox):
    """
    Compute the center (x, y) of a bounding box.

    Args:
        bbox: (x1, y1, x2, y2) tuple

    Returns:
        (cx, cy) — center as floats
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_bbox(point, bbox):
    """
    Check if a 2D point lies inside a bounding box.

    Used for the proximity / impact-zone check in strike validation:
    is the attacker's wrist inside the opponent's bounding box?

    Args:
        point: (x, y)
        bbox:  (x1, y1, x2, y2)

    Returns:
        bool
    """
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 < x < x2 and y1 < y < y2


# ── Drawing helpers ───────────────────────────────────────────

def draw_fighter_overlay(frame, roi_offset, bbox, pose, color, label, stance, strike_count):
    """
    Draw bounding box, pose keypoints, and HUD label for one fighter.

    Args:
        frame:        full video frame (NumPy array) — drawn on directly
        roi_offset:   (left, top) pixel offset of ROI inside full frame
        bbox:         (x1, y1, x2, y2) in ROI coordinates
        pose:         (17, 2) NumPy array of keypoint (x, y) coordinates, or None
        color:        BGR tuple for this fighter (e.g. (255, 0, 0) for blue)
        label:        "A" or "B"
        stance:       "ORTHODOX", "SOUTHPAW", or "UNKNOWN"
        strike_count: int — current validated strike count
    """
    ox, oy = roi_offset  # ROI origin in frame coordinates

    if bbox is not None:
        # Translate bbox from ROI coords → full frame coords
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (ox + x1, oy + y1), (ox + x2, oy + y2), color, 2)

    if pose is not None and bbox is not None:
        x1, y1, _, _ = bbox
        for kx, ky in pose:
            # Translate keypoint from crop coords → full frame coords
            cv2.circle(frame, (ox + x1 + int(kx), oy + y1 + int(ky)), 3, color, -1)

    return frame


def draw_hud(frame, strike_count_a, strike_count_b, stance_a, stance_b, timestamp):
    """
    Draw heads-up display (HUD) showing fighter stats on the frame.

    Args:
        frame:          full video frame
        strike_count_a: int
        strike_count_b: int
        stance_a:       str
        stance_b:       str
        timestamp:      float — current time in seconds
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Fighter A — top left
    cv2.putText(frame,
                f"A | {stance_a} | Strikes: {strike_count_a}",
                (20, 50), font, 0.9, (255, 0, 0), 2)

    # Fighter B — top right
    text_b = f"B | {stance_b} | Strikes: {strike_count_b}"
    cv2.putText(frame, text_b,
                (w - 430, 50), font, 0.9, (0, 255, 255), 2)

    # Timestamp — bottom left
    mins = int(timestamp) // 60
    secs = int(timestamp) % 60
    cv2.putText(frame,
                f"Time: {mins:02d}:{secs:02d}",
                (20, h - 20), font, 0.7, (200, 200, 200), 1)

    return frame
