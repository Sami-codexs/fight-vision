

import cv2
from pathlib import Path


def extract_frames(video_path, output_folder):
    video_path = Path(video_path)
    output_folder = Path(output_folder)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_number / fps

        frame_filename = output_folder / f"frame_{frame_number:06d}.jpg"
        cv2.imwrite(str(frame_filename), frame)

        frame_number += 1

    cap.release()

    print(f"Extracted {frame_number} frames to {output_folder}")