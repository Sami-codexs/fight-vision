# =============================================================
# fight-vision | src/video/frame_extractor.py
#
# Utility: extract individual frames from a video file and save
# them as numbered .jpg images.
#
# Use case: debugging the pipeline by inspecting specific frames,
# or building a labeled dataset for future model training.
# =============================================================

import cv2
from pathlib import Path


def extract_frames(video_path, output_folder, every_n_frames=1):
    """
    Extract frames from a video file and save as JPEG images.

    Filenames are zero-padded for correct alphabetical sorting:
        frame_000001_0.03s.jpg
        frame_000002_0.07s.jpg
        ...

    Args:
        video_path     (str): path to the input video file
        output_folder  (str): directory where frames will be saved
        every_n_frames (int): save every Nth frame (default=1 = every frame)
                              use every_n_frames=30 to save ~1 frame/second at 30fps

    Returns:
        int: total number of frames saved

    Raises:
        FileNotFoundError: if the video file doesn't exist
        RuntimeError:      if the video cannot be opened by OpenCV

    Example:
        >>> extract_frames("data/raw/fight_01.mp4", "data/frames", every_n_frames=30)
        Extracted 450 frames to data/frames
    """
    video_path    = Path(video_path)
    output_folder = Path(output_folder)

    # Validate input
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Create output directory (and any parent directories) if needed
    # exist_ok=True → no crash if folder already exists
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    saved_count  = 0

    print(f"[frame_extractor] Starting extraction")
    print(f"  Source:   {video_path}")
    print(f"  Output:   {output_folder}")
    print(f"  FPS:      {fps:.2f}")
    print(f"  Total frames in video: {total_frames}")
    print(f"  Saving every {every_n_frames} frame(s)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break   # end of video or read error

        if frame_number % every_n_frames == 0:
            # Compute timestamp for this frame
            timestamp = frame_number / fps   # seconds

            # Zero-padded filename includes frame number AND timestamp
            # :06d → 6-digit zero-padded integer (sorts correctly in file browsers)
            # e.g. frame_000150_5.00s.jpg
            filename = output_folder / f"frame_{frame_number:06d}_{timestamp:.2f}s.jpg"
            cv2.imwrite(str(filename), frame)
            saved_count += 1

        frame_number += 1

    cap.release()

    print(f"[frame_extractor] Done. Saved {saved_count} frames to {output_folder}")
    return saved_count
