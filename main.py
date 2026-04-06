# =============================================================
# fight-vision | main.py
#
# Entry point. Run this to process a fight video.
#
# Usage:
#   python main.py
#
# Configure video path and all parameters in config.py
# =============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # fixes OpenMP conflict on some systems

import config
from src.video.loader import load_video, read_frames

VIDEO_PATH = "data/raw/fight_01.mp4"   # ← put your video here

if __name__ == "__main__":
    print("=" * 50)
    print("  fight-vision | fight analytics pipeline")
    print("=" * 50)

    cap, fps, size = load_video(VIDEO_PATH)
    read_frames(cap, fps, size, output_path=config.OUTPUT_VIDEO)
