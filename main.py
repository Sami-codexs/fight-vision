import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.video.loader import load_video, read_frames

VIDEO_PATH = "data/raw/fight_01.mp4"

cap, fps, size = load_video(VIDEO_PATH)
read_frames(cap, fps, size, "outputs/final_fight_analysis.mp4")
