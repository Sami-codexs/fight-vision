# 🥊 fight-vision

> Computer vision pipeline for combat sports analytics — real-time fighter detection, pose estimation, strike validation, and momentum tracking using YOLOv8.

![CI](https://github.com/Sami-codexs/fight-vision/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## What it does

fight-vision takes raw fight footage and converts it into structured performance metrics — automatically, frame by frame.

| Input | Output |
|-------|--------|
| Raw fight video (.mp4) | Annotated video with fighter bounding boxes, pose keypoints, stance labels, and live strike counter |
| | Momentum graph (PNG) — strike frequency per fighter over time |
| | Console summary of total validated strikes |

---

## Pipeline

```
Video → ROI Crop → YOLOv8 Detection → Centroid Tracking
     → YOLOv8-Pose → Strike Validation (5 constraints) → Time Bucketing → Momentum Graph
```

Each frame goes through 9 steps:

1. **ROI extraction** — crops the octagon area using percentage-based coordinates (resolution-independent)
2. **Person detection** — YOLOv8n detects fighters, filters by class=0 (person) and conf ≥ 0.4
3. **Identity tracking** — centroid-based assignment maintains consistent Fighter A/B labels across frames
4. **Pose estimation** — YOLOv8n-pose returns 17 COCO keypoints per fighter from their individual crop
5. **Stance detection** — compares ankle-to-opponent distances; majority vote over 15 frames for stability
6. **Standing check** — hip vertical displacement < threshold filters out grappling/takedown frames
7. **Strike validation** — 3-condition gate: velocity + direction + proximity (see below)
8. **Cooldown enforcement** — 0.8s minimum between strikes prevents double-counting
9. **Time bucketing** — strikes grouped into 10s windows → momentum graph

---

## Strike Validation Logic

A strike is counted only when **all 3 conditions pass** for either wrist (keypoints 9/10):

| # | Condition | How it works |
|---|-----------|-------------|
| 1 | **Velocity** | Wrist must travel ≥ 18px between consecutive frames |
| 2 | **Direction** | `dist(curr_wrist, opp_center) < dist(prev_wrist, opp_center)` — wrist must be getting closer to opponent |
| 3 | **Proximity** | Wrist must enter opponent's bounding box (impact zone) |

This multi-constraint approach significantly reduces false positives from guard adjustments, feints, and model noise.

---

## Output

### Annotated Video
- 🔵 Blue box + red dots — Fighter A bounding box and pose keypoints
- 🟡 Cyan box + magenta dots — Fighter B
- 🟢 Green box — octagon ROI boundary
- HUD overlay with stance label and live strike count per fighter

### Momentum Graph
Strike frequency per 10-second interval — shows dominance shifts and fight momentum over time.

> 📌 Add your `outputs/momentum_graph.png` here after running the pipeline.

---

## Tech Stack

| Library | Version | Role |
|---------|---------|------|
| `ultralytics` | 8.1.47 | YOLOv8 detection + pose estimation |
| `opencv-python` | 4.9.0 | Video I/O, frame processing, annotation |
| `numpy` | 1.26.4 | Array math — distance, velocity, keypoint ops |
| `matplotlib` | 3.8.4 | Momentum graph generation |

---

## Project Structure

```
fight-vision/
├── main.py                    # entry point
├── config.py                  # all tunable parameters (thresholds, paths)
├── requirements.txt
├── src/
│   └── video/
│       ├── loader.py          # full pipeline — detection, pose, strike logic, output
│       ├── frame_extractor.py # utility — save individual frames as JPEGs
│       └── utils.py           # pure helper functions — geometry, drawing
├── data/
│   └── raw/                   # place your fight video here (gitignored)
├── outputs/                   # generated videos and graphs (gitignored)
└── demo/                      # sample output clips
```

---

## Setup

```bash
git clone https://github.com/Sami-codexs/fight-vision.git
cd fight-vision

pip install -r requirements.txt
```

Download model weights from [Ultralytics](https://docs.ultralytics.com/models/yolov8/):
```bash
# place both files in the project root
yolov8n.pt
yolov8n-pose.pt
```

Place your fight video:
```
data/raw/fight_01.mp4
```

Run:
```bash
python main.py
```

All parameters (thresholds, ROI ratios, bucket size, start frame) are in `config.py`.

---

## Configuration

All tuning knobs are in `config.py` — no hardcoded values in logic files:

```python
WRIST_SPEED_THRESH = 18    # min pixels/frame for strike velocity
HIP_DROP_THRESH    = 20    # max hip drop to confirm standing posture
STRIKE_COOLDOWN    = 0.8   # seconds between valid strikes per fighter
STANCE_HISTORY     = 15    # frames for stance majority vote
BUCKET_SIZE        = 10    # seconds per momentum graph window
START_FRAME        = 0     # skip to this frame before processing
```

---

## Limitations

- CPU inference is slow (~3-5 FPS on standard hardware). GPU recommended for real-time use.
- Rule-based strike detection — not a learned classifier. Accuracy depends on tuned thresholds.
- No advanced re-identification (DeepSORT/ByteTrack) — identity swaps can occur during clinches.
- Pose model accuracy degrades under heavy occlusion or extreme camera angles.
- No model fine-tuning on combat-specific footage.

---

## Future Improvements

- [ ] GPU acceleration (CUDA) for real-time inference
- [ ] FastAPI endpoint — POST video → receive JSON strike data
- [ ] DeepSORT tracking for robust fighter re-identification
- [ ] Strike type classification (jab, cross, hook, kick) using sequence model
- [ ] Hugging Face Spaces demo
- [ ] Fine-tuning YOLOv8 on labeled combat sports dataset

---

## Notes

This project focuses on **system design**, **temporal state management**, and **structured analytics extraction from unstructured video data**.

It is not intended for official judging or real-time scoring systems.

---

*Built with Python, OpenCV, and Ultralytics YOLOv8*
