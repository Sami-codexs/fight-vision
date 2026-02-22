<img width="593" height="518" alt="image" src="https://github.com/user-attachments/assets/737d4412-7152-41e6-8eab-2290d7091409" /># 🥊 Fight Vision

Computer vision pipeline for fight analytics using YOLOv8-based detection, pose estimation, custom strike validation logic, and temporal momentum tracking.

## 📌 Overview

Fight Vision analyzes fight footage to detect fighters, estimate pose keypoints, validate potential strikes using motion constraints, and generate a momentum graph over time.

The system combines:
- **YOLOv8** person detection
- **YOLOv8** pose estimation
- **Frame-based temporal tracking**
- **Custom strike validation logic**
- **Momentum aggregation** over fixed time buckets

The goal is to transform raw fight footage into structured performance metrics.

## 🎯 Problem Statement

Fight videos contain rich motion information, but lack structured quantitative insights such as:

- Number of validated strikes
- Temporal dominance shifts
- Momentum progression over time

This project attempts to extract structured performance metrics from unstructured fight footage using computer vision and rule-based validation logic.

## 🏗 Pipeline Architecture

The system processes video frame-by-frame using the following pipeline:

1. Load fight video
2. Extract octagon Region of Interest (ROI)
3. Detect fighters using YOLOv8
4. Assign fighter identity across frames
5. Estimate pose keypoints (YOLOv8-Pose)
6. Apply strike validation logic
7. Enforce cooldown and temporal constraints
8. Aggregate strikes into time buckets
9. Generate momentum graph
10. Export annotated output video

## 🧠 Strike Validation Logic

A strike is counted only when all conditions are satisfied:

| Condition | Description |
|-----------|-------------|
| **Velocity Threshold** | Wrist velocity exceeds threshold |
| **Direction Constraint** | Wrist moves closer to opponent center |
| **Proximity Check** | Wrist enters opponent bounding box |
| **Posture Validation** | Fighter remains in standing posture |
| **Cooldown Window** | Temporal spacing between strikes is respected |

This reduces false positives from random movement and ensures more reliable detection.

## ⏱ Temporal Momentum Tracking

Strikes are grouped into fixed time buckets (**10-second intervals**).

This allows visualization of:
- Strike frequency trends
- Momentum shifts
- Fighter dominance over time

The final output includes a momentum graph showing strike counts per time interval.




## ⚙️ Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Matplotlib

## ▶️ How To Run

## 1️⃣ Install Dependencies


**pip install -r requirements.txt**

2️⃣ Download Model Weights
Download the following from Ultralytics:
**yolov8n.pt**
**yolov8n-pose.pt**
Place them in the project root directory.
3️⃣ Place Video
Add fight footage inside:

**data/raw/**


**python main.py**

The system will:
Process the video
Generate annotated output
Create a momentum graph


## 📊 Output
| Output              | Description                                  |
| ------------------- | -------------------------------------------- |
| **Annotated Video** | Visual overlay of detections and strikes     |
| **Momentum Graph**  | Strikes per 10-second interval visualization |

💡 Tip: Add your generated momentum_graph.png here for visual demonstration in your repo.


# ⚠️ Limitations
CPU inference is computationally expensive
No advanced fighter re-identification
Pose-based logic depends on model accuracy
No model fine-tuning performed
Rule-based strike detection (not learned classification)
# 🚀 Future Improvements :
- GPU acceleration
- Advanced tracking & re-identification
- Strike type classification (jab, cross, hook, etc.)
- Real-time inference pipeline
- Web-based visualization dashboard
- Model fine-tuning on combat datasets
# 📌 Notes
This project focuses on system design, temporal state management, and structured analytics from unstructured video data.


# It is not intended for official judging or real-time scoring systems.











