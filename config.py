# =============================================================
# fight-vision | config.py
# All tunable parameters live here.
# Change values here — never hardcode them inside logic files.
# =============================================================

# ── Model weights ─────────────────────────────────────────────
DETECTION_MODEL  = "yolov8n.pt"       # person bounding box model
POSE_MODEL       = "yolov8n-pose.pt"  # 17-keypoint pose model

# ── Detection thresholds ──────────────────────────────────────
DETECTION_CONF   = 0.4   # min confidence to accept a person detection (0-1)
POSE_CONF        = 0.3   # min confidence to accept a pose keypoint    (0-1)

# ── ROI (Region of Interest) — octagon crop ratios ───────────
# These define the percentage of the frame to keep.
# Tune these if your video has a different broadcast layout.
ROI_TOP    = 0.28   # start 28% from top    (cuts broadcast banner)
ROI_BOTTOM = 0.82   # end   82% from top    (cuts bottom crowd)
ROI_LEFT   = 0.18   # start 18% from left   (cuts side padding)
ROI_RIGHT  = 0.82   # end   82% from left   (cuts side padding)

# ── Strike validation ─────────────────────────────────────────
WRIST_SPEED_THRESH = 18    # min pixels/frame wrist must travel  → velocity check
HIP_DROP_THRESH    = 20    # max pixel drop of hip between frames → standing check
STRIKE_COOLDOWN    = 0.8   # seconds between valid strikes per fighter → cooldown

# ── Stance smoothing ──────────────────────────────────────────
STANCE_HISTORY = 15   # number of recent frames used for majority-vote stance

# ── Momentum graph ────────────────────────────────────────────
BUCKET_SIZE = 10   # seconds per time bucket in the momentum graph

# ── Video processing ──────────────────────────────────────────
START_FRAME = 0   # frame to start processing from (0 = beginning)
                  # set to e.g. 9000 to skip to 5 min mark during dev

# ── Output paths ─────────────────────────────────────────────
OUTPUT_VIDEO = "outputs/final_fight_analysis.mp4"
OUTPUT_GRAPH = "outputs/momentum_graph.png"
