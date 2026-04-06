# =============================================================
# fight-vision | Dockerfile
#
# Builds a container that can run the fight-vision pipeline.
#
# Build:  docker build -t fight-vision .
# Run:    docker run -v $(pwd)/data:/app/data \
#                    -v $(pwd)/outputs:/app/outputs \
#                    fight-vision
# =============================================================

# Base image: Python 3.10 slim (smaller than full python image)
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# ── System dependencies ───────────────────────────────────────
# OpenCV needs these system libraries to work inside a container.
# libglib2.0-0    → required by OpenCV internals
# libgl1          → required for cv2 video codec support
# libgomp1        → OpenMP — used by YOLOv8 for parallel inference
# --no-install-recommends → keeps image small
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────
# Copy requirements first (Docker layer caching: if requirements
# don't change, this layer is reused — much faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project source ───────────────────────────────────────
COPY . .

# ── Create required directories ───────────────────────────────
RUN mkdir -p data/raw outputs

# ── Environment ───────────────────────────────────────────────
# Fix OpenMP conflict (same as main.py)
ENV KMP_DUPLICATE_LIB_OK=TRUE

# ── Default command ───────────────────────────────────────────
# Runs the pipeline. Video must be mounted at data/raw/fight_01.mp4
CMD ["python", "main.py"]
