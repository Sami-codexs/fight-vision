# fight-vision — Deployment Guide

Complete step-by-step instructions to push to GitHub, Dockerize, and set up CI/CD.
Follow in order. Do not skip steps.

---

## PHASE 1 — Push to GitHub

### Step 1: Extract the zip

Unzip `fight-vision-upgraded.zip` on your machine.
You should see this structure:

```
fight-vision/
├── main.py
├── config.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── __init__.py
│   └── video/
│       ├── __init__.py
│       ├── loader.py
│       ├── utils.py
│       └── frame_extractor.py
├── data/
│   └── raw/         ← put your video here
├── outputs/         ← pipeline writes here
└── demo/            ← put your demo clip here
```

---

### Step 2: Open terminal inside the folder

```bash
cd fight-vision
```

---

### Step 3: Initialize git (if starting fresh)

If you're replacing the old repo entirely:

```bash
# Delete the old remote link (you'll reconnect below)
git init
git remote add origin https://github.com/Sami-codexs/fight-vision.git
```

If you're updating the existing repo:

```bash
git remote -v   # confirm origin is already set
```

---

### Step 4: Stage all files

```bash
git add .
git status    # check what's being added — should see all your files
```

---

### Step 5: Commit — use meaningful messages, not "update"

```bash
git commit -m "refactor: restructure project with config, utils, and modular src layout"
```

---

### Step 6: Push to GitHub

```bash
git push -u origin main
```

If you get a "rejected" error (because old repo has different history):

```bash
git push -u origin main --force
```

> ⚠️ --force overwrites the remote. Only use this because you're replacing the old repo intentionally.

---

### Step 7: Verify on GitHub

Open https://github.com/Sami-codexs/fight-vision

Check:
- [ ] All files visible (src/video/loader.py should be readable)
- [ ] README renders correctly with the pipeline table
- [ ] Actions tab shows the CI workflow running

---

## PHASE 2 — Add the momentum graph to README

This is the single highest-impact thing you can do.

### Step 1: Run the pipeline locally

```bash
# Put your video at:
data/raw/fight_01.mp4

# Run
python main.py
```

### Step 2: Find the output

```
outputs/momentum_graph.png
```

### Step 3: Add it to the repo

```bash
# Create an assets folder for images
mkdir -p assets
cp outputs/momentum_graph.png assets/momentum_graph.png

git add assets/momentum_graph.png
git commit -m "docs: add momentum graph output screenshot"
git push
```

### Step 4: Add to README.md

Find this line in README.md:

```
> 📌 Add your `outputs/momentum_graph.png` here after running the pipeline.
```

Replace it with:

```markdown
![Momentum Graph](assets/momentum_graph.png)
```

Then push again:

```bash
git add README.md
git commit -m "docs: embed momentum graph in README"
git push
```

---

## PHASE 3 — Docker

Docker packages your entire project — code, dependencies, environment — into one container that runs identically on any machine.

### Step 1: Install Docker Desktop

Download from: https://www.docker.com/products/docker-desktop/
Install it, open it, wait for the whale icon in your taskbar.

Verify:
```bash
docker --version
# Should print: Docker version 24.x.x or similar
```

---

### Step 2: Build the Docker image

```bash
# Run from inside the fight-vision/ folder
docker build -t fight-vision .
```

What this does:
- Reads your `Dockerfile`
- Downloads Python 3.10-slim base image
- Installs system libraries (libglib, libgl)
- Installs your pip requirements
- Copies your source code into the container

Expected output (last few lines):
```
Successfully built abc123def456
Successfully tagged fight-vision:latest
```

---

### Step 3: Run the container

```bash
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  fight-vision
```

What the flags mean:
- `-v $(pwd)/data:/app/data` — mounts your local `data/` folder into the container at `/app/data`
  - This is how the container sees your video file
- `-v $(pwd)/outputs:/app/outputs` — mounts your local `outputs/` folder
  - This is how the container writes results back to your machine

> On Windows PowerShell, replace `$(pwd)` with `${PWD}`:
> ```
> docker run -v ${PWD}/data:/app/data -v ${PWD}/outputs:/app/outputs fight-vision
> ```

---

### Step 4: Verify it worked

```bash
ls outputs/
# Should show: final_fight_analysis.mp4  momentum_graph.png
```

---

### Step 5: Useful Docker commands

```bash
# List all images
docker images

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Remove the image
docker rmi fight-vision

# Remove stopped containers
docker container prune

# See logs from a container
docker logs <container_id>
```

---

## PHASE 4 — CI/CD (GitHub Actions)

CI/CD stands for Continuous Integration / Continuous Deployment.
Your `.github/workflows/ci.yml` already does CI — it runs automatically.

### How it works

Every time you `git push` to main:

1. GitHub spins up a fresh Ubuntu machine in the cloud
2. Checks out your code
3. Runs syntax check on all Python files
4. Verifies all required files exist
5. Validates config values are in safe ranges
6. Builds your Docker image

If any step fails → GitHub marks the commit as ❌ and emails you.
If all pass → GitHub marks the commit as ✅.

---

### Step 1: Confirm it's running

After your first push:

1. Go to https://github.com/Sami-codexs/fight-vision
2. Click the **Actions** tab
3. You should see a workflow run called "CI — fight-vision"
4. Click it to see each step's output

---

### Step 2: Add the CI badge to README

Once CI is working, get your badge URL:

```
https://github.com/Sami-codexs/fight-vision/actions/workflows/ci.yml/badge.svg
```

The README already has this badge line — it will show green automatically once CI passes:

```markdown
![CI](https://github.com/Sami-codexs/fight-vision/actions/workflows/ci.yml/badge.svg)
```

Add this to the badges section at the top of README.md.

---

## PHASE 5 — Commit hygiene (important for resume)

Recruiters look at your commit history. 5 commits that say "update" look bad.
Meaningful commits look like a real engineer.

### Good commit message format:

```
type: short description

feat:     new feature added
fix:      bug fixed
refactor: code restructured without behavior change
docs:     README or comments updated
config:   config values changed
ci:       CI/CD pipeline changes
```

### Commit sequence to aim for (do these as you work):

```bash
git commit -m "refactor: restructure project with config, utils, modular src layout"
git commit -m "feat: add Dockerfile and .dockerignore for containerized deployment"
git commit -m "ci: add GitHub Actions workflow for lint, validation, and Docker build"
git commit -m "docs: rewrite README with pipeline breakdown and config reference"
git commit -m "docs: add momentum graph output to README"
git commit -m "fix: embed timestamp in frame_extractor output filenames"
git commit -m "refactor: extract geometry helpers into utils.py"
```

This gives you 7+ meaningful commits immediately.

---

## PHASE 6 — Final checklist before applying

Go through this before you send any application:

- [ ] All source files visible on GitHub (src/video/loader.py readable)
- [ ] README has the pipeline table, tech stack table, and project structure
- [ ] `momentum_graph.png` embedded in README
- [ ] CI badge shows green (✅) in README
- [ ] Dockerfile present and builds successfully
- [ ] At least 10 meaningful commits in history
- [ ] No broken links in README
- [ ] Demo video in `demo/` folder and linked from README

When all boxes are checked — start applying. Not before.
