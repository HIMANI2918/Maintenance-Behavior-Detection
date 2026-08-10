# Cattle Behavior Detection System

A YOLOv8n-based system that detects and tracks cattle behaviors — **drinking, standing, eating, lying** — in video footage, and generates HTML/JSON reports summarizing behavior over time.

All core logic lives in `cattle_detection_system.py` (the `CattleBehaviorDetector` class). Two other files help you run it on an HPC/Slurm cluster:

| File | Purpose |
|---|---|
| `cattle_detection_system.py` | Core class: training, validation, and video processing |
| `automate.py` | Loops `CattleBehaviorDetector` over every video in a folder (testing/inference) |
| `Slurm_python_auto.sh` | Slurm batch script that runs `automate.py` on a GPU node |

---

## ⚙️ Lines you need to edit before running

If you're a new user cloning this repo, here's exactly what to change:

### `cattle_detection_system.py`
- **Lines 1332–1334** (inside `main()`, only used if you run this file directly, e.g. `python cattle_detection_system.py`): update `MODEL_PATH`, `INPUT_VIDEO`, `OUTPUT_VIDEO` to your own paths.
- No other edits needed — the old hardcoded report path is gone. Reports now go to `<project_dir>/results/reports` by default, or wherever you pass via `report_dir=` (see Section 2 below).

### `automate.py`
- **Line 33**: model weights path —
  `detector = CattleBehaviorDetector("cattle_behavior_project/models/cattle_behavior/weights/best.pt")`
- **Line 40**: `input_folder = "/scratch/username/Input_cows"` → your video input folder
- **Line 41**: `output_folder = "/scratch/username/cattle_behavior_project/results/videos/output/Output_cows"` → your desired output folder

### `Slurm_python_auto.sh`
- **Line 5**: `#SBATCH --mail-user=username@example.edu` → your email
- **Line 9** (`#SBATCH --partition=...`) and **line 10** (`#SBATCH --account=...`): your cluster's GPU partition/account
- **Line 17**: `source /home/username/miniconda3/etc/profile.d/conda.sh` → your conda install path
- **Line 18**: `conda activate /scratch/username/myenv` → your conda environment path
- **Line 20**: `python3 /scratch/username/automate.py` → the actual path to your copy of `automate.py`

---

## Requirements

- Python 3.9+
- `ultralytics` (YOLOv8n)
- `torch` (with CUDA if using GPU)
- `opencv-python`
- `supervision`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `pyyaml`

```bash
pip install ultralytics torch opencv-python supervision pandas numpy matplotlib seaborn scikit-learn pyyaml
```

---

## 1. Training

Training uses **only** `cattle_detection_system.py`. All you need is a Roboflow-exported dataset (YOLOv8 format, with a `data.yaml`).

```python
from cattle_detection_system import CattleBehaviorDetector

# Start from a pretrained YOLOv8n model (no model_path = uses yolov8n.pt)
detector = CattleBehaviorDetector()

# 1. Create the project folder structure (data/, models/, results/, configs/, etc.)
detector.setup_project("cattle_behavior_project")

# 2. Import your Roboflow-exported dataset into the project
detector.add_roboflow_data("path/to/roboflow_export")

# 3. Train
results = detector.train_model(epochs=100, batch_size=16, img_size=640)

# 4. (Optional) Check accuracy
metrics = detector.validate_model()
```

**Notes:**
- The four behavior classes are fixed: `['drinking', 'standing', 'eating', 'lying']`. Your Roboflow class names must match these (case-insensitive).
- Trained weights are saved to:
  `cattle_behavior_project/models/cattle_behavior/weights/best.pt`
- To keep training an existing model with new/additional data, use `detector.continue_training(additional_epochs=50)` instead of `train_model()`. This loads the previous `best.pt` and saves a new model under `models/cattle_behavior_continued/`.

---

## 2. Testing / Inference (single video)

Once you have a trained model (`best.pt`), you can run it on a single video directly:

```python
from cattle_detection_system import CattleBehaviorDetector

detector = CattleBehaviorDetector("cattle_behavior_project/models/cattle_behavior/weights/best.pt")

stats = detector.process_video(
    "path/to/input_video.mp4",
    "path/to/output_video.mp4",
    conf_threshold=0.3,        # detection confidence threshold
    expected_cattle_count=12   # number of cattle expected in frame, used to sanity-check counts
)
```

This produces:
- An **annotated output video** (bounding boxes + behavior labels)
- An **HTML report** with interactive charts
- A **JSON file** with the raw stats

By default, reports are saved to `<project_dir>/results/reports`. To save them somewhere else, pass `report_dir` when creating the detector:

```python
detector = CattleBehaviorDetector(
    "cattle_behavior_project/models/cattle_behavior/weights/best.pt",
    report_dir="/path/to/your/reports/folder"
)
```

---

## 3. Testing / Inference (batch — many videos on a Slurm cluster)

For processing an entire folder of videos automatically, use `automate.py` + `Slurm_python_auto.sh`.

### Step 1 — Edit paths in `automate.py`

```python
input_folder = "/scratch/username/Input_cows"
output_folder = "/scratch/username/cattle_behavior_project/results/videos/output/Output_cows"
process_all_videos(input_folder, output_folder, expected_cattle_count=12)
```

- Replace `username` with your actual cluster username / account path.
- `input_folder`: recursively searched for `.mp4`, `.avi`, `.mov`, `.mkv` files
- `output_folder`: mirrors the same subfolder structure as the input
- Also update the model path used inside `automate.py`:
  ```python
  detector = CattleBehaviorDetector("cattle_behavior_project/models/cattle_behavior/weights/best.pt")
  ```
- If you want reports saved to a specific folder, pass `report_dir=...` when creating the detector (see section 2 above).

### Step 2 — Edit `Slurm_python_auto.sh` if needed

Update the placeholders (`username`, email, account/partition/GPU settings, conda env path) for your cluster, then confirm the script path:

```bash
python3 /scratch/username/automate.py
```

### Step 3 — Submit the job

```bash
sbatch Slurm_python_auto.sh
```

This will:
1. Load CUDA and activate the conda environment
2. Confirm GPU availability (`torch.cuda.is_available()`)
3. Run `automate.py`, which processes every video in `Input_cows/` and writes annotated videos + reports to the output folder

Check job status/logs via the files defined in the script:
- `video.out` — standard output
- `video.err` — standard error

---

## Quick Reference

| Task | Command / Call |
|---|---|
| Setup project | `detector.setup_project("cattle_behavior_project")` |
| Add training data | `detector.add_roboflow_data("path/to/export")` |
| Train from scratch | `detector.train_model(epochs=100)` |
| Continue training | `detector.continue_training(additional_epochs=50)` |
| Validate model | `detector.validate_model()` |
| Process one video | `detector.process_video(input, output, conf_threshold, expected_cattle_count)` |
| Process a folder of videos | `python automate.py` (local) or `sbatch Slurm_python_auto.sh` (cluster) |
