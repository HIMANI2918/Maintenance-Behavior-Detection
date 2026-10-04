# Cattle Behavior Detection System

A YOLOv8n-based system that detects and tracks cattle behaviors — drinking, standing, eating, lying — in video footage, and generates HTML/JSON reports summarizing behavior over time. The repository also includes a multi-architecture training/validation/testing pipeline used to compare YOLOv3-tiny, YOLOv8n/s/m/x, YOLOv10l, YOLO11l, and YOLO12l prior to selecting YOLOv8n as the deployed model.

## Repository Contents

| File | Purpose |
|---|---|
| `cattle_detection_system.py` | Core class (`CattleBehaviorDetector`): training, validation, and video processing for the deployed YOLOv8n model |
| `multi_model_training_and_validation.py` | Trains and compares all eight architectures, evaluates trained checkpoints on the held-out test split (overall + per-behavior-class metrics), and runs tracked video inference with duration calculation |
| `automate.py` | Loops `CattleBehaviorDetector` over every video in a folder (batch inference) |
| `Slurm_python_auto.sh` | Slurm batch script that runs `automate.py` on a GPU node |
| `dataset.yaml` | Dataset configuration (class names, train/val/test paths) in YOLO format |
| `daytime_model_best.pt` | Trained YOLOv8n checkpoint — Daytime model (see [Pretrained Model Weights](#pretrained-model-weights) below) |
| `daytime_nighttime_model_best.pt` | Trained YOLOv8n checkpoint — Daytime + Nighttime model (see [Pretrained Model Weights](#pretrained-model-weights) below) |

## ⚙️ Lines you need to edit before running

If you're a new user cloning this repo, here's exactly what to change:

### `cattle_detection_system.py`
- Lines 1332–1334 (inside `main()`, only used if you run this file directly, e.g. `python cattle_detection_system.py`): update `MODEL_PATH`, `INPUT_VIDEO`, `OUTPUT_VIDEO` to your own paths.
- No other edits needed — reports go to `<project_dir>/results/reports` by default, or wherever you pass via `report_dir=` (see Section 2 below).

### `automate.py`
- Line 33: model weights path — `detector = CattleBehaviorDetector("cattle_behavior_project/models/cattle_behavior/weights/best.pt")` (or point this at `daytime_model_best.pt` / `daytime_nighttime_model_best.pt` to use the pretrained checkpoints included in this repo)
- Line 40: `input_folder = "/scratch/username/Input_cows"` → your video input folder
- Line 41: `output_folder = "/scratch/username/cattle_behavior_project/results/videos/output/Output_cows"` → your desired output folder

### `Slurm_python_auto.sh`
- Line 5: `#SBATCH --mail-user=username@example.edu` → your email
- Line 9 (`#SBATCH --partition=...`) and line 10 (`#SBATCH --account=...`): your cluster's GPU partition/account
- Line 17: `source /home/username/miniconda3/etc/profile.d/conda.sh` → your conda install path
- Line 18: `conda activate /scratch/username/myenv` → your conda environment path
- Line 20: `python3 /scratch/username/automate.py` → the actual path to your copy of `automate.py`

## Requirements

- Python 3.9+
- `ultralytics` (YOLOv8n and comparison architectures)
- `torch` (with CUDA if using GPU)
- `opencv-python`
- `supervision` (ByteTrack)
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `pyyaml`
- `openpyxl` (for Excel output from the multi-model test-evaluation pipeline)

```bash
pip install ultralytics torch opencv-python supervision pandas numpy matplotlib seaborn scikit-learn pyyaml openpyxl
```
## Training Configuration (all models)

All eight architectures compared in this work (YOLOv3-tiny, YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8x, YOLOv10l, YOLO11l, YOLO12l) were trained under the following explicitly specified hyperparameters:

- Epochs: 100, batch size: 16, image size: 640 × 640
- Optimizer: SGD, initial/final learning rate (lr0/lrf): 0.01 / 0.01, momentum: 0.937, weight decay: 0.0005
- Cosine learning-rate schedule with 3-epoch linear warm-up (warmup momentum 0.8, warmup bias LR 0.1)
- Data augmentation: HSV color jitter (h=0.015, s=0.7, v=0.4), horizontal flip (p=0.5), mosaic (p=1.0, disabled for the final 10 epochs), random translation (0.1), random scaling (0.5), RandAugment, random erasing (0.4)
- Early stopping: patience of 20 epochs (no improvement in validation mAP@0.5)
- Checkpoints saved every 10 epochs; best checkpoint selected via Ultralytics' internal fitness score (weighted mAP@0.5 + mAP@0.5:0.95)
- Random seed fixed across all runs for reproducibility
- A fixed random seed, SGD optimizer, and the full augmentation parameter set above are set explicitly in `multi_model_training_and_validation.py` (`TRAIN_HPARAMS`), rather than left at framework defaults.

## 1. Training

Training uses only `cattle_detection_system.py`. All you need is a Roboflow-exported dataset (YOLOv8 format, with a `data.yaml`).

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

Notes:
- The four behavior classes are fixed: `['drinking', 'standing', 'eating', 'lying']`. Your Roboflow class names must match these (case-insensitive).
- Trained weights are saved to: `cattle_behavior_project/models/cattle_behavior/weights/best.pt`
- To keep training an existing model with new/additional data, use `detector.continue_training(additional_epochs=50)` instead of `train_model()`. This loads the previous `best.pt` and saves a new model under `models/cattle_behavior_continued/`.

## 2. Testing / Inference (single video)

Once you have a trained model (`best.pt`, or one of the included pretrained checkpoints), you can run it on a single video directly:

```python
from cattle_detection_system import CattleBehaviorDetector

detector = CattleBehaviorDetector("daytime_model_best.pt")

stats = detector.process_video(
    "path/to/input_video.mp4",
    "path/to/output_video.mp4",
    conf_threshold=0.3,        # detection confidence threshold
    expected_cattle_count=12   # number of cattle expected in frame, used to sanity-check counts
)
```

This produces:
- An annotated output video (bounding boxes + behavior labels)
- An HTML report with interactive charts
- A JSON file with the raw stats

By default, reports are saved to `<project_dir>/results/reports`. To save them somewhere else, pass `report_dir` when creating the detector:

```python
detector = CattleBehaviorDetector(
    "daytime_model_best.pt",
    report_dir="/path/to/your/reports/folder"
)
```

## 3. Testing / Inference (batch — many videos on a Slurm cluster)

For processing an entire folder of videos automatically, use `automate.py` + `Slurm_python_auto.sh`.

**Step 1 — Edit paths in `automate.py`**

```python
input_folder = "/scratch/username/Input_cows"
output_folder = "/scratch/username/cattle_behavior_project/results/videos/output/Output_cows"
process_all_videos(input_folder, output_folder, expected_cattle_count=12)
```

Replace `username` with your actual cluster username/account path.
- `input_folder`: recursively searched for `.mp4`, `.avi`, `.mov`, `.mkv` files
- `output_folder`: mirrors the same subfolder structure as the input

Also update the model path used inside `automate.py`:

```python
detector = CattleBehaviorDetector("cattle_behavior_project/models/cattle_behavior/weights/best.pt")
```

If you want reports saved to a specific folder, pass `report_dir=...` when creating the detector (see Section 2 above).

**Step 2 — Edit `Slurm_python_auto.sh` if needed**

Update the placeholders (username, email, account/partition/GPU settings, conda env path) for your cluster, then confirm the script path:

```bash
python3 /scratch/username/automate.py
```

**Step 3 — Submit the job**

```bash
sbatch Slurm_python_auto.sh
```

This will:
- Load CUDA and activate the conda environment
- Confirm GPU availability (`torch.cuda.is_available()`)
- Run `automate.py`, which processes every video in `Input_cows/` and writes annotated videos + reports to the output folder

Check job status/logs via the files defined in the script:
- `video.out` — standard output
- `video.err` — standard error

## 4. Multi-Model Comparison & Testing

`multi_model_training_and_validation.py` trains and compares all eight architectures, and separately evaluates any already-trained checkpoint (including the two included in this repo) on the held-out **test** split.

**Train and compare all architectures** (validation-set metrics):

```bash
python multi_model_training_and_validation.py \
    --data dataset.yaml \
    --models yolov3-tiny yolov8n yolov8s yolov8m yolov8x yolov10l yolo11l yolo12l \
    --epochs 100 --batch 16 --imgsz 640 --seed 0 \
    --deployed-model yolov8n
```

**Evaluate an already-trained checkpoint on the test split** (no retraining needed — works with `daytime_model_best.pt` / `daytime_nighttime_model_best.pt` directly):

```bash
python multi_model_training_and_validation.py \
    --data dataset.yaml \
    --models yolov8n \
    --skip-training --eval-test \
    --test-weights-map yolov8n=daytime_model_best.pt \
    --deployed-model yolov8n \
    --test-out test_comparison_table.csv
```

This requires `dataset.yaml` to define a `test:` key pointing at your held-out test images. It reports Precision, Recall, F1, mAP@0.5, and mAP@0.5:0.95 both overall and per behavior class, saving results as CSV and a formatted Excel workbook (`test_comparison_table.xlsx`).

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
| Train & compare all architectures | `python multi_model_training_and_validation.py --data dataset.yaml --models ...` |
| Evaluate a checkpoint on the test split | `python multi_model_training_and_validation.py --skip-training --eval-test ...` |
