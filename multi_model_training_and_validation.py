"""
Cattle Behavior Detection — Multi-Model Training, Validation & Testing Pipeline
================================================================================

Trains and evaluates YOLO object detection architectures (YOLOv3-tiny, YOLOv8n/s/m/x,
YOLOv10l, YOLO11l, YOLO12l) for multi-class cattle behavior detection
(drinking / standing / eating / lying), with explicit, fully pinned hyperparameters
for reproducibility.

Capabilities:
  1. Trains each model with identical, explicitly specified hyperparameters
     (optimizer, learning rate, augmentation, early stopping, checkpoint selection).
  2. Reports validation-set metrics (Ultralytics' `val:` split) for each model in a
     unified comparison table.
  3. Evaluates already-trained checkpoints on the held-out test split (`split="test"`)
     independently of training, reporting both overall and per-behavior-class
     Precision / Recall / F1 / mAP@0.5 / mAP@0.5:0.90, saved as CSV and Excel.
  4. Runs tracked video inference (ByteTrack) with duplicate-detection removal and
     per-individual behavior-duration calculation.
  5. Prints a reproducibility report (software/hardware versions, hyperparameters,
     inference settings) for the current run.

USAGE
-----
  Train and validate:
    python multi_model_training_and_validation.py \\
        --data configs/dataset.yaml \\
        --models yolov3-tiny yolov8n yolov8s yolov8m yolov8x yolov10l yolo11l yolo12l \\
        --epochs 100 --batch 16 --imgsz 640 --seed 0 \\
        --deployed-model yolov8n \\
        --video path/to/video.mp4

  Evaluate already-trained checkpoints on the test split:
    python multi_model_training_and_validation.py \\
        --data configs/dataset.yaml \\
        --models yolov3-tiny yolov8n yolov8s yolov8m yolov8x yolov10l yolo11l yolo12l \\
        --skip-training --eval-test \\
        --project-dir cattle_behavior_project/models \\
        --deployed-model yolov8n \\
        --test-out test_comparison_table.csv

  --eval-test can also be combined with training in a single run.
"""

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import yaml


# =========================================================================
# Reproducibility settings
# =========================================================================

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


TRAIN_HPARAMS = dict(
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    auto_augment="randaugment",
    erasing=0.4,
    close_mosaic=10,
    patience=20,
    save_period=10,
)

INFERENCE_CONF_THRESHOLD = 0.25
INFERENCE_IOU_THRESHOLD = 0.5
INFERENCE_MAX_DET = 300
INFERENCE_AGNOSTIC_NMS = True

BYTETRACK_SETTINGS = dict(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30,
)

MAX_EXPECTED_CATTLE = 12

TEST_EVAL_CONF = 0.001
TEST_EVAL_IOU = 0.6


# =========================================================================
# Model registry
# =========================================================================

MODEL_REGISTRY = {
    "yolov3-tiny": "yolov3-tinyu.pt",
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8x": "yolov8x.pt",
    "yolov10l": "yolov10l.pt",
    "yolo11l": "yolo11l.pt",
    "yolo12l": "yolo12l.pt",
}


# =========================================================================
# Training + validation
# =========================================================================

def train_one_model(name, data_yaml, epochs, batch, imgsz, seed, project_dir, workers=8, weights_dir=None):
    from ultralytics import YOLO

    weight = MODEL_REGISTRY[name]
    local_weight_path = os.path.join(weights_dir, weight) if weights_dir else None

    if local_weight_path and os.path.exists(local_weight_path):
        weight_to_load = local_weight_path
    else:
        weight_to_load = weight

    print(f"\n{'='*70}\nTRAINING {name}  (weights: {weight_to_load})\n{'='*70}")
    model = YOLO(weight_to_load)

    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        seed=seed,
        deterministic=True,
        project=project_dir,
        name=name,
        workers=workers,
        device="cuda" if _cuda_available() else "cpu",
        **TRAIN_HPARAMS,
    )

    t0 = time.time()
    model.train(**train_kwargs)
    training_time_min = (time.time() - t0) / 60.0

    best_path = os.path.join(project_dir, name, "weights", "best.pt")
    size_mb = os.path.getsize(best_path) / (1024 * 1024) if os.path.exists(best_path) else None

    val_model = YOLO(best_path) if os.path.exists(best_path) else model
    val_metrics = val_model.val(data=data_yaml, project=project_dir, name=f"{name}_val")

    return {
        "model": name,
        "epochs": epochs,
        "training_time_min": round(training_time_min, 2),
        "precision": round(float(val_metrics.box.mp), 4),
        "recall": round(float(val_metrics.box.mr), 4),
        "map50": round(float(val_metrics.box.map50) * 100, 2),
        "map50_90": round(float(val_metrics.box.map) * 100, 2),
        "size_mb": round(size_mb, 2) if size_mb else None,
        "best_weights_path": best_path,
    }


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def build_comparison_table(rows, deployed_model_name, external_csv=None):
    all_rows = list(rows)
    if external_csv and os.path.exists(external_csv):
        with open(external_csv) as f:
            for r in csv.DictReader(f):
                r["best_weights_path"] = "(external)"
                all_rows.append(r)

    print(f"\n{'='*90}")
    print("VALIDATION-SET RESULTS")
    print(f"{'Model':<22} {'Epochs':>7} {'Time(min)':>10} {'P':>7} {'R':>7} {'AP50(%)':>9} {'AP50-90(%)':>11} {'Size(MB)':>9}")
    print("-" * 90)
    for r in all_rows:
        tag = "  <-- DEPLOYED" if r["model"] == deployed_model_name else ""
        print(
            f"{r['model']:<22} {r['epochs']:>7} {r['training_time_min']:>10} "
            f"{r['precision']:>7} {r['recall']:>7} {r['map50']:>9} {r['map50_90']:>11} "
            f"{r['size_mb']:>9}{tag}"
        )
    print("=" * 90)
    return all_rows



def build_tracker():
    import inspect
    import supervision as sv

    sig = inspect.signature(sv.ByteTrack.__init__)
    accepted = set(sig.parameters.keys())
    kwargs = {k: v for k, v in BYTETRACK_SETTINGS.items() if k in accepted}
    tracker = sv.ByteTrack(**kwargs)
    print(f"ByteTrack instantiated with: {kwargs}")
    return tracker


def remove_duplicate_boxes(xyxy, confidences, class_ids, iou_thresh=0.5, max_keep=MAX_EXPECTED_CATTLE):
    if len(xyxy) == 0:
        return xyxy, confidences, class_ids

    order = np.argsort(-confidences)
    keep = []

    def iou(a, b):
        xa1, ya1, xa2, ya2 = a
        xb1, yb1, xb2, yb2 = b
        inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
        inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
        inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
        area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    for idx in order:
        box = xyxy[idx]
        if all(iou(box, xyxy[k]) < iou_thresh for k in keep):
            keep.append(idx)

    if len(keep) > max_keep:
        keep = sorted(keep, key=lambda i: -confidences[i])[:max_keep]

    keep = sorted(keep)
    return xyxy[keep], confidences[keep], class_ids[keep]


def process_video_with_duration_tracking(video_path, weights_path, behaviors, fps_override=None):
    import cv2
    import supervision as sv
    from ultralytics import YOLO
    from collections import defaultdict

    model = YOLO(weights_path)
    tracker = build_tracker()

    cap = cv2.VideoCapture(video_path)
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30

    track_behavior_frames = defaultdict(lambda: defaultdict(int))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=INFERENCE_CONF_THRESHOLD,
            iou=INFERENCE_IOU_THRESHOLD,
            max_det=INFERENCE_MAX_DET,
            agnostic_nms=INFERENCE_AGNOSTIC_NMS,
            verbose=False,
        )[0]

        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls_id = boxes.cls.cpu().numpy().astype(int)

            xyxy, conf, cls_id = remove_duplicate_boxes(xyxy, conf, cls_id)

            detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls_id)
            detections = tracker.update_with_detections(detections)

            for i, track_id in enumerate(detections.tracker_id):
                if track_id is not None:
                    behavior = behaviors[detections.class_id[i]]
                    track_behavior_frames[int(track_id)][behavior] += 1

        frame_idx += 1

    cap.release()

    duration_report = {
        track_id: {behavior: round(count / fps, 2) for behavior, count in behavior_counts.items()}
        for track_id, behavior_counts in track_behavior_frames.items()
    }
    return duration_report, fps, frame_idx


# =========================================================================
# Reproducibility report
# =========================================================================

def print_reproducibility_report(deployed_weights_path, repo_dir="."):
    print(f"\n{'='*78}\nREPRODUCIBILITY REPORT — generated {datetime.now().isoformat()}\n{'='*78}")

    def kv(label, value, note=""):
        val = value if value not in (None, "", []) else "NOT FOUND / NOT SET"
        line = f"  {label:<40s}: {val}"
        if note:
            line += f"   [{note}]"
        print(line)

    print("\n-- Environment --")
    kv("Python version", sys.version.split()[0])
    kv("OS", f"{platform.system()} {platform.release()}")
    try:
        import cv2
        kv("OpenCV version", cv2.__version__)
    except ImportError:
        kv("OpenCV version", None)
    try:
        import supervision as sv
        kv("Supervision version", sv.__version__)
    except ImportError:
        kv("Supervision version", None)
    try:
        import torch
        kv("PyTorch version", torch.__version__)
        kv("CUDA version", torch.version.cuda)
        kv("CUDA available", torch.cuda.is_available())
    except ImportError:
        kv("PyTorch version", None)
    nvidia_smi = subprocess.run(
        "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader",
        shell=True, capture_output=True, text=True,
    )
    kv("GPU / driver", nvidia_smi.stdout.strip() or None)

    print("\n-- Git --")
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        commit = subprocess.run(f"git -C {repo_dir} rev-parse HEAD", shell=True, capture_output=True, text=True).stdout.strip()
        tag = subprocess.run(f"git -C {repo_dir} describe --tags --always", shell=True, capture_output=True, text=True).stdout.strip()
        kv("Commit hash", commit or None)
        kv("Tag/release", tag or None)
    else:
        kv("Commit hash", None, note="no .git found at repo_dir")

    print("\n-- Training configuration --")
    kv("Random seed", "set via set_global_seed() + train(seed=...)")
    kv("Optimizer", TRAIN_HPARAMS["optimizer"])
    kv("lr0 / lrf", f"{TRAIN_HPARAMS['lr0']} / {TRAIN_HPARAMS['lrf']}")
    kv("momentum / weight_decay", f"{TRAIN_HPARAMS['momentum']} / {TRAIN_HPARAMS['weight_decay']}")
    kv("Early-stopping patience", TRAIN_HPARAMS["patience"])
    kv("Checkpoint save_period", TRAIN_HPARAMS["save_period"])

    print("\n-- Inference configuration --")
    kv("Confidence threshold", INFERENCE_CONF_THRESHOLD)
    kv("NMS IoU threshold", INFERENCE_IOU_THRESHOLD)
    kv("Max detections per frame", INFERENCE_MAX_DET)
    kv("Class-agnostic NMS", INFERENCE_AGNOSTIC_NMS)
    kv("ByteTrack settings", BYTETRACK_SETTINGS)
    kv("Duplicate-box removal", f"class-agnostic NMS + IoU dedup + hard cap at {MAX_EXPECTED_CATTLE}")

    if deployed_weights_path and os.path.exists(deployed_weights_path):
        try:
            import torch
            ckpt = torch.load(deployed_weights_path, map_location="cpu", weights_only=False)
            print("\n-- Deployed checkpoint --")
            kv("Ultralytics version", ckpt.get("version"))
            kv("Training date", ckpt.get("date"))
        except Exception as e:
            kv("Deployed checkpoint metadata", None, note=str(e))


# =========================================================================
# Test-set evaluation
# =========================================================================

def resolve_weights_path(project_dir, model_name):
    return os.path.join(project_dir, model_name, "weights", "best.pt")


def check_test_split_defined(data_yaml):
    with open(data_yaml, "r") as f:
        d = yaml.safe_load(f)
    if "test" not in d or not d["test"]:
        print(
            f"ERROR: '{data_yaml}' has no 'test:' key. Add one pointing at your "
            f"held-out test folder, then rerun with --eval-test."
        )
        sys.exit(1)


def evaluate_one_on_test(model_name, weights_path, data_yaml, project_dir, imgsz, batch, conf, iou, device):
    from ultralytics import YOLO

    if not os.path.exists(weights_path):
        print(f"SKIP {model_name}: weights not found at {weights_path}")
        return None, []

    print(f"\n{'='*80}\nEvaluating {model_name} on TEST split\n  weights: {weights_path}\n{'='*80}")
    model = YOLO(weights_path)

    results = model.val(
        data=data_yaml,
        split="test",
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=device,
        project=project_dir,
        name=f"{model_name}_test",
        plots=True,
        save_json=False,
    )

    size_mb = round(os.path.getsize(weights_path) / (1024 * 1024), 2)
    precision = float(results.box.mp)
    recall = float(results.box.mr)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    overall = {
        "model": model_name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "map50": round(float(results.box.map50) * 100, 2),
        "map50_90": round(float(results.box.map) * 100, 2),
        "size_mb": size_mb,
        "weights_path": weights_path,
        "output_dir": os.path.join(project_dir, f"{model_name}_test"),
    }

    per_class_rows = []
    try:
        class_indices = list(results.box.ap_class_index)
        names = results.names
        p_per_class = results.box.p
        r_per_class = results.box.r
        ap50_per_class = results.box.ap50
        ap5090_per_class = results.box.ap

        for i, cls_id in enumerate(class_indices):
            cls_name = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else names[int(cls_id)]
            p_c = float(p_per_class[i])
            r_c = float(r_per_class[i])
            f1_c = (2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0
            per_class_rows.append({
                "model": model_name,
                "class": cls_name,
                "precision": round(p_c, 4),
                "recall": round(r_c, 4),
                "f1": round(f1_c, 4),
                "map50": round(float(ap50_per_class[i]) * 100, 2),
                "map50_90": round(float(ap5090_per_class[i]) * 100, 2),
            })
    except Exception as e:
        print(f"  WARNING: could not extract per-class metrics for {model_name}: {e}")

    return overall, per_class_rows


def run_test_evaluation(models, data_yaml, project_dir, weights_overrides, imgsz, batch, conf, iou,
                         deployed_model, out_csv):
    check_test_split_defined(data_yaml)
    device = 0 if _cuda_available() else "cpu"

    rows = []
    per_class_all = []
    missing = []
    for name in models:
        weights_path = weights_overrides.get(name, resolve_weights_path(project_dir, name))
        row, per_class_rows = evaluate_one_on_test(
            model_name=name,
            weights_path=weights_path,
            data_yaml=data_yaml,
            project_dir=project_dir,
            imgsz=imgsz,
            batch=batch,
            conf=conf,
            iou=iou,
            device=device,
        )
        if row is None:
            missing.append(name)
            continue
        rows.append(row)
        per_class_all.extend(per_class_rows)

    if not rows:
        print("\nNo models evaluated -- check weights paths and try --test-weights-map.")
        return

    print(f"\n\n{'='*110}")
    print("TEST-SET RESULTS")
    print(f"{'='*110}")
    print(f"{'Model':<18} {'P':>8} {'R':>8} {'F1':>8} {'mAP@0.5(%)':>11} {'mAP@0.5:0.90(%)':>16} {'Size(MB)':>10}")
    print("-" * 110)
    for r in rows:
        tag = "  <-- DEPLOYED" if r["model"] == deployed_model else ""
        print(
            f"{r['model']:<18} {r['precision']:>8} {r['recall']:>8} {r['f1']:>8} "
            f"{r['map50']:>11} {r['map50_90']:>16} {r['size_mb']:>10}{tag}"
        )
    print("=" * 110)

    if missing:
        print(f"\nSkipped (best.pt not found): {', '.join(missing)}")

    fieldnames = ["model", "precision", "recall", "f1", "map50", "map50_90", "size_mb", "weights_path", "output_dir"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved test-set comparison table (CSV) to: {out_csv}")

    per_class_csv = os.path.splitext(out_csv)[0] + "_per_class.csv"
    if per_class_all:
        print(f"\n{'='*100}")
        print("PER-CLASS (BEHAVIOR) RESULTS -- TEST SET")
        print(f"{'='*100}")
        print(f"{'Model':<18} {'Class':<14} {'P':>8} {'R':>8} {'F1':>8} {'mAP@0.5(%)':>11} {'mAP@0.5:0.90(%)':>16}")
        print("-" * 100)
        for r in per_class_all:
            print(
                f"{r['model']:<18} {r['class']:<14} {r['precision']:>8} {r['recall']:>8} {r['f1']:>8} "
                f"{r['map50']:>11} {r['map50_90']:>16}"
            )
        print("=" * 100)

        pc_fieldnames = ["model", "class", "precision", "recall", "f1", "map50", "map50_90"]
        with open(per_class_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pc_fieldnames)
            writer.writeheader()
            writer.writerows(per_class_all)
        print(f"\nSaved per-class comparison table (CSV) to: {per_class_csv}")

    xlsx_path = os.path.splitext(out_csv)[0] + ".xlsx"
    try:
        save_test_results_xlsx(rows, per_class_all, deployed_model, xlsx_path)
        print(f"Saved test-set comparison table (Excel) to: {xlsx_path}")
    except ImportError:
        print("\nopenpyxl not installed -- skipped Excel output. "
              "Install with: pip install openpyxl")


def _style_header_row(ws, n_cols, header_fill, header_font, border, center):
    for col_idx in range(1, n_cols + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center
        cell.border = border


def save_test_results_xlsx(rows, per_class_rows, deployed_model, xlsx_path):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()

    header_fill = PatternFill(start_color="305496", end_color="305496", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    thin = Side(style="thin", color="B0B0B0")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")
    deployed_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    ws = wb.active
    ws.title = "Overall Results"
    headers = ["Model", "Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.90", "Size (MB)"]
    ws.append(headers)
    _style_header_row(ws, len(headers), header_fill, header_font, border, center)

    for r in rows:
        ws.append([
            r["model"],
            round(r["precision"], 4),
            round(r["recall"], 4),
            round(r["f1"], 4),
            round(r["map50"] / 100, 4),
            round(r["map50_90"] / 100, 4),
            r["size_mb"],
        ])
        row_idx = ws.max_row
        is_deployed = (r["model"] == deployed_model)
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.border = border
            cell.alignment = center
            if is_deployed:
                cell.fill = deployed_fill
        for col_idx in (2, 3, 4, 5, 6):
            ws.cell(row=row_idx, column=col_idx).number_format = "0.0000"

    if deployed_model:
        ws.append([])
        note = ws.cell(row=ws.max_row + 1, column=1, value=f"Highlighted row = deployed model ({deployed_model})")
        note.font = Font(italic=True, size=9)

    for col_idx, header in enumerate(headers, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = max(14, len(header) + 4)
    ws.freeze_panes = "A2"

    if per_class_rows:
        ws2 = wb.create_sheet("Per-Class Results")
        headers2 = ["Model", "Class", "Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.90"]
        ws2.append(headers2)
        _style_header_row(ws2, len(headers2), header_fill, header_font, border, center)

        for r in per_class_rows:
            ws2.append([
                r["model"],
                r["class"],
                round(r["precision"], 4),
                round(r["recall"], 4),
                round(r["f1"], 4),
                round(r["map50"] / 100, 4),
                round(r["map50_90"] / 100, 4),
            ])
            row_idx = ws2.max_row
            is_deployed = (r["model"] == deployed_model)
            for col_idx in range(1, len(headers2) + 1):
                cell = ws2.cell(row=row_idx, column=col_idx)
                cell.border = border
                cell.alignment = center
                if is_deployed:
                    cell.fill = deployed_fill
            for col_idx in (3, 4, 5, 6, 7):
                ws2.cell(row=row_idx, column=col_idx).number_format = "0.0000"

        for col_idx, header in enumerate(headers2, start=1):
            ws2.column_dimensions[get_column_letter(col_idx)].width = max(14, len(header) + 4)
        ws2.freeze_panes = "A2"

    wb.save(xlsx_path)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                         help=f"Models to train/evaluate. Choices: {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--weights-dir", default=None,
                         help="Folder of pre-downloaded pretrained weights, for offline training nodes.")
    parser.add_argument("--project-dir", default="cattle_behavior_project/models")
    parser.add_argument("--deployed-model", default="yolov8n")
    parser.add_argument("--video", default=None, help="Optional: run tracked duration inference on this video")
    parser.add_argument("--external-results", default=None, help="Optional CSV of models trained outside this script")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--deployed-weights", default=None)

    parser.add_argument("--eval-test", action="store_true",
                         help="Evaluate each model in --models on the test split (data.yaml 'test:' key required).")
    parser.add_argument("--test-weights-map", nargs="*", default=[],
                         help="Overrides as name=path pairs for --eval-test.")
    parser.add_argument("--test-conf", type=float, default=TEST_EVAL_CONF)
    parser.add_argument("--test-iou", type=float, default=TEST_EVAL_IOU)
    parser.add_argument("--test-out", default="test_comparison_table.csv")
    args = parser.parse_args()

    unknown = [m for m in args.models if m not in MODEL_REGISTRY]
    if unknown:
        print(f"WARNING: unknown models skipped: {unknown}")

    set_global_seed(args.seed)

    behaviors = ["drinking", "standing", "eating", "lying"]
    rows = []
    deployed_weights_path = args.deployed_weights

    if not args.skip_training:
        for name in args.models:
            if name not in MODEL_REGISTRY:
                continue
            row = train_one_model(name, args.data, args.epochs, args.batch, args.imgsz, args.seed,
                                   args.project_dir, args.workers, args.weights_dir)
            rows.append(row)
            if name == args.deployed_model:
                deployed_weights_path = row["best_weights_path"]

        build_comparison_table(rows, args.deployed_model, args.external_results)

    if args.eval_test:
        test_weights_overrides = {}
        for pair in args.test_weights_map:
            if "=" not in pair:
                print(f"ERROR: --test-weights-map entries must be name=path, got: {pair}")
                sys.exit(1)
            name, path = pair.split("=", 1)
            test_weights_overrides[name] = path

        valid_models = [m for m in args.models if m in MODEL_REGISTRY]
        run_test_evaluation(
            models=valid_models,
            data_yaml=args.data,
            project_dir=args.project_dir,
            weights_overrides=test_weights_overrides,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.test_conf,
            iou=args.test_iou,
            deployed_model=args.deployed_model,
            out_csv=args.test_out,
        )

    if args.video and deployed_weights_path and os.path.exists(deployed_weights_path):
        print(f"\nRunning tracked inference with duration calculation on: {args.video}")
        duration_report, fps, n_frames = process_video_with_duration_tracking(
            args.video, deployed_weights_path, behaviors
        )
        print(f"Processed {n_frames} frames at {fps:.1f} FPS")
        for track_id, durations in sorted(duration_report.items()):
            print(f"  Track {track_id}: {durations}")

        with open("duration_report.json", "w") as f:
            json.dump(duration_report, f, indent=2)
        print("Saved to duration_report.json")

    print_reproducibility_report(deployed_weights_path, args.repo_dir)


if __name__ == "__main__":
    main()
