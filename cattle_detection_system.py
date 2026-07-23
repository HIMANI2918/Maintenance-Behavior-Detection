# Cattle Behavior Detection System using YOLOv8
# Complete project for detecting cattle behaviors: drinking, standing, lying, eating

import os
import cv2
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict, deque
import torch
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import supervision as sv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CattleBehaviorDetector:
    """
    Main class for cattle behavior detection system
    """
    
    def __init__(self, model_path: str = None, report_dir: str = None):
        """
        Args:
            model_path: Path to a trained .pt weights file. Falls back to a
                pretrained YOLOv8n model if not provided or not found.
            report_dir: Where HTML/JSON reports from process_video() get
                saved. Defaults to '<project_dir>/results/reports'.
        """
        self.behaviors = ['drinking', 'standing', 'eating', 'lying']  # Added drinking first to match Roboflow order
        self.behavior_colors = {
            'drinking': (171, 255, 255),   # #FFABAB in BGR format (OpenCV uses BGR)
            'standing': (255, 0, 0),       # Blue
            'eating': (0, 255, 0),         # Green  
            'lying': (0, 0, 255)           # Red
        }
        
        # Load or initialize model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            logger.info(f"Loaded custom model from {model_path}")
        else:
            self.model = YOLO('yolov8n.pt')  # Start with pretrained model
            logger.info("Loaded pretrained YOLOv8 model")
        
        # Tracking and behavior analysis
        self.tracker = sv.ByteTrack()
        self.behavior_history = defaultdict(lambda: deque(maxlen=30))  # 30 frame history
        self.cattle_stats = defaultdict(lambda: defaultdict(int))
        
        # Project directory setup
        project_dir = "cattle_behavior_project"
        self.project_dir = Path(project_dir)

        # Where generated HTML/JSON reports are saved
        self.report_dir = Path(report_dir) if report_dir else self.project_dir / 'results' / 'reports'
        
    def setup_project(self, project_dir: str = "cattle_behavior_project"):
        """
        Setup project directory structure
        """
        self.project_dir = Path(project_dir)
        
        # Create directory structure
        dirs = [
            'data/images/train',
            'data/images/val',
            'data/images/test',
            'data/labels/train',
            'data/labels/val', 
            'data/labels/test',
            'models',
            'results/reports',
            'results/videos/input',
            'results/videos/output',
            'logs',
            'configs'
        ]
        
        for dir_path in dirs:
            (self.project_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create dataset configuration
        self.create_dataset_config()
        
        logger.info(f"Project structure created at {self.project_dir}")
        
    def create_dataset_config(self):
        """
        Create YAML configuration for dataset
        """
        config = {
            'path': str(self.project_dir / 'data'),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.behaviors),
            'names': self.behaviors
        }
        
        config_path = self.project_dir / 'configs' / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Dataset config created at {config_path}")
        return config_path
    
    def prepare_training_data(self, video_path: str = None, roboflow_dataset_path: str = None):
        """
        Prepare training data from video OR Roboflow dataset
        """
        if roboflow_dataset_path:
            # Add Roboflow dataset
            logger.info("Adding Roboflow dataset...")
            success = self.add_roboflow_data(roboflow_dataset_path)
            if success:
                logger.info("Roboflow dataset added successfully!")
            return success
            
        elif video_path:
            # Extract frames from video for manual annotation
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            extracted_count = 0
            
            # Extract every 10th frame for annotation
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % 10 == 0:  # Extract every 10th frame
                    frame_path = self.project_dir / 'data' / 'images' / 'train' / f'frame_{extracted_count:06d}.jpg'
                    cv2.imwrite(str(frame_path), frame)
                    extracted_count += 1
                    
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {extracted_count} frames from {video_path}")
            logger.info("Please annotate these frames using Roboflow or the built-in annotation tool")
            return True
        else:
            logger.error("Please provide either video_path OR roboflow_dataset_path")
            return False
    
    def create_annotation_template(self):
        """
        Create annotation template and instructions for Roboflow
        """
        instructions = """
        CATTLE BEHAVIOR ANNOTATION INSTRUCTIONS FOR ROBOFLOW
        ===================================================
        
        Behaviors to annotate:
        - drinking (cattle drinking water, head near water source)
        - standing (upright, four legs supporting)
        - lying (body on ground, legs folded)
        - eating (head down, grazing or feeding)
        
        Instructions:
        1. Upload extracted frames to Roboflow
        2. Create bounding boxes around cattle
        3. Label each box with the appropriate behavior
        4. Export as YOLOv8 format when complete
        5. Use the exported data with add_roboflow_data() function
        
        Note: This system is designed to work with Roboflow annotations only.
        """
        
        with open(self.project_dir / 'ROBOFLOW_ANNOTATION_INSTRUCTIONS.txt', 'w') as f:
            f.write(instructions)
    
    def add_roboflow_data(self, roboflow_dataset_path: str, merge_with_existing: bool = True):
        """
        Add Roboflow YOLOv8 exported data to existing training dataset
        """
        roboflow_path = Path(roboflow_dataset_path)
        
        if not roboflow_path.exists():
            logger.error(f"Roboflow dataset path not found: {roboflow_dataset_path}")
            return False
        
        # Check if it's a Roboflow export structure
        if (roboflow_path / 'data.yaml').exists():
            data_yaml_path = roboflow_path / 'data.yaml'
        else:
            logger.error("No data.yaml found. Please ensure this is a YOLOv8 export from Roboflow")
            return False
        
        # Read Roboflow data.yaml
        with open(data_yaml_path, 'r') as f:
            roboflow_config = yaml.safe_load(f)
        
        # Verify class names match our behaviors
        roboflow_names = roboflow_config.get('names', [])
        if isinstance(roboflow_names, dict):
            roboflow_names = list(roboflow_names.values())
        
        # Check if classes match (case-insensitive)
        roboflow_names_lower = [name.lower() for name in roboflow_names]
        missing_behaviors = []
        for behavior in self.behaviors:
            if behavior not in roboflow_names_lower:
                missing_behaviors.append(behavior)
        
        if missing_behaviors:
            logger.warning(f"Missing behaviors in Roboflow data: {missing_behaviors}")
            logger.info(f"Roboflow classes: {roboflow_names}")
            logger.info(f"Expected classes: {self.behaviors}")
        
        # Create mapping from Roboflow class IDs to our class IDs
        class_mapping = {}
        for i, rf_name in enumerate(roboflow_names):
            rf_name_lower = rf_name.lower()
            if rf_name_lower in self.behaviors:
                our_class_id = self.behaviors.index(rf_name_lower)
                class_mapping[i] = our_class_id
                logger.info(f"Mapped Roboflow class {i} ({rf_name}) to our class {our_class_id} ({rf_name_lower})")
        
        if not class_mapping:
            logger.error("No matching classes found between Roboflow data and our behaviors")
            return False
        
        # Copy and process Roboflow data
        splits = ['train', 'valid', 'test']  # Roboflow uses 'valid' instead of 'val'
        total_added = 0
        
        for split in splits:
            rf_split_name = split
            our_split_name = 'val' if split == 'valid' else split
            
            rf_images_dir = roboflow_path / rf_split_name / 'images'
            rf_labels_dir = roboflow_path / rf_split_name / 'labels'
            
            if not rf_images_dir.exists():
                logger.info(f"No {rf_split_name} split found in Roboflow data")
                continue
            
            our_images_dir = self.project_dir / 'data' / 'images' / our_split_name
            our_labels_dir = self.project_dir / 'data' / 'labels' / our_split_name
            
            # Ensure directories exist
            our_images_dir.mkdir(parents=True, exist_ok=True)
            our_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each image and label
            split_count = 0
            for img_file in rf_images_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Copy image
                    dest_img = our_images_dir / img_file.name
                    if merge_with_existing and dest_img.exists():
                        # Create unique name if file exists
                        stem = img_file.stem
                        suffix = img_file.suffix
                        counter = 1
                        while dest_img.exists():
                            dest_img = our_images_dir / f"{stem}_rf{counter}{suffix}"
                            counter += 1
                    
                    import shutil
                    shutil.copy2(img_file, dest_img)
                    
                    # Process corresponding label
                    label_file = rf_labels_dir / f"{img_file.stem}.txt"
                    if label_file.exists():
                        dest_label = our_labels_dir / f"{dest_img.stem}.txt"
                        
                        # Read and convert class IDs
                        converted_lines = []
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    rf_class_id = int(parts[0])
                                    if rf_class_id in class_mapping:
                                        our_class_id = class_mapping[rf_class_id]
                                        converted_line = f"{our_class_id} {' '.join(parts[1:])}\n"
                                        converted_lines.append(converted_line)
                        
                        # Write converted labels
                        if converted_lines:
                            with open(dest_label, 'w') as f:
                                f.writelines(converted_lines)
                            split_count += 1
            
            logger.info(f"Added {split_count} images to {our_split_name} split")
            total_added += split_count
        
        logger.info(f"Successfully added {total_added} images from Roboflow dataset")
        
        # Update dataset config
        self.create_dataset_config()
        
        return True
    
    def train_model(self, epochs: int = 100, batch_size: int = 16, img_size: int = 640):
        """
        Train the YOLO model on cattle behavior data
        """
        config_path = self.project_dir / 'configs' / 'dataset.yaml'
        
        # Training parameters
        train_params = {
            'data': str(config_path),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'project': str(self.project_dir / 'models'),
            'name': 'cattle_behavior',
            'save_period': 10,
            'patience': 20,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        logger.info("Starting model training...")
        results = self.model.train(**train_params)
        
        # Save best model
        best_model_path = self.project_dir / 'models' / 'cattle_behavior' / 'weights' / 'best.pt'
        logger.info(f"Training completed. Best model saved at {best_model_path}")
        
        return results
    
    def continue_training(self, additional_epochs: int = 50, learning_rate: float = 0.001):
        """
        Continue training from the best checkpoint with new data
        """
        best_model_path = self.project_dir / 'models' / 'cattle_behavior' / 'weights' / 'best.pt'
        
        if not best_model_path.exists():
            logger.error("No previous training found. Use train_model() instead.")
            return None
        
        # Load the best model
        self.model = YOLO(str(best_model_path))
        logger.info(f"Loaded best model from {best_model_path}")
        
        config_path = self.project_dir / 'configs' / 'dataset.yaml'
        
        # Continue training with lower learning rate
        train_params = {
            'data': str(config_path),
            'epochs': additional_epochs,
            'batch': 16,
            'imgsz': 640,
            'project': str(self.project_dir / 'models'),
            'name': 'cattle_behavior_continued',
            'save_period': 10,
            'patience': 20,
            'lr0': learning_rate,  # Lower learning rate for fine-tuning
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'resume': False  # Start fresh with new data but use pretrained weights
        }
        
        logger.info("Continuing training with new data...")
        results = self.model.train(**train_params)
        
        # Update model path
        new_model_path = self.project_dir / 'models' / 'cattle_behavior_continued' / 'weights' / 'best.pt'
        logger.info(f"Continued training completed. New best model: {new_model_path}")
        
        return results
    
    def validate_model(self):
        """
        Validate trained model and generate metrics
        """
        config_path = self.project_dir / 'configs' / 'dataset.yaml'
        
        # Run validation
        metrics = self.model.val(
            data=str(config_path),
            project=str(self.project_dir / 'results'),
            name='validation'
        )
        
        logger.info("Model validation completed")
        return metrics
    
    def detect_behaviors(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect cattle behaviors in a single frame
        """
        results = self.model(image, conf=conf_threshold)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    if cls < len(self.behaviors):
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'behavior': self.behaviors[cls],
                            'class_id': cls
                        }
                        detections.append(detection)
        
        return detections
    
    def track_cattle(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Track individual cattle across frames
        """
        # Convert detections to supervision format
        detection_list = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            detection_list.append([x1, y1, x2, y2, det['confidence'], det['class_id']])
        
        if detection_list:
            detections_sv = sv.Detections(
                xyxy=np.array([d[:4] for d in detection_list]),
                confidence=np.array([d[4] for d in detection_list]),
                class_id=np.array([d[5] for d in detection_list])
            )
            
            # Update tracker
            detections_sv = self.tracker.update_with_detections(detections_sv)
            
            # Update behavior history
            tracked_detections = []
            for i, track_id in enumerate(detections_sv.tracker_id):
                if track_id is not None:
                    behavior = self.behaviors[detections_sv.class_id[i]]
                    self.behavior_history[track_id].append(behavior)
                    
                    tracked_det = detections[i].copy()
                    tracked_det['track_id'] = track_id
                    tracked_detections.append(tracked_det)
            
            return tracked_detections
        
        return []
    
    def analyze_behavior_patterns(self, track_id: int) -> Dict:
        """
        Analyze behavior patterns for a specific cattle
        """
        if track_id not in self.behavior_history:
            return {}
        
        history = list(self.behavior_history[track_id])
        if not history:
            return {}
        
        # Calculate behavior statistics
        behavior_counts = defaultdict(int)
        for behavior in history:
            behavior_counts[behavior] += 1
        
        total_frames = len(history)
        behavior_percentages = {
            behavior: (count / total_frames) * 100 
            for behavior, count in behavior_counts.items()
        }
        
        # Detect behavior transitions
        transitions = []
        for i in range(1, len(history)):
            if history[i] != history[i-1]:
                transitions.append((history[i-1], history[i]))
        
        analysis = {
            'total_observations': total_frames,
            'behavior_distribution': behavior_percentages,
            'dominant_behavior': max(behavior_counts.keys(), key=behavior_counts.get),
            'behavior_transitions': len(transitions),
            'current_behavior': history[-1] if history else 'unknown'
        }
        
        return analysis
    
    def draw_annotations(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and behavior labels on frame
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            behavior = detection['behavior']
            confidence = detection['confidence']
            color = self.behavior_colors.get(behavior, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{behavior}: {confidence:.2f}"
            if 'track_id' in detection:
                label = f"ID{detection['track_id']}: {label}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def process_video_with_complete_report(self, input_path: str, output_path: str = None, 
                                          conf_threshold: float = 0.3, expected_cattle_count: int = 12) -> Dict:
        """
        Process video with complete behavior analysis report for specified number of cattle
        """
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if fps > 0 else 0
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Enhanced tracking and statistics
        frame_count = 0
        behavior_stats = defaultdict(int)
        frame_behavior_counts = []  # Store behavior counts per frame
        detected_cattle_per_frame = []
        
        # For 12-cattle analysis (now with drinking)
        total_drinking = 0
        total_standing = 0
        total_lying = 0
        total_eating = 0
        
        logger.info(f"Processing video: {input_path}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps}, Duration: {duration_seconds:.1f}s")
        logger.info(f"Expected cattle count: {expected_cattle_count}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect behaviors
            detections = self.detect_behaviors(frame, conf_threshold)
            
            # Track cattle
            tracked_detections = self.track_cattle(detections, frame)
            
            # Count behaviors in this frame
            frame_behaviors = {'drinking': 0, 'standing': 0, 'lying': 0, 'eating': 0}
            detected_count = len(tracked_detections)
            
            for detection in tracked_detections:
                behavior = detection['behavior']
                frame_behaviors[behavior] += 1
                behavior_stats[behavior] += 1
            
            # Apply cattle count logic: undetected cattle are assumed to be lying
            if detected_count < expected_cattle_count and (frame_behaviors['lying'] >= frame_behaviors['standing'] and frame_behaviors['lying'] >= frame_behaviors['eating']):
                missing_cattle = expected_cattle_count - detected_count
                frame_behaviors['lying'] += missing_cattle
                behavior_stats['lying'] += missing_cattle
                logger.debug(f"Frame {frame_count}: Added {missing_cattle} lying cattle (detected: {detected_count}/{expected_cattle_count})")
            elif detected_count < expected_cattle_count and (frame_behaviors['standing'] > frame_behaviors['lying'] and frame_behaviors['standing'] >= frame_behaviors['eating']):
                #assuming undetected are standing if too many animals are standing
                missing_cattle = expected_cattle_count - detected_count
                frame_behaviors['standing'] += missing_cattle
                behavior_stats['standing'] += missing_cattle
                logger.debug(f"Frame {frame_count}: Adjusted standing count by {missing_cattle} (detected: {detected_count}/{expected_cattle_count})")
            elif detected_count < expected_cattle_count and (frame_behaviors['eating'] > frame_behaviors['lying'] and frame_behaviors['eating'] > frame_behaviors['standing']):
                # assuming undetected are eating if too many animals are eating
                missing_cattle = expected_cattle_count - detected_count
                frame_behaviors['eating'] += missing_cattle
                behavior_stats['eating'] += missing_cattle
                logger.debug(f"Frame {frame_count}: Adjusted eating count by {missing_cattle} (detected: {detected_count}/{expected_cattle_count})")
            
            # Store frame statistics
            frame_behavior_counts.append(frame_behaviors.copy())
            detected_cattle_per_frame.append(detected_count)
            
            # Add to totals for averaging
            total_drinking += frame_behaviors['drinking']
            total_standing += frame_behaviors['standing']
            total_lying += frame_behaviors['lying']
            total_eating += frame_behaviors['eating']
            
            # Draw annotations with enhanced info
            annotated_frame = self.draw_annotations(frame, tracked_detections)
            
            # Add comprehensive frame info
            info_lines = [
                f"Frame: {frame_count}/{total_frames} | Detected: {detected_count}/{expected_cattle_count}",
                f"Drinking: {frame_behaviors['drinking']} | Standing: {frame_behaviors['standing']}",
                f"Lying: {frame_behaviors['lying']} | Eating: {frame_behaviors['eating']}"
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = 30 + (i * 25)
                cv2.putText(annotated_frame, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame if output specified
            if writer:
                writer.write(annotated_frame)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}%")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        # Calculate comprehensive statistics
        total_observations = frame_count * expected_cattle_count
        
        # Average behaviors per cattle
        avg_drinking = total_drinking / frame_count if frame_count > 0 else 0
        avg_standing = total_standing / frame_count if frame_count > 0 else 0
        avg_lying = total_lying / frame_count if frame_count > 0 else 0
        avg_eating = total_eating / frame_count if frame_count > 0 else 0
        
        # Percentage of time each behavior occurs
        drinking_percentage = (total_drinking / total_observations) * 100 if total_observations > 0 else 0
        standing_percentage = (total_standing / total_observations) * 100 if total_observations > 0 else 0
        lying_percentage = (total_lying / total_observations) * 100 if total_observations > 0 else 0
        eating_percentage = (total_eating / total_observations) * 100 if total_observations > 0 else 0
        
        # Detection statistics
        avg_detected_per_frame = sum(detected_cattle_per_frame) / len(detected_cattle_per_frame) if detected_cattle_per_frame else 0
        detection_rate = (avg_detected_per_frame / expected_cattle_count) * 100
        
        # Generate comprehensive report
        processing_stats = {
            'video_info': {
                'filename': Path(input_path).name,
                'duration_seconds': duration_seconds,
                'total_frames': frame_count,
                'fps': fps,
                'resolution': f"{width}x{height}",
                'processing_time': datetime.now().isoformat()
            },
            'cattle_analysis': {
                'expected_cattle_count': expected_cattle_count,
                'average_detected_per_frame': round(avg_detected_per_frame, 2),
                'detection_rate_percentage': round(detection_rate, 2),
                'total_observations': total_observations
            },
            'behavior_averages_per_frame': {
                'drinking': round(avg_drinking, 2),
                'standing': round(avg_standing, 2),
                'lying': round(avg_lying, 2),
                'eating': round(avg_eating, 2)
            },
            'behavior_percentages': {
                'drinking': round(drinking_percentage, 2),
                'standing': round(standing_percentage, 2),
                'lying': round(lying_percentage, 2),
                'eating': round(eating_percentage, 2)
            },
            'behavior_totals': {
                'drinking': total_drinking,
                'standing': total_standing,
                'lying': total_lying,
                'eating': total_eating
            },
            'frame_by_frame_analysis': {
                'detected_cattle_per_frame': detected_cattle_per_frame,
                'behavior_counts_per_frame': frame_behavior_counts
            }
        }
        
        # Save detailed report to file
        report_filename = self.generate_detailed_report(processing_stats, input_path)
        
        logger.info("Video processing completed")
        logger.info(f"Detailed report saved to: {report_filename}")
        
        return processing_stats
    
    def generate_detailed_report(self, stats: Dict, video_path: str) -> str:
        """
        Generate comprehensive behavior analysis report with charts and statistics
        """
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"cattle_report_{video_name}.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cattle Behavior Analysis Report - {video_name}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; border-radius: 5px; margin-bottom: 30px; }}
                .section {{ margin: 30px 0; padding: 20px; background-color: #f8f9fa; border-radius: 5px; }}
                .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-box {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
                .stat-number {{ font-size: 2em; font-weight: bold; color: #3498db; }}
                .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
                th {{ background-color: #34495e; color: white; }}
                .chart-container {{ margin: 30px 0; }}
                .behavior-drinking {{ color: #FFABAB; }}
                .behavior-standing {{ color: #3498db; }}
                .behavior-lying {{ color: #e74c3c; }}
                .behavior-eating {{ color: #27ae60; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🐄 Cattle Behavior Analysis Report</h1>
                    <h2>{video_name}</h2>
                    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                
                <div class="section">
                    <h2>📊 Video Summary</h2>
                    <div class="stat-grid">
                        <div class="stat-box">
                            <div class="stat-number">{stats['video_info']['duration_seconds']:.1f}s</div>
                            <div class="stat-label">Duration</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-number">{stats['video_info']['total_frames']}</div>
                            <div class="stat-label">Total Frames</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-number">{stats['cattle_analysis']['expected_cattle_count']}</div>
                            <div class="stat-label">Expected Cattle</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-number">{stats['cattle_analysis']['detection_rate_percentage']:.1f}%</div>
                            <div class="stat-label">Detection Rate</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Behavior Analysis Summary</h2>
                    <div class="stat-grid">
                        <div class="stat-box">
                            <div class="stat-number behavior-drinking">{stats['behavior_averages_per_frame']['drinking']}</div>
                            <div class="stat-label">Avg Drinking per Frame</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-number behavior-standing">{stats['behavior_averages_per_frame']['standing']}</div>
                            <div class="stat-label">Avg Standing per Frame</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-number behavior-lying">{stats['behavior_averages_per_frame']['lying']}</div>
                            <div class="stat-label">Avg Lying per Frame</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-number behavior-eating">{stats['behavior_averages_per_frame']['eating']}</div>
                            <div class="stat-label">Avg Eating per Frame</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📈 Behavior Distribution</h2>
                    <table>
                        <tr>
                            <th>Behavior</th>
                            <th>Total Occurrences</th>
                            <th>Percentage of Time</th>
                            <th>Average per Frame</th>
                        </tr>
                        <tr class="behavior-drinking">
                            <td><strong>Drinking</strong></td>
                            <td>{stats['behavior_totals']['drinking']}</td>
                            <td>{stats['behavior_percentages']['drinking']:.1f}%</td>
                            <td>{stats['behavior_averages_per_frame']['drinking']}</td>
                        </tr>
                        <tr class="behavior-standing">
                            <td><strong>Standing</strong></td>
                            <td>{stats['behavior_totals']['standing']}</td>
                            <td>{stats['behavior_percentages']['standing']:.1f}%</td>
                            <td>{stats['behavior_averages_per_frame']['standing']}</td>
                        </tr>
                        <tr class="behavior-lying">
                            <td><strong>Lying</strong></td>
                            <td>{stats['behavior_totals']['lying']}</td>
                            <td>{stats['behavior_percentages']['lying']:.1f}%</td>
                            <td>{stats['behavior_averages_per_frame']['lying']}</td>
                        </tr>
                        <tr class="behavior-eating">
                            <td><strong>Eating</strong></td>
                            <td>{stats['behavior_totals']['eating']}</td>
                            <td>{stats['behavior_percentages']['eating']:.1f}%</td>
                            <td>{stats['behavior_averages_per_frame']['eating']}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>📊 Behavior Charts</h2>
                    <div class="chart-container">
                        <div id="behaviorPieChart" style="height: 400px;"></div>
                    </div>
                    <div class="chart-container">
                        <div id="timeSeriesChart" style="height: 400px;"></div>
                    </div>
                    <div class="chart-container">
                        <div id="detectionChart" style="height: 400px;"></div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📋 Technical Details</h2>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Video File</td><td>{stats['video_info']['filename']}</td></tr>
                        <tr><td>Resolution</td><td>{stats['video_info']['resolution']}</td></tr>
                        <tr><td>Frame Rate</td><td>{stats['video_info']['fps']} FPS</td></tr>
                        <tr><td>Total Observations</td><td>{stats['cattle_analysis']['total_observations']}</td></tr>
                        <tr><td>Processing Time</td><td>{stats['video_info']['processing_time']}</td></tr>
                    </table>
                </div>
            </div>
            
            <script>
                // Behavior Pie Chart (now with 4 behaviors: drinking, standing, lying, eating)
                var pieData = [{{
                    values: [{stats['behavior_percentages']['drinking']}, {stats['behavior_percentages']['standing']}, 
                             {stats['behavior_percentages']['lying']}, {stats['behavior_percentages']['eating']}],
                    labels: ['Drinking', 'Standing', 'Lying', 'Eating'],
                    type: 'pie',
                    marker: {{
                        colors: ['#FFABAB', '#3498db', '#e74c3c', '#27ae60']
                    }}
                }}];
                
                var pieLayout = {{
                    title: 'Behavior Distribution',
                    font: {{size: 14}}
                }};
                
                Plotly.newPlot('behaviorPieChart', pieData, pieLayout);
                
                // Time Series Chart (now with 4 behaviors)
                var frames = Array.from({{length: {len(stats['frame_by_frame_analysis']['behavior_counts_per_frame'])}}}, (_, i) => i);
                var drinkingData = {[frame['drinking'] for frame in stats['frame_by_frame_analysis']['behavior_counts_per_frame']]};
                var standingData = {[frame['standing'] for frame in stats['frame_by_frame_analysis']['behavior_counts_per_frame']]};
                var lyingData = {[frame['lying'] for frame in stats['frame_by_frame_analysis']['behavior_counts_per_frame']]};
                var eatingData = {[frame['eating'] for frame in stats['frame_by_frame_analysis']['behavior_counts_per_frame']]};
                
                var timeSeriesData = [
                    {{x: frames, y: drinkingData, name: 'Drinking', type: 'scatter', line: {{color: '#FFABAB'}}}},
                    {{x: frames, y: standingData, name: 'Standing', type: 'scatter', line: {{color: '#3498db'}}}},
                    {{x: frames, y: lyingData, name: 'Lying', type: 'scatter', line: {{color: '#e74c3c'}}}},
                    {{x: frames, y: eatingData, name: 'Eating', type: 'scatter', line: {{color: '#27ae60'}}}}
                ];
                
                var timeSeriesLayout = {{
                    title: 'Behavior Over Time',
                    xaxis: {{title: 'Frame Number'}},
                    yaxis: {{title: 'Number of Cattle'}},
                    font: {{size: 14}}
                }};
                
                Plotly.newPlot('timeSeriesChart', timeSeriesData, timeSeriesLayout);
                
                // Detection Rate Chart
                var detectedData = {stats['frame_by_frame_analysis']['detected_cattle_per_frame']};
                var detectionData = [{{
                    x: frames,
                    y: detectedData,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Detected Cattle',
                    line: {{color: '#9b59b6'}}
                }}, {{
                    x: frames,
                    y: Array({len(stats['frame_by_frame_analysis']['detected_cattle_per_frame'])}).fill({stats['cattle_analysis']['expected_cattle_count']}),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Expected Count',
                    line: {{color: '#95a5a6', dash: 'dash'}}
                }}];
                
                var detectionLayout = {{
                    title: 'Detection Rate Over Time',
                    xaxis: {{title: 'Frame Number'}},
                    yaxis: {{title: 'Number of Cattle Detected'}},
                    font: {{size: 14}}
                }};
                
                Plotly.newPlot('detectionChart', detectionData, detectionLayout);
            </script>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        # Also save JSON report for data analysis
        json_report_path = report_path.with_suffix('.json')
        with open(json_report_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"HTML report saved to: {report_path}")
        logger.info(f"JSON data saved to: {json_report_path}")
        
        return str(report_path)
    
    def process_video(self, input_path: str, output_path: str = None, 
                     conf_threshold: float = 0.5, expected_cattle_count: int = 12) -> Dict:
        """
        Process video file and detect cattle behaviors with complete reporting
        """
        return self.process_video_with_complete_report(input_path, output_path, conf_threshold, expected_cattle_count)
    
    def real_time_detection(self, camera_id: int = 0):
        """
        Real-time cattle behavior detection from camera
        """
        cap = cv2.VideoCapture(camera_id)
        
        logger.info("Starting real-time detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect behaviors
            detections = self.detect_behaviors(frame)
            
            # Track cattle
            tracked_detections = self.track_cattle(detections, frame)
            
            # Draw annotations
            annotated_frame = self.draw_annotations(frame, tracked_detections)
            
            # Display frame
            cv2.imshow('Cattle Behavior Detection', annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def get_model_accuracy_with_confusion_matrix(self):
        """
        Get detailed model accuracy metrics including confusion matrix
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Run validation on test set
        results = self.model.val(
            data=str(self.project_dir / 'configs' / 'dataset.yaml'),
            split='test',
            save_json=True  # Save results for confusion matrix
        )
        
        # Extract metrics
        metrics = {
            'overall_mAP50': float(results.box.map50),
            'overall_mAP50_95': float(results.box.map),
            'per_class_mAP50': results.box.maps.tolist() if results.box.maps is not None else [],
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'class_names': self.behaviors
        }
        
        # Print detailed results
        print(f"\n🎯 MODEL ACCURACY REPORT")
        print(f"{'='*60}")
        print(f"Overall mAP@0.5: {metrics['overall_mAP50']:.3f} ({metrics['overall_mAP50']*100:.1f}%)")
        print(f"Overall mAP@0.5:0.95: {metrics['overall_mAP50_95']:.3f} ({metrics['overall_mAP50_95']*100:.1f}%)")
        print(f"Overall Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
        print(f"Overall Recall: {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
        print(f"\nPer-Class Performance:")
        print(f"{'Behavior':<12} {'mAP@0.5':<10} {'Percentage':<12}")
        print(f"{'-'*35}")
        
        for i, behavior in enumerate(self.behaviors):
            if i < len(metrics['per_class_mAP50']):
                map_val = metrics['per_class_mAP50'][i]
                print(f"{behavior:<12} {map_val:.3f}     {map_val*100:.1f}%")
        
        # Generate confusion matrix
        self.generate_confusion_matrix()
        
        # Generate detailed performance report
        self.generate_performance_report(metrics)
        
        return metrics
    
    def generate_confusion_matrix(self):
        """
        Generate and visualize confusion matrix for the model
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path
        import json
        
        # Run prediction on test set to get predictions
        test_images_dir = self.project_dir / 'data' / 'images' / 'test'
        test_labels_dir = self.project_dir / 'data' / 'labels' / 'test'
        
        if not test_images_dir.exists():
            logger.warning("No test images found. Using validation set for confusion matrix.")
            test_images_dir = self.project_dir / 'data' / 'images' / 'val'
            test_labels_dir = self.project_dir / 'data' / 'labels' / 'val'
        
        if not test_images_dir.exists():
            logger.error("No test or validation images found for confusion matrix.")
            return
        
        # Collect predictions and ground truth
        y_true = []
        y_pred = []
        
        for img_file in test_images_dir.glob('*.jpg'):
            # Get ground truth labels
            label_file = test_labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue
            
            # Read ground truth
            gt_classes = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_classes.append(int(parts[0]))
            
            if not gt_classes:
                continue
            
            # Get predictions
            results = self.model(str(img_file), conf=0.25)
            pred_classes = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        pred_classes.append(int(box.cls.cpu().numpy()))
            
            # Match predictions to ground truth (simplified approach)
            # For each GT, find closest prediction
            for gt_class in gt_classes:
                y_true.append(gt_class)
                if pred_classes:
                    # Simple matching - use most confident prediction
                    y_pred.append(pred_classes[0] if pred_classes else -1)
                else:
                    y_pred.append(-1)  # No prediction
        
        if not y_true:
            logger.warning("No valid predictions found for confusion matrix.")
            return
        
        # Handle missing predictions (class -1)
        unique_classes = sorted(set(y_true + y_pred))
        if -1 in unique_classes:
            class_names = self.behaviors + ['No Detection']
        else:
            class_names = self.behaviors
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Cattle Behavior Detection', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Behavior', fontsize=12)
        plt.ylabel('True Behavior', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save confusion matrix
        cm_path = self.project_dir / 'results' / 'confusion_matrix.png'
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate per-class metrics from confusion matrix
        self.calculate_per_class_metrics(cm, class_names)
        
        logger.info(f"Confusion matrix saved to: {cm_path}")
        
        return cm
    
    def calculate_per_class_metrics(self, cm, class_names):
        """
        Calculate detailed per-class metrics from confusion matrix
        """
        print(f"\n📊 DETAILED PER-CLASS METRICS")
        print(f"{'='*70}")
        print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print(f"{'-'*70}")
        
        metrics_data = []
        
        for i, class_name in enumerate(class_names):
            if i >= len(cm):
                continue
                
            # True Positives
            tp = cm[i, i]
            
            # False Positives (predicted as this class but was other class)
            fp = cm[:, i].sum() - tp
            
            # False Negatives (was this class but predicted as other)
            fn = cm[i, :].sum() - tp
            
            # True Negatives
            tn = cm.sum() - tp - fp - fn
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            support = tp + fn
            
            print(f"{class_name:<12} {precision:.3f}     {recall:.3f}     {f1_score:.3f}     {support:<10}")
            
            metrics_data.append({
                'class': class_name,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': support,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            })
        
        # Calculate overall metrics
        total_tp = sum([m['tp'] for m in metrics_data])
        total_fp = sum([m['fp'] for m in metrics_data])
        total_fn = sum([m['fn'] for m in metrics_data])
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        print(f"{'-'*70}")
        print(f"{'Overall':<12} {overall_precision:.3f}     {overall_recall:.3f}     {overall_f1:.3f}     {sum([m['support'] for m in metrics_data])}")
        
        return metrics_data
    
    def generate_performance_report(self, metrics):
        """
        Generate comprehensive performance report with visualizations
        """
        import matplotlib.pyplot as plt
        
        # Create performance visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Per-class mAP bar chart
        if metrics['per_class_mAP50']:
            ax1.bar(self.behaviors, metrics['per_class_mAP50'], 
                   color=['#FFABAB', '#3498db', '#27ae60', '#e74c3c'])
            ax1.set_title('Per-Class mAP@0.5', fontweight='bold')
            ax1.set_ylabel('mAP@0.5')
            ax1.set_ylim(0, 1)
            for i, v in enumerate(metrics['per_class_mAP50']):
                ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. Overall metrics pie chart
        overall_metrics = [
            metrics['precision'], 
            metrics['recall'], 
            metrics['overall_mAP50']
        ]
        metric_labels = ['Precision', 'Recall', 'mAP@0.5']
        ax2.pie(overall_metrics, labels=metric_labels, autopct='%1.1f%%', 
               colors=['#3498db', '#e74c3c', '#27ae60'])
        ax2.set_title('Overall Performance Metrics', fontweight='bold')
        
        # 3. Performance comparison
        if metrics['per_class_mAP50']:
            behaviors_short = [b[:4] for b in self.behaviors]  # Shorten names
            x_pos = range(len(behaviors_short))
            
            # Create grouped bar chart for precision, recall, mAP
            width = 0.25
            ax3.bar([x - width for x in x_pos], [0.5] * len(behaviors_short), width, 
                   label='Target (50%)', alpha=0.7, color='gray')
            ax3.bar(x_pos, metrics['per_class_mAP50'], width, 
                   label='Current mAP@0.5', color='#3498db')
            
            ax3.set_title('Performance vs Target', fontweight='bold')
            ax3.set_ylabel('Score')
            ax3.set_xlabel('Behavior')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(behaviors_short)
            ax3.legend()
            ax3.set_ylim(0, 1)
        
        # 4. Model summary
        ax4.axis('off')
        summary_text = f"""
        MODEL PERFORMANCE SUMMARY
        
        Overall Accuracy: {metrics['overall_mAP50']*100:.1f}%
        
        Best Performing: {self.behaviors[metrics['per_class_mAP50'].index(max(metrics['per_class_mAP50']))] if metrics['per_class_mAP50'] else 'N/A'}
        Worst Performing: {self.behaviors[metrics['per_class_mAP50'].index(min(metrics['per_class_mAP50']))] if metrics['per_class_mAP50'] else 'N/A'}
        
        Precision: {metrics['precision']*100:.1f}%
        Recall: {metrics['recall']*100:.1f}%
        
        Total Classes: {len(self.behaviors)}
        """
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save performance report
        report_path = self.project_dir / 'results' / 'performance_report.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Performance report saved to: {report_path}")
        
        return report_path


# Updated BehaviorAnalyzer class to include drinking
class BehaviorAnalyzer:
    """
    Advanced behavior analysis and pattern recognition
    """
    
    def __init__(self):
        self.behavior_patterns = {
            'normal_drinking': {'duration': (60, 600), 'frequency': 'medium'},  # 1-10 minutes
            'normal_standing': {'duration': (60, 3600), 'frequency': 'high'},  # 1min-1hr
            'normal_lying': {'duration': (1800, 14400), 'frequency': 'medium'},  # 30min-4hrs
            'normal_eating': {'duration': (300, 1800), 'frequency': 'high'},  # 5-30 minutes
            'abnormal_patterns': {
                'excessive_lying': {'lying_percentage': 80},
                'no_eating': {'eating_percentage': 0},
                'no_drinking': {'drinking_percentage': 0},
                'excessive_drinking': {'drinking_percentage': 30}  # More than 30% might indicate illness
            }
        }
    
    def detect_anomalies(self, behavior_history: Dict) -> List[str]:
        """
        Detect abnormal behavior patterns that might indicate health issues
        """
        anomalies = []
        
        for track_id, history in behavior_history.items():
            if len(history) < 100:  # Need sufficient history
                continue
                
            behavior_counts = defaultdict(int)
            for behavior in history:
                behavior_counts[behavior] += 1
            
            total = len(history)
            
            # Check for anomalies
            drinking_pct = (behavior_counts['drinking'] / total) * 100
            lying_pct = (behavior_counts['lying'] / total) * 100
            eating_pct = (behavior_counts['eating'] / total) * 100
            standing_pct = (behavior_counts['standing'] / total) * 100
            
            if lying_pct > 80:
                anomalies.append(f"Cattle {track_id}: Excessive lying ({lying_pct:.1f}%)")
            
            if eating_pct == 0:
                anomalies.append(f"Cattle {track_id}: No eating behavior detected")
            
            if drinking_pct == 0:
                anomalies.append(f"Cattle {track_id}: No drinking behavior detected")
            
            if drinking_pct > 30:
                anomalies.append(f"Cattle {track_id}: Excessive drinking ({drinking_pct:.1f}%)")
                
            if standing_pct < 10:
                anomalies.append(f"Cattle {track_id}: Very low standing activity ({standing_pct:.1f}%)")
        
        return anomalies


def main():
    """
    Minimal example: load a trained model and process a single video.
    Update MODEL_PATH, INPUT_VIDEO, and OUTPUT_VIDEO for your setup, or see
    the README for the full training + batch-testing workflow.
    """
    MODEL_PATH = "cattle_behavior_project/models/cattle_behavior/weights/best.pt"
    INPUT_VIDEO = "cattle_behavior_project/results/videos/input/example.mp4"
    OUTPUT_VIDEO = "cattle_behavior_project/results/videos/output/example.mp4"

    start_time = datetime.now()

    detector = CattleBehaviorDetector(MODEL_PATH)
    stats = detector.process_video(
        INPUT_VIDEO,
        OUTPUT_VIDEO,
        conf_threshold=0.3,
        expected_cattle_count=12
    )

    time_taken = (datetime.now() - start_time).total_seconds() / 60
    print("Processing complete! Check the results folder for:")
    print("- HTML report with interactive charts")
    print("- JSON data file")
    print("- Annotated output video")
    print(f"- Time taken: {time_taken:.2f} minutes")


if __name__ == "__main__":
    main()


# Updated configuration templates
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 16,
    'img_size': 640,
    'learning_rate': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1
}

DETECTION_CONFIG = {
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    'max_detections': 100,
    'agnostic_nms': False
}
