import os
from pathlib import Path
from datetime import datetime
from cattle_detection_system import CattleBehaviorDetector


def process_all_videos(input_dir, output_dir, expected_cattle_count=12, conf_threshold=0.3):
    detector = CattleBehaviorDetector("cattle_behavior_project/models/cattle_behavior/weights/best.pt")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    # Find all video files recursively
    video_files = [f for f in input_dir.rglob('*') if f.suffix.lower() in video_extensions]

    print(f"Found {len(video_files)} video(s) in {input_dir}")

    for video_path in video_files:
        # Mirror input folder structure in output
        rel_path = video_path.relative_to(input_dir)
        output_video_path = output_dir / rel_path
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing: {video_path}")
        start_time = datetime.now()
        stats = detector.process_video(
            str(video_path),
            str(output_video_path),
            conf_threshold=conf_threshold,
            expected_cattle_count=expected_cattle_count
        )
        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds() / 60
        print(f"Done: {output_video_path} | Time taken: {time_taken:.2f} min")
        print("-" * 60)


if __name__ == "__main__":
    # Update these paths for your environment
    input_folder = "/scratch/username/Input_cows"
    output_folder = "/scratch/username/cattle_behavior_project/results/videos/output/Output_cows"
    # The report save folder is set via CattleBehaviorDetector(report_dir=...) or
    # defaults to <project_dir>/results/reports — see README for details.
    process_all_videos(input_folder, output_folder, expected_cattle_count=12)
