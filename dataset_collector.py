"""
dataset_collector.py - Gender/Age Dataset Collection Tool

Allows users to manually annotate gender and age of people in videos
and saves labeled person crops for training.

Usage:
    python dataset_collector.py --source video.mp4
    or use as module: from dataset_collector import DatasetCollector
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from detector import PeopleDetector


class DatasetCollector:
    """
    Tool for collecting labeled person crops from video.

    How to use:
    1. Start video
    2. When a clear person appears, press SPACE to pause
    3. Click on the person to select
    4. Press 'M' for Male or 'F' for Female
    5. Press 'S' to save the crop
    6. Press SPACE to resume
    """

    def __init__(self, output_dir="data/gender_dataset"):
        """
        Initialize the dataset collector.

        Args:
            output_dir: Directory to save labeled person crops
        """
        self.output_dir = output_dir
        self.detector = PeopleDetector(confidence=0.4)

        # Create output structure
        self.male_dir = os.path.join(output_dir, "male")
        self.female_dir = os.path.join(output_dir, "female")
        os.makedirs(self.male_dir, exist_ok=True)
        os.makedirs(self.female_dir, exist_ok=True)

        # Annotation tracking
        self.annotations = []
        self.annotation_file = os.path.join(output_dir, "annotations.json")
        self._load_existing_annotations()

        print(f"[DatasetCollector] Output directory: {output_dir}")
        print(f"[DatasetCollector] Males: {self.male_dir}")
        print(f"[DatasetCollector] Females: {self.female_dir}")

    def _load_existing_annotations(self):
        """Load existing annotations if they exist."""
        if os.path.exists(self.annotation_file):
            with open(self.annotation_file, 'r') as f:
                self.annotations = json.load(f)
            print(f"[DatasetCollector] Loaded {len(self.annotations)} existing annotations")

    def _save_annotations(self):
        """Save annotations to JSON file."""
        with open(self.annotation_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)

    def get_next_filename(self, gender):
        """Get the next available filename for a labeled crop."""
        target_dir = self.male_dir if gender == 'M' else self.female_dir
        existing = len([f for f in os.listdir(target_dir) if f.endswith('.jpg')])
        return f"{gender.lower()}_{existing + 1:04d}.jpg"

    def save_crop(self, frame, bbox, gender):
        """
        Save a person crop with gender label.

        Args:
            frame: Full frame from video
            bbox: [x1, y1, x2, y2] bounding box
            gender: 'M' or 'F'

        Returns:
            Filename saved, or None if failed
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        # Extract and validate crop
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 50 or crop.shape[1] < 30:
            print("[DatasetCollector] ⚠ Crop too small, skipping")
            return None

        # Save crop
        filename = self.get_next_filename(gender)
        target_dir = self.male_dir if gender == 'M' else self.female_dir
        filepath = os.path.join(target_dir, filename)

        cv2.imwrite(filepath, crop)

        # Record annotation
        gender_label = 'Male' if gender == 'M' else 'Female'
        annotation = {
            'filename': filename,
            'gender': gender_label,
            'bbox': [int(v) for v in [x1, y1, x2, y2]],
            'timestamp': datetime.now().isoformat()
        }
        self.annotations.append(annotation)
        self._save_annotations()

        print(f"[DatasetCollector] ✓ Saved: {filename} ({gender_label})")
        return filename

    def collect_from_video(self, video_path):
        """
        Interactive mode: collect labeled crops from video.

        Controls:
            SPACE  - Pause/Resume
            Click  - Select person in paused frame
            M      - Mark as Male
            F      - Mark as Female
            S      - Save the selected person
            Q      - Quit
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[DatasetCollector] Error opening video: {video_path}")
            return

        paused = False
        current_frame = None
        detections = []
        selected_detection = None

        print("\n" + "="*60)
        print("Dataset Collection Mode")
        print("="*60)
        print("\nControls:")
        print("  SPACE  - Pause/Resume")
        print("  Click  - Select person in paused frame")
        print("  M      - Mark selected as Male")
        print("  F      - Mark selected as Female")
        print("  S      - Save selected person")
        print("  Q      - Quit")
        print("="*60 + "\n")

        frame_count = 0

        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_detection
            if event == cv2.EVENT_LBUTTONDOWN and paused and detections:
                # Find detection closest to click
                min_dist = float('inf')
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = [int(v) for v in det[:4]]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    dist = ((x - cx)**2 + (y - cy)**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        selected_detection = i

        cv2.namedWindow('Dataset Collector')
        cv2.setMouseCallback('Dataset Collector', mouse_callback)

        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[DatasetCollector] End of video reached")
                    break

                current_frame = frame
                frame_count += 1

                # Run detection
                detections = self.detector.detect(frame)
                selected_detection = None

            # Draw display frame
            display = current_frame.copy()

            # Draw all detections
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = [int(v) for v in det[:4]]
                conf = det[4]

                # Highlight selected detection
                if i == selected_detection:
                    color = (0, 255, 0)  # Green for selected
                    thickness = 3
                else:
                    color = (100, 100, 255)  # Light red for others
                    thickness = 2

                cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(display, f"{conf:.2f}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Display info
            status = "PAUSED" if paused else "PLAYING"
            cv2.putText(display, f"Frame: {frame_count} [{status}]", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"Detections: {len(detections)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if selected_detection is not None:
                cv2.putText(display, f"Selected: Det #{selected_detection}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Dataset Collector', display)

            key = cv2.waitKey(30 if paused else 1) & 0xFF

            if key == ord('q'):
                print("[DatasetCollector] Quitting...")
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('m') and selected_detection is not None and paused:
                self.save_crop(current_frame, detections[selected_detection], 'M')
            elif key == ord('f') and selected_detection is not None and paused:
                self.save_crop(current_frame, detections[selected_detection], 'F')

        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "="*60)
        print(f"Collection complete!")
        print(f"Total annotations: {len(self.annotations)}")
        print(f"Male samples: {len([a for a in self.annotations if a['gender'] == 'Male'])}")
        print(f"Female samples: {len([a for a in self.annotations if a['gender'] == 'Female'])}")
        print("="*60 + "\n")

    def get_statistics(self):
        """Get dataset statistics."""
        male_count = len([a for a in self.annotations if a['gender'] == 'Male'])
        female_count = len([a for a in self.annotations if a['gender'] == 'Female'])

        return {
            'total': len(self.annotations),
            'male': male_count,
            'female': female_count,
            'male_dir': self.male_dir,
            'female_dir': self.female_dir
        }

    def create_train_val_split(self, val_ratio=0.2):
        """
        Create train/validation split and save to separate directories.

        Args:
            val_ratio: Ratio of data to use for validation
        """
        np.random.seed(42)

        train_dir = os.path.join(self.output_dir, "train")
        val_dir = os.path.join(self.output_dir, "val")

        for split_dir in [train_dir, val_dir]:
            for sub in ['male', 'female']:
                os.makedirs(os.path.join(split_dir, sub), exist_ok=True)

        # Shuffle and split annotations
        indices = np.arange(len(self.annotations))
        np.random.shuffle(indices)

        train_size = int(len(indices) * (1 - val_ratio))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Move files to train/val directories
        for idx in train_indices:
            ann = self.annotations[idx]
            gender_lower = ann['gender'].lower()
            src = os.path.join(self.output_dir, gender_lower, ann['filename'])
            dst = os.path.join(train_dir, gender_lower, ann['filename'])
            if os.path.exists(src):
                os.rename(src, dst)

        for idx in val_indices:
            ann = self.annotations[idx]
            gender_lower = ann['gender'].lower()
            src = os.path.join(self.output_dir, gender_lower, ann['filename'])
            dst = os.path.join(val_dir, gender_lower, ann['filename'])
            if os.path.exists(src):
                os.rename(src, dst)

        print(f"[DatasetCollector] Train/Val split created")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Val: {len(val_indices)} samples")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Collection Tool")
    parser.add_argument("--source", type=str, required=True, help="Video file path")
    parser.add_argument("--output", type=str, default="data/gender_dataset", help="Output directory")
    args = parser.parse_args()

    collector = DatasetCollector(output_dir=args.output)
    collector.collect_from_video(args.source)

    stats = collector.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total: {stats['total']}")
    print(f"  Male: {stats['male']}")
    print(f"  Female: {stats['female']}")
