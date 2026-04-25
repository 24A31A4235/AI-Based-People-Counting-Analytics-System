# Fix for Windows DLL loading issues with PyTorch + Python 3.14
# Must execute BEFORE any torch/ultralytics imports
import os as _os
_os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
try:
    _torch_lib = _os.path.join(
        _os.environ.get('APPDATA', ''), 'Python', 'Python314',
        'site-packages', 'torch', 'lib'
    )
    if _os.path.isdir(_torch_lib):
        _os.add_dll_directory(_torch_lib)
except (OSError, AttributeError):
    pass

"""
main.py - Entry Point for the People Counting System (Improved)

This is the main application controller that orchestrates:
- Video capture (webcam or file)
- YOLOv8 person detection
- SORT-based multi-object tracking (with hybrid association)
- Person Re-ID (runs BEFORE counter for deduplication)
- Line-crossing counting logic (ReID-aware)
- PyQt5 dashboard UI rendering
- Frame processing pipeline

Key improvements:
- ReID runs BEFORE counter → stable IDs feed into deduplication
- Config values are used properly (no more hardcoded buffer_zone/cooldown)
- Improved pipeline ordering for maximum accuracy

Usage:
    python main.py                  # Start with webcam 0
    python main.py --source 1       # Use webcam 1
    python main.py --source video.mp4  # Use video file
"""

import os
import sys
import cv2
import numpy as np
import argparse
import time

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer, Qt

from detector import PeopleDetector
from tracker import SORTTracker
from counter import LineCrossingCounter
from ui import PeopleCountingUI
from gender_detector import GenderAgeDetector
from reid import PersonReID
from config import get_config, print_config
from utils import (
    FPSCounter, draw_fancy_bbox, draw_counting_line,
    draw_trail, draw_overlay_counters, save_counts_to_file,
    resize_frame_aspect, COLORS
)


class PeopleCountingApp:
    """
    Main application controller.
    
    Manages the complete pipeline:
    1. Capture video frames
    2. Run YOLOv8 detection
    3. Update SORT tracker
    4. Run ReID (BEFORE counter — critical for deduplication)
    5. Check line crossings (with ReID stable IDs)
    6. Draw visualizations
    7. Update UI
    """

    def __init__(self, source=0, model_path="yolov8m.pt", confidence=0.40, preset="MEDIUM"):
        """
        Initialize the application.

        Args:
            source: Video source (0/1 for webcam, filepath for video)
            model_path: Path to YOLO model weights
            confidence: Detection confidence threshold
            preset: Configuration preset ("SPARSE", "MEDIUM", "DENSE")
        """
        # ==========================================
        # Load Configuration
        # ==========================================
        self.config = get_config(preset)
        print_config(self.config)

        # ==========================================
        # Initialize PyQt5 Application
        # ==========================================
        self.app = QApplication(sys.argv)
        self.app.setStyle("Fusion")

        # ==========================================
        # Initialize UI
        # ==========================================
        self.ui = PeopleCountingUI()
        self.ui.show()
        self.ui.update_status("Initializing...")

        # ==========================================
        # Initialize Detection & Tracking Pipeline
        # ==========================================
        print("=" * 60)
        print("  AI-Based People Counting & Analytics System")
        print("  [Improved Pipeline v2.0]")
        print("=" * 60)

        # Initialize detector
        self.ui.update_status("Loading YOLOv8 model...")
        QApplication.processEvents()  # Force UI update

        try:
            self.detector = PeopleDetector(
                model_path=self.config.detector.model_path,
                confidence=self.config.detector.confidence,
                iou_threshold=self.config.detector.iou_threshold
            )
        except Exception as e:
            QMessageBox.critical(
                self.ui, "Model Error",
                f"Failed to load YOLOv8 model:\n{str(e)}\n\n"
                f"Make sure 'ultralytics' is installed:\n"
                f"pip install ultralytics"
            )
            sys.exit(1)

        # Initialize tracker (uses config values now)
        self.tracker = SORTTracker(
            max_age=self.config.tracker.max_age,
            min_hits=self.config.tracker.min_hits,
            iou_threshold=self.config.tracker.iou_threshold
        )

        # Counter will be initialized when video starts (needs frame width)
        self.counter = None

        # FPS counter
        self.fps_counter = FPSCounter(avg_count=30)

        # ==========================================
        # Initialize Gender/Age & Re-ID Systems
        # ==========================================
        self.ui.update_status("Loading AI Models...")
        QApplication.processEvents()

        self.gender_detector = GenderAgeDetector(
            custom_model_path=self.config.gender.model_path if self.config.gender.model_path else None
        )
        self.reid = PersonReID(
            similarity_threshold=self.config.reid.similarity_threshold,
            max_gallery_size=self.config.reid.max_gallery_size,
            feature_update_rate=self.config.reid.feature_update_rate
        )

        # ==========================================
        # Analytics Tracking
        # ==========================================
        self.unique_person_ids = set()  # All unique stable IDs ever seen
        self.peak_count = 0             # Peak simultaneous persons
        self.is_webcam_source = True    # Track if current source is webcam
        self.track_to_stable = {}       # Mapping: track.id -> reid stable_id

        # ==========================================
        # Video Capture Setup
        # ==========================================
        self.video_source = source
        self.cap = None
        self.is_running = False
        self.frame_width = 0
        self.frame_height = 0

        # ==========================================
        # Processing Timer (drives the main loop)
        # ==========================================
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_frame)

        # ==========================================
        # Connect UI Signals
        # ==========================================
        self.ui.start_btn.clicked.connect(self._toggle_running)
        self.ui.reset_btn.clicked.connect(self._reset_counters)
        self.ui.save_btn.clicked.connect(self._save_counts)
        self.ui.line_slider.valueChanged.connect(self._update_line_position)

        # ==========================================
        # Ready
        # ==========================================
        self.ui.update_status("Ready — Press START")
        print("\n[App] System ready. Press START to begin.")

    def _open_video(self, source):
        """
        Open a video source (webcam or file).
        
        Args:
            source: Integer for webcam index, or string path for video file
            
        Returns:
            True if opened successfully, False otherwise
        """
        if self.cap is not None:
            self.cap.release()

        # Determine source
        if isinstance(source, str) and source != "file":
            if not os.path.exists(source):
                QMessageBox.warning(
                    self.ui, "File Not Found",
                    f"Video file not found:\n{source}"
                )
                return False
            self.cap = cv2.VideoCapture(source)
            print(f"[App] Opening video file: {source}")
        else:
            cam_index = int(source) if isinstance(source, (int, float)) else 0
            self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[App] Opening webcam: {cam_index}")

        if not self.cap.isOpened():
            QMessageBox.warning(
                self.ui, "Video Error",
                f"Could not open video source: {source}\n\n"
                "If using webcam, make sure it's connected.\n"
                "If using a video file, check the path."
            )
            return False

        # Get frame dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[App] Frame size: {self.frame_width}x{self.frame_height}")

        # Initialize counter with CONFIG VALUES (not hardcoded!)
        line_x = int(self.frame_width * self.ui.get_line_position_ratio())
        self.counter = LineCrossingCounter(
            line_x=line_x,
            buffer_zone=self.config.counter.buffer_zone,
            cooldown_frames=self.config.counter.cooldown_frames,
            frame_width=self.frame_width
        )
        print(f"[App] Counter initialized with config: buffer_zone={self.config.counter.buffer_zone}, "
              f"cooldown={self.config.counter.cooldown_frames}")

        return True

    def _toggle_running(self):
        """Toggle between running and stopped states."""
        if self.is_running:
            self._stop()
        else:
            self._start()

    def _start(self):
        """Start the video processing pipeline."""
        # Get source from UI
        source_data = self.ui.source_combo.currentData()
        
        if source_data == "file":
            # File was already selected in the combo change handler
            if self.ui.video_source and isinstance(self.ui.video_source, str):
                source = self.ui.video_source
            else:
                QMessageBox.information(
                    self.ui, "Select File",
                    "Please select a video file from the dropdown first."
                )
                return
        else:
            source = source_data
            self.is_webcam_source = True

        # Open video source
        if not self._open_video(source):
            return

        self.is_running = True
        self.ui.set_running_state(True)

        # Start the processing timer
        # Target ~30 FPS processing rate
        self.timer.start(1)  # 1ms timer, actual FPS depends on processing
        print("[App] Processing started")

    def _stop(self):
        """Stop the video processing pipeline."""
        self.is_running = False
        self.timer.stop()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.ui.set_running_state(False)
        print("[App] Processing stopped")

    def _reset_counters(self):
        """Reset all counters and tracking data."""
        if self.counter:
            self.counter.reset()
        self.tracker.reset()
        self.reid.reset()
        self.gender_detector.reset()
        self.unique_person_ids.clear()
        self.track_to_stable.clear()
        self.peak_count = 0
        self.ui.update_counters(0, 0)
        self.ui.update_persons_count(0)
        self.ui.update_gender_counts(0, 0)
        self.ui.update_analytics(0, 0)
        print("[App] All counters and tracking data reset")

    def _save_counts(self):
        """Save current counts to a JSON log file."""
        if self.counter:
            in_c, out_c = self.counter.get_counts()
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "counts_log.json")
            save_counts_to_file(in_c, out_c, filepath)
            self.ui.update_status(f"Counts saved to {filepath}")
            QMessageBox.information(
                self.ui, "Saved",
                f"Counts saved to:\n{filepath}\n\n"
                f"IN: {in_c}\nOUT: {out_c}"
            )
        else:
            QMessageBox.information(self.ui, "Info", "No data to save yet.")

    def _update_line_position(self, value):
        """Update the counting line position from slider."""
        if self.counter and self.frame_width > 0:
            line_x = int(self.frame_width * (value / 100.0))
            self.counter.set_line_position(line_x)
            self.counter.set_frame_width(self.frame_width)

    def _process_frame(self):
        """
        Main frame processing pipeline.
        Called by the QTimer at regular intervals.
        
        IMPROVED Pipeline Order:
        1. Read frame from video
        2. Run YOLOv8 detection (with temporal smoothing)
        3. Update SORT tracker (with hybrid association)
        4. Run ReID FIRST → build stable_id_map 
        5. Check line crossings (with stable_id deduplication)
        6. Gender classification
        7. Draw visualizations
        8. Update UI
        """
        if not self.is_running or self.cap is None:
            return

        # ==========================================
        # Step 1: Read frame
        # ==========================================
        ret, frame = self.cap.read()
        if not ret:
            # Video ended - check if it's a file (loop) or webcam (error)
            if isinstance(self.video_source, str):
                # Video file ended - loop back to beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
            else:
                self.ui.update_status("⚠ Camera feed lost")
                return

        # ==========================================
        # Step 1.5: Fix camera mirror issue
        # Flip horizontally for webcam so left/right are correct
        # ==========================================
        source_data = self.ui.source_combo.currentData()
        is_webcam = isinstance(source_data, int) or (isinstance(source_data, str) and source_data.isdigit())
        if is_webcam:
            frame = cv2.flip(frame, 1)  # Horizontal flip to fix mirror effect

        # ==========================================
        # Step 2: Run detection (with temporal smoothing)
        # ==========================================
        detections = self.detector.detect(frame)

        # ==========================================
        # Step 3: Update tracker (hybrid IoU + centroid association)
        # ==========================================
        active_tracks = self.tracker.update(detections)

        # ==========================================
        # Step 4: Run ReID FIRST → build stable_id_map
        # This MUST happen before the counter so dedup works!
        # ==========================================
        stable_id_map = {}  # {track_id: stable_id}
        display_data = {}   # {track_id: {'stable_id': int, 'gender': str, 'age': str}}
        
        for track in active_tracks:
            bbox = track.get_bbox()
            
            # Get stable unique ID from ReID
            stable_id, is_new = self.reid.get_unique_id(frame, bbox, track.id)
            stable_id_map[track.id] = stable_id
            self.track_to_stable[track.id] = stable_id
            
            # Record unique persons by stable ID
            self.unique_person_ids.add(stable_id)

        # ==========================================
        # Step 5: Check line crossings (with stable_id deduplication!)
        # ==========================================
        if self.counter:
            # Update line position and frame width for distance calculations
            line_ratio = self.ui.get_line_position_ratio()
            line_x = int(self.frame_width * line_ratio)
            self.counter.line_x = line_x
            self.counter.set_frame_width(self.frame_width)
            
            # Pass stable_id_map for deduplication!
            in_count, out_count, crossed_ids = self.counter.update(
                active_tracks, stable_id_map=stable_id_map
            )
        else:
            in_count, out_count, crossed_ids = 0, 0, set()
            line_x = self.frame_width // 2

        # ==========================================
        # Step 6: Gender/Age classification
        # ==========================================
        for track in active_tracks:
            bbox = track.get_bbox()
            stable_id = stable_id_map.get(track.id, track.id)
            
            # Classify gender using stable ID for consistent voting
            res = self.gender_detector.classify(frame, bbox, stable_id)
            
            # Update global gender counts (only once per unique person)
            if res['gender'] in ('Male', 'Female'):
                self.gender_detector.update_gender_counts(stable_id, res['gender'])
                
            display_data[track.id] = {
                'stable_id': stable_id,
                'gender': res['gender'],
                'age': res['age']
            }

        # Update peak count
        current_count = len(active_tracks)
        if current_count > self.peak_count:
            self.peak_count = current_count

        # ==========================================
        # Step 7: Visualizations
        # ==========================================
        display_frame = frame.copy()

        # Draw counting line
        flash = self.counter.is_flashing() if self.counter else False
        draw_counting_line(display_frame, line_x, self.frame_height, flash)

        # Draw tracked persons
        for track in active_tracks:
            bbox = track.get_bbox()
            crossed = self.counter.has_crossed(track.id) if self.counter else False
            
            d_info = display_data.get(track.id, {})
            stable_id = d_info.get('stable_id', track.id)
            gender = d_info.get('gender', 'Unknown')
            age = d_info.get('age', '')
            
            # Bbox label shows Stable Person ID and Gender/Age
            draw_fancy_bbox(
                display_frame, bbox, stable_id,
                crossed=crossed, gender=gender, age=age
            )
            
            if len(track.trail) > 1:
                draw_trail(display_frame, track.trail)

        # Draw frame overlays
        fps = self.fps_counter.tick()
        m_count, f_count = self.gender_detector.get_gender_counts()
        draw_overlay_counters(
            display_frame, in_count, out_count, fps,
            male_count=m_count, female_count=f_count,
            total_unique=len(self.unique_person_ids)
        )

        # Timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            display_frame, timestamp, (self.frame_width - 240, self.frame_height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
        )

        # ==========================================
        # Step 8: Update UI
        # ==========================================
        self.ui.display_frame(display_frame)
        self.ui.update_counters(in_count, out_count)
        self.ui.update_fps(fps)
        self.ui.update_persons_count(len(active_tracks))
        self.ui.update_gender_counts(m_count, f_count)
        self.ui.update_analytics(len(self.unique_person_ids), self.peak_count)

    def run(self):
        """Start the application event loop."""
        return self.app.exec_()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Based People Counting & Analytics System"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: webcam index (0, 1, ...) or video file path"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8m.pt",
        help="YOLOv8 model path (default: yolov8m.pt, auto-downloads)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.40,
        help="Detection confidence threshold (default: 0.40)"
    )
    parser.add_argument(
        "--preset", type=str, default="MEDIUM",
        help="Configuration preset: SPARSE, MEDIUM, DENSE (default: MEDIUM)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Parse source (could be int for webcam or string for file)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # Create and run the application
    app = PeopleCountingApp(
        source=source,
        model_path=args.model,
        confidence=args.confidence,
        preset=args.preset
    )

    sys.exit(app.run())
