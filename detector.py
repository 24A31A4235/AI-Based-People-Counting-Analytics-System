"""
detector.py - People Detection Module (Improved)
Uses YOLOv8 (Ultralytics) for real-time person detection.
Only detects humans (COCO class 0 = 'person').

Improvements:
- Temporal smoothing to suppress flickering detections
- Configurable confidence with higher default for webcam
- Better NMS handling
"""

# Fix for Windows DLL loading issues with PyTorch + Python 3.14
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
try:
    _torch_lib = os.path.join(
        os.environ.get('APPDATA', ''), 'Python', 'Python314',
        'site-packages', 'torch', 'lib'
    )
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
except (OSError, AttributeError):
    pass

import numpy as np
from ultralytics import YOLO


class PeopleDetector:
    """
    YOLOv8-based people detector with temporal smoothing.
    
    Loads a pre-trained YOLOv8 model and filters detections
    to only return bounding boxes for the 'person' class.
    Applies temporal smoothing to suppress flickering detections.
    """

    # COCO class ID for 'person'
    PERSON_CLASS_ID = 0

    def __init__(self, model_path="yolov8m.pt", confidence=0.40, iou_threshold=0.5, device=None):
        """
        Initialize the detector.

        Args:
            model_path: Path to YOLO model weights (will auto-download if not present)
            confidence: Minimum confidence threshold for detections
            iou_threshold: IoU threshold for Non-Maximum Suppression
            device: Inference device ('cpu', 'cuda', '0', etc.). None = auto-detect.
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # Temporal smoothing: suppress detections that appear for < N frames
        self._prev_detections = []
        self._detection_history = []  # List of recent detection arrays
        self._history_len = 3         # Number of frames to consider

        # Load the YOLOv8 model
        print(f"[Detector] Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)

        # Set device (auto-detect GPU if available)
        if device:
            self.device = device
        else:
            self.device = 'cuda' if self._check_cuda() else 'cpu'

        print(f"[Detector] Using device: {self.device}")
        print(f"[Detector] Confidence: {self.confidence}, IoU: {self.iou_threshold}")

    def update_config(self, confidence, iou_threshold):
        """Update detector configuration at runtime."""
        self.confidence = max(0.1, min(1.0, confidence))
        self.iou_threshold = max(0.1, min(1.0, iou_threshold))
        print(f"[Detector] Config updated - Confidence: {self.confidence}, IoU: {self.iou_threshold}")

    def _check_cuda(self):
        """Check if CUDA (GPU) is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _compute_iou(self, box_a, box_b):
        """Compute IoU between two single bboxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        
        return inter / max(union, 1e-6)

    def _temporal_smooth(self, detections):
        """
        Apply temporal smoothing to suppress flickering detections.
        
        A detection is considered stable if a similar detection existed
        in at least 1 of the previous N frames (IoU > 0.3).
        
        This prevents ghost detections from triggering new tracks.
        """
        # Store history
        self._detection_history.append(detections.copy() if len(detections) > 0 else np.empty((0, 5)))
        if len(self._detection_history) > self._history_len:
            self._detection_history.pop(0)
        
        # If not enough history yet, return all detections
        if len(self._detection_history) < 2:
            return detections
        
        if len(detections) == 0:
            return detections
        
        # Check each detection against previous frames
        stable_mask = np.ones(len(detections), dtype=bool)
        
        for i, det in enumerate(detections):
            # Check if this detection has a match in any previous frame
            found_in_history = False
            for prev_dets in self._detection_history[:-1]:  # Exclude current
                if len(prev_dets) == 0:
                    continue
                for prev_det in prev_dets:
                    if self._compute_iou(det[:4], prev_det[:4]) > 0.3:
                        found_in_history = True
                        break
                if found_in_history:
                    break
            
            # If high confidence (>0.7), always keep even without history
            if not found_in_history and det[4] < 0.7:
                stable_mask[i] = False
        
        return detections[stable_mask]

    def detect(self, frame):
        """
        Run person detection on a single frame.
        
        Args:
            frame: OpenCV image (BGR format, numpy array)
            
        Returns:
            List of detections, each as [x1, y1, x2, y2, confidence]
            where (x1, y1) is top-left and (x2, y2) is bottom-right
        """
        # Run inference with YOLO
        # verbose=False suppresses console output each frame
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS_ID],  # Only detect persons
            device=self.device,
            verbose=False
        )

        detections = []

        # Process results
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get bounding box coordinates (xyxy format)
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box

                # Get confidence score
                conf = float(boxes.conf[i].cpu().numpy())

                # Get class ID (should always be 0 = person, but double check)
                cls_id = int(boxes.cls[i].cpu().numpy())
                if cls_id != self.PERSON_CLASS_ID:
                    continue

                detections.append([x1, y1, x2, y2, conf])

        raw_dets = np.array(detections) if detections else np.empty((0, 5))
        
        # Apply temporal smoothing
        return self._temporal_smooth(raw_dets)

    def set_confidence(self, confidence):
        """Update the confidence threshold."""
        self.confidence = max(0.1, min(1.0, confidence))

    def set_iou_threshold(self, iou):
        """Update the IoU threshold for NMS."""
        self.iou_threshold = max(0.1, min(1.0, iou))

    def get_model_info(self):
        """Return basic info about the loaded model."""
        return {
            "model": str(self.model.model_name) if hasattr(self.model, 'model_name') else "YOLOv8",
            "device": self.device,
            "confidence": self.confidence,
            "iou_threshold": self.iou_threshold
        }
