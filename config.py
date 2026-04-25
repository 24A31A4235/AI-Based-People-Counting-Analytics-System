"""
config.py - Centralized Configuration for People Counting System

This module provides easy-to-tune parameters for different scenarios.
Presets are provided for common use cases:
- SPARSE: Low crowd density (office, classroom)
- MEDIUM: Medium crowds with some overlap (mall corridor)
- DENSE: Heavy crowding (markets, events)
"""

from dataclasses import dataclass


@dataclass
class DetectorConfig:
    """YOLOv8 Detection Configuration"""
    confidence: float = 0.40          # Detection confidence threshold (0.0-1.0)
    iou_threshold: float = 0.5        # IoU threshold for NMS (0.0-1.0)
    model_path: str = "yolov8m.pt"   # YOLOv8 model (n/s/m/l/x)


@dataclass
class TrackerConfig:
    """SORT Tracker Configuration"""
    max_age: int = 90                 # Max frames to keep track without detection
    min_hits: int = 3                 # Min detections to confirm a track
    iou_threshold: float = 0.20       # Min IoU for detection-to-track matching


@dataclass
class CounterConfig:
    """Line Crossing Counter Configuration"""
    min_track_age: int = 5            # Min frames before counting valid crossing
    min_settle_frames: int = 7        # Min frames to settle on one side
    min_distance_ratio: float = 0.03  # Min distance from line (ratio of width)
    buffer_zone: int = 35             # Pixel buffer around line
    cooldown_frames: int = 30         # Min frames between same-person counts


@dataclass
class ReIDConfig:
    """Person Re-Identification Configuration"""
    similarity_threshold: float = 0.60    # Cosine similarity threshold
    max_gallery_size: int = 100           # Max persons to remember
    feature_update_rate: float = 0.3      # EMA rate for features


@dataclass
class GenderConfig:
    """Gender/Age Detection Configuration"""
    enabled: bool = True
    model_path: str = ""              # Path to custom trained model (empty = use pre-trained)
    classify_interval: int = 8        # Classify every N frames


@dataclass
class SystemConfig:
    """Complete System Configuration"""
    detector: DetectorConfig
    tracker: TrackerConfig
    counter: CounterConfig
    reid: ReIDConfig
    gender: GenderConfig
    preset_name: str = "MEDIUM"       # Current preset


# ============================================================
# Preset Configurations for Different Scenarios
# ============================================================

def get_sparse_config():
    """
    Sparse scenario (office, classroom, sparse room).
    Low crowd density = can afford stricter detection/matching.
    """
    return SystemConfig(
        detector=DetectorConfig(
            confidence=0.50,          # Stricter - fewer false positives
            iou_threshold=0.5,
            model_path="yolov8m.pt"
        ),
        tracker=TrackerConfig(
            max_age=100,              # Keep tracks longer (less likely to lose people)
            min_hits=2,               # Lower bar for confirming tracks
            iou_threshold=0.25
        ),
        counter=CounterConfig(
            min_track_age=4,
            min_settle_frames=6,
            min_distance_ratio=0.04,
            buffer_zone=30,
            cooldown_frames=25
        ),
        reid=ReIDConfig(
            similarity_threshold=0.63,  # Stricter re-ID
            max_gallery_size=50,
            feature_update_rate=0.2
        ),
        gender=GenderConfig(
            enabled=True,
            classify_interval=10
        ),
        preset_name="SPARSE"
    )


def get_medium_config():
    """
    Medium scenario (mall corridor, typical retail).
    Moderate crowd density = balanced detection/tracking.
    """
    return SystemConfig(
        detector=DetectorConfig(
            confidence=0.40,          # Balanced
            iou_threshold=0.5,
            model_path="yolov8m.pt"
        ),
        tracker=TrackerConfig(
            max_age=90,               # Strong tracking persistence
            min_hits=3,               # Balanced track confirmation
            iou_threshold=0.20        # Flexible matching threshold
        ),
        counter=CounterConfig(
            min_track_age=5,
            min_settle_frames=7,      # Stronger settling requirement
            min_distance_ratio=0.03,
            buffer_zone=35,           # Good buffer for jitter
            cooldown_frames=30
        ),
        reid=ReIDConfig(
            similarity_threshold=0.60,  # Balanced
            max_gallery_size=100,
            feature_update_rate=0.3
        ),
        gender=GenderConfig(
            enabled=True,
            classify_interval=8
        ),
        preset_name="MEDIUM"
    )


def get_dense_config():
    """
    Dense scenario (market, event, crowded area).
    High crowd density = looser matching, more persistence.
    """
    return SystemConfig(
        detector=DetectorConfig(
            confidence=0.35,          # Looser - capture all potential people
            iou_threshold=0.45,       # Stricter NMS
            model_path="yolov8m.pt"   # Use medium model for balance
        ),
        tracker=TrackerConfig(
            max_age=120,              # Much longer tracking (hard to match with overlap)
            min_hits=4,               # Higher threshold for confirmation
            iou_threshold=0.18        # Loose matching (will use centroid dist too)
        ),
        counter=CounterConfig(
            min_track_age=8,
            min_settle_frames=8,      # Very stable settling
            min_distance_ratio=0.05,
            buffer_zone=50,           # Large buffer for jitter
            cooldown_frames=45
        ),
        reid=ReIDConfig(
            similarity_threshold=0.55,  # Loose re-ID (more matches)
            max_gallery_size=200,
            feature_update_rate=0.35
        ),
        gender=GenderConfig(
            enabled=False,            # Disable gender in dense crowds (low facedetect rate)
            classify_interval=12
        ),
        preset_name="DENSE"
    )


# Default config selection
CONFIG_PRESETS = {
    "SPARSE": get_sparse_config,
    "MEDIUM": get_medium_config,
    "DENSE": get_dense_config,
}


def get_config(preset_name="MEDIUM"):
    """
    Get configuration for a preset.

    Args:
        preset_name: "SPARSE", "MEDIUM", or "DENSE"

    Returns:
        SystemConfig instance
    """
    if preset_name not in CONFIG_PRESETS:
        print(f"[Config] Unknown preset '{preset_name}', using MEDIUM")
        preset_name = "MEDIUM"

    config = CONFIG_PRESETS[preset_name]()
    print(f"[Config] Loaded preset: {preset_name}")
    return config


def print_config(config):
    """Pretty-print configuration for debugging."""
    print("\n" + "="*60)
    print(f"Configuration Preset: {config.preset_name}")
    print("="*60)
    print(f"\nDetector:")
    print(f"  Confidence: {config.detector.confidence}")
    print(f"  IoU Threshold: {config.detector.iou_threshold}")
    print(f"  Model: {config.detector.model_path}")

    print(f"\nTracker:")
    print(f"  Max Age: {config.tracker.max_age}")
    print(f"  Min Hits: {config.tracker.min_hits}")
    print(f"  IoU Threshold: {config.tracker.iou_threshold}")

    print(f"\nCounter:")
    print(f"  Min Track Age: {config.counter.min_track_age}")
    print(f"  Min Settle Frames: {config.counter.min_settle_frames}")
    print(f"  Min Distance Ratio: {config.counter.min_distance_ratio}")
    print(f"  Buffer Zone: {config.counter.buffer_zone}px")
    print(f"  Cooldown: {config.counter.cooldown_frames}")

    print(f"\nRe-ID:")
    print(f"  Similarity Threshold: {config.reid.similarity_threshold}")
    print(f"  Gallery Size: {config.reid.max_gallery_size}")

    print(f"\nGender:")
    print(f"  Enabled: {config.gender.enabled}")
    print(f"  Model: {config.gender.model_path if config.gender.model_path else 'pre-trained'}")
    print("="*60 + "\n")
