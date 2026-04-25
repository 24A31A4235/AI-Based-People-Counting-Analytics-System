"""
utils.py - Helper functions for the People Counting System.
Contains utility functions for coordinate conversion, drawing,
color management, and file I/O operations.

Updated with cyberpunk-themed video overlays matching the new UI.
"""

import cv2
import numpy as np
import time
import os
import json
from datetime import datetime


# ============================================================
# Color Palette - Cyberpunk theme matching ui.py
# ============================================================
COLORS = {
    'box': (136, 240, 0),            # Cyan-green - default bounding boxes
    'box_crossed': (0, 165, 255),    # Orange - crossed persons
    'box_male': (255, 136, 68),      # Blue tint for male (BGR)
    'box_female': (180, 50, 255),    # Pink/Purple tint for female (BGR)
    'id_text': (240, 240, 240),      # Near white - ID labels
    'id_bg': (30, 30, 50),           # Dark navy - ID label background
    'line': (255, 240, 0),           # Cyan - counting line (BGR)
    'line_crossed': (0, 100, 255),   # Red-orange - line flash
    'in_color': (136, 255, 0),       # Cyan-green - IN count
    'out_color': (110, 0, 255),      # Magenta - OUT count
    'fps_text': (0, 240, 255),       # Gold - FPS display
    'trail': (255, 200, 80),         # Light cyan - movement trail
    'overlay_bg': (14, 14, 30),      # Dark overlay background
}


class FPSCounter:
    """
    Smoothed FPS counter using a rolling average.
    Provides stable FPS readings without rapid fluctuations.
    """
    def __init__(self, avg_count=30):
        self.avg_count = avg_count
        self.frame_times = []
        self.last_time = time.time()

    def tick(self):
        """Record a new frame timestamp and return smoothed FPS."""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        if dt > 0:
            self.frame_times.append(dt)

        # Keep only the last N frame times for rolling average
        if len(self.frame_times) > self.avg_count:
            self.frame_times.pop(0)

        if self.frame_times:
            avg_dt = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_dt if avg_dt > 0 else 0.0
        return 0.0


def compute_centroid(bbox):
    """
    Compute the centroid (center point) of a bounding box.
    
    Args:
        bbox: Tuple of (x1, y1, x2, y2) - top-left and bottom-right corners
        
    Returns:
        Tuple (cx, cy) - center coordinates
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def compute_bottom_center(bbox):
    """
    Compute the bottom-center point of a bounding box.
    More stable for tracking people as feet position changes less than head.
    
    Args:
        bbox: Tuple of (x1, y1, x2, y2)
        
    Returns:
        Tuple (cx, bottom_y)
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int(y2))


def draw_rounded_rect(img, pt1, pt2, color, thickness=1, radius=10, fill=False):
    """
    Draw a rectangle with rounded corners on an image.
    
    Args:
        img: OpenCV image
        pt1: Top-left corner (x, y)
        pt2: Bottom-right corner (x, y)
        color: BGR color tuple
        thickness: Line thickness
        radius: Corner radius
        fill: Whether to fill the rectangle
    """
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

    if fill:
        # Fill the main body
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        # Fill corners
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        # Draw edges
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        # Draw corner arcs
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_fancy_bbox(frame, bbox, track_id, color=None, crossed=False, gender='', age=''):
    """
    Draw a cyberpunk-styled bounding box with corner accents, ID label, and gender/age.
    
    Args:
        frame: OpenCV image
        bbox: (x1, y1, x2, y2) bounding box
        track_id: Unique person ID
        color: Optional override color
        crossed: Whether this person has crossed the line
        gender: Gender classification ('Male', 'Female', or '')
        age: Age group string
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # Color coding by gender with cyberpunk palette
    if color is None:
        if crossed:
            color = COLORS['box_crossed']
        elif gender == 'Male':
            color = COLORS['box_male']
        elif gender == 'Female':
            color = COLORS['box_female']
        else:
            color = COLORS['box']

    # Draw semi-transparent fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)

    # Draw main bounding box (thin)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # Draw corner accents (thicker L-shaped corners) — cyberpunk style
    corner_len = min(25, abs(x2 - x1) // 4, abs(y2 - y1) // 4)
    t = 2  # corner thickness
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, t)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, t)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, t)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, t)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, t)

    # Small dot at centroid
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 3, color, -1)

    # Build label text
    label_parts = [f"#{track_id}"]
    if gender and gender != 'Unknown':
        gender_icon = 'M' if gender == 'Male' else 'F'
        label_parts.append(gender_icon)
    if age:
        label_parts.append(age)
    label = " ".join(label_parts)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Label background with accent color
    label_x1 = x1
    label_y1 = y1 - th - 10
    label_x2 = x1 + tw + 10
    label_y2 = y1

    # Ensure label is within frame bounds
    if label_y1 < 0:
        label_y1 = y2
        label_y2 = y2 + th + 10

    # Draw label bg with slight transparency
    overlay = frame.copy()
    cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    cv2.putText(frame, label, (label_x1 + 5, label_y2 - 5),
                font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, label, (label_x1 + 5, label_y2 - 5),
                font, font_scale, COLORS['id_text'], thickness, cv2.LINE_AA)


def draw_counting_line(frame, line_x, height, flash=False):
    """
    Draw the vertical counting line with cyberpunk styling.
    
    Args:
        frame: OpenCV image
        line_x: X-coordinate of the line
        height: Frame height
        flash: Whether to flash the line (when someone crosses)
    """
    color = COLORS['line_crossed'] if flash else COLORS['line']
    thickness = 3 if flash else 2

    # Draw the main line
    cv2.line(frame, (line_x, 0), (line_x, height), color, thickness)

    # Draw dashed overlay (white highlight)
    dash_len = 20
    gap_len = 12
    y = 0
    while y < height:
        end_y = min(y + dash_len, height)
        cv2.line(frame, (line_x, y), (line_x, end_y), (255, 255, 255), 1)
        y += dash_len + gap_len

    # Draw direction labels
    mid_y = height // 2
    
    # OUT arrow (left)
    arrow_offset = 35
    cv2.arrowedLine(frame, (line_x - 5, mid_y - arrow_offset),
                    (line_x - 40, mid_y - arrow_offset),
                    COLORS['out_color'], 2, tipLength=0.4)
    cv2.putText(frame, "EXIT", (line_x - 60, mid_y - arrow_offset - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['out_color'], 1, cv2.LINE_AA)

    # IN arrow (right)
    cv2.arrowedLine(frame, (line_x + 5, mid_y + arrow_offset),
                    (line_x + 40, mid_y + arrow_offset),
                    COLORS['in_color'], 2, tipLength=0.4)
    cv2.putText(frame, "ENTER", (line_x + 10, mid_y + arrow_offset - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['in_color'], 1, cv2.LINE_AA)


def draw_trail(frame, trail_points, color=None):
    """
    Draw a fading trail showing the movement path of a tracked person.
    
    Args:
        frame: OpenCV image
        trail_points: List of (x, y) positions
        color: Optional trail color
    """
    if len(trail_points) < 2:
        return

    if color is None:
        color = COLORS['trail']

    for i in range(1, len(trail_points)):
        # Fade effect: older points are more transparent
        alpha = i / len(trail_points)
        thickness = max(1, int(alpha * 3))
        pt_color = tuple(int(c * alpha) for c in color)
        cv2.line(frame, trail_points[i - 1], trail_points[i], pt_color, thickness)


def draw_overlay_counters(frame, in_count, out_count, fps=0, male_count=0, female_count=0, total_unique=0):
    """
    Draw minimal cyberpunk HUD overlay on the video frame.
    
    Args:
        frame: OpenCV image
        in_count: Number of people entered
        out_count: Number of people exited
        fps: Current frames per second
        male_count: Total males detected
        female_count: Total females detected
        total_unique: Total unique persons ever tracked
    """
    h, w = frame.shape[:2]
    
    # FPS in top-right
    fps_text = f"FPS {fps:.0f}"
    cv2.putText(frame, fps_text, (w - 100, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['fps_text'], 2, cv2.LINE_AA)

    # Compact IN/OUT counters at top-left with glassmorphic background
    overlay = frame.copy()
    
    # IN box
    cv2.rectangle(overlay, (8, 8), (105, 44), COLORS['overlay_bg'], -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (8, 8), (105, 44), COLORS['in_color'], 1)
    cv2.putText(frame, f"IN {in_count}", (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['in_color'], 2, cv2.LINE_AA)

    overlay = frame.copy()
    # OUT box
    cv2.rectangle(overlay, (112, 8), (220, 44), COLORS['overlay_bg'], -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (112, 8), (220, 44), COLORS['out_color'], 1)
    cv2.putText(frame, f"OUT {out_count}", (120, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['out_color'], 2, cv2.LINE_AA)

    # Gender row (compact)
    if male_count > 0 or female_count > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 50), (220, 80), COLORS['overlay_bg'], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"M:{male_count}  F:{female_count}  U:{total_unique}", (14, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 200), 1, cv2.LINE_AA)


def save_counts_to_file(in_count, out_count, filepath="counts_log.json"):
    """
    Save current counts to a JSON log file with timestamps.
    
    Args:
        in_count: Current IN count
        out_count: Current OUT count
        filepath: Path to the log file
    """
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "in_count": in_count,
        "out_count": out_count,
        "net_inside": in_count - out_count
    }

    # Load existing data or create new
    data = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = []

    data.append(entry)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def resize_frame_aspect(frame, target_width=None, target_height=None):
    """
    Resize a frame while maintaining aspect ratio.
    
    Args:
        frame: OpenCV image
        target_width: Desired width (optional)
        target_height: Desired height (optional)
        
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]

    if target_width and target_height:
        # Fit within both constraints
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h)
    elif target_width:
        scale = target_width / w
    elif target_height:
        scale = target_height / h
    else:
        return frame

    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
