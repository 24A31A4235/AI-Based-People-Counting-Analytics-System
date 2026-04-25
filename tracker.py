"""
tracker.py - Multi-Object Tracking Module (Improved)
Implements a SORT-inspired tracker using Kalman filtering,
Hungarian algorithm, and HYBRID IoU + centroid-distance association.

Key Features:
- Kalman filter for smooth position prediction
- Hungarian (linear assignment) for detection-to-track matching
- **Cascaded matching: IoU first, then centroid-distance fallback**
- Track lifecycle management (creation, update, deletion)
- Trail history for visualization
- Longer track persistence to prevent ID fragmentation
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict


class KalmanBoxTracker:
    """
    Kalman Filter tracker for a single bounding box.
    
    State vector: [x_center, y_center, area, aspect_ratio, vx, vy, va]
    - Tracks center position, area, and aspect ratio
    - Predicts velocity for smooth motion estimation
    """
    
    _id_counter = 0  # Class-level ID counter for unique IDs
    
    def __init__(self, bbox):
        """
        Initialize a new tracker with a bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
        """
        # Assign unique ID
        KalmanBoxTracker._id_counter += 1
        self.id = KalmanBoxTracker._id_counter
        
        # Convert bbox to state: [cx, cy, area, aspect_ratio]
        self.state = self._bbox_to_state(bbox)
        
        # Velocity estimates
        self.velocity = np.zeros(3)  # [vx, vy, va]
        
        # Track management
        self.hits = 1             # Number of successful updates
        self.age = 0              # Total frames since creation
        self.time_since_update = 0  # Frames since last detection match
        self.hit_streak = 1       # Consecutive successful updates
        
        # Store raw bbox for accurate drawing
        self.bbox = np.array(bbox[:4], dtype=float)
        
        # Movement trail (history of centroid positions)
        self.trail = []
        cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
        self.trail.append((cx, cy))
        
        # Maximum trail length
        self.max_trail_length = 30
        
        # Previous centroid for direction detection
        self.prev_centroid_x = cx
    
    def _bbox_to_state(self, bbox):
        """Convert [x1, y1, x2, y2] to [cx, cy, area, aspect_ratio]."""
        x1, y1, x2, y2 = bbox[:4]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        area = w * h
        aspect = w / max(h, 1e-6)
        return np.array([cx, cy, area, aspect])
    
    def _state_to_bbox(self, state):
        """Convert [cx, cy, area, aspect_ratio] back to [x1, y1, x2, y2]."""
        cx, cy, area, aspect = state
        area = max(area, 1)
        w = np.sqrt(area * aspect)
        h = area / max(w, 1e-6)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])
    
    def predict(self):
        """
        Predict the next state using simple motion model.
        Returns predicted bounding box.
        """
        # Apply velocity to state
        self.state[:3] += self.velocity
        
        # Ensure area stays positive
        self.state[2] = max(self.state[2], 1)
        
        self.age += 1
        self.time_since_update += 1
        
        # Decay hit_streak if not updated
        if self.time_since_update > 1:
            self.hit_streak = 0
        
        # Update bbox from predicted state
        self.bbox = self._state_to_bbox(self.state)
        return self.bbox.copy()
    
    def update(self, bbox):
        """
        Update the tracker with a new matched detection.
        
        Args:
            bbox: [x1, y1, x2, y2] or [x1, y1, x2, y2, conf]
        """
        new_state = self._bbox_to_state(bbox)
        
        # Update velocity with smoothing (exponential moving average)
        alpha = 0.3  # Smoothing factor
        new_velocity = new_state[:3] - self.state[:3]
        self.velocity = alpha * new_velocity + (1 - alpha) * self.velocity
        
        # Update state with measurement (weighted blend for stability)
        beta = 0.7  # Measurement weight (higher = trust detection more)
        self.state = beta * new_state + (1 - beta) * self.state
        
        # Update raw bbox
        self.bbox = np.array(bbox[:4], dtype=float)
        
        # Update trail
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        
        # Store previous x for direction detection
        if self.trail:
            self.prev_centroid_x = self.trail[-1][0]
        
        self.trail.append((cx, cy))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)
        
        # Update track management
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
    
    def get_centroid(self):
        """Get current centroid position."""
        return (int(self.state[0]), int(self.state[1]))
    
    def get_bbox(self):
        """Get current bounding box as [x1, y1, x2, y2]."""
        return self.bbox.copy()


def iou_batch(bb_test, bb_gt):
    """
    Compute IoU (Intersection over Union) between two sets of bounding boxes.
    
    Args:
        bb_test: (N, 4) array of bounding boxes
        bb_gt: (M, 4) array of bounding boxes
        
    Returns:
        (N, M) IoU matrix
    """
    bb_test = np.expand_dims(bb_test, 1)  # (N, 1, 4)
    bb_gt = np.expand_dims(bb_gt, 0)      # (1, M, 4)
    
    # Intersection coordinates
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    # Intersection area
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h
    
    # Areas of each box
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    
    # Union
    union = area_test + area_gt - intersection
    
    return intersection / np.maximum(union, 1e-6)


def centroid_distance_batch(bb_test, bb_gt):
    """
    Compute normalized centroid distance between two sets of bounding boxes.
    Distance is normalized by the average diagonal of the bounding boxes.
    
    Args:
        bb_test: (N, 4) array of bounding boxes
        bb_gt: (M, 4) array of bounding boxes
        
    Returns:
        (N, M) distance matrix (lower = closer)
    """
    # Centroids of test boxes
    cx_test = (bb_test[:, 0] + bb_test[:, 2]) / 2  # (N,)
    cy_test = (bb_test[:, 1] + bb_test[:, 3]) / 2  # (N,)
    
    # Centroids of gt boxes
    cx_gt = (bb_gt[:, 0] + bb_gt[:, 2]) / 2  # (M,)
    cy_gt = (bb_gt[:, 1] + bb_gt[:, 3]) / 2  # (M,)
    
    # Euclidean distance matrix (N, M)
    dx = cx_test[:, None] - cx_gt[None, :]
    dy = cy_test[:, None] - cy_gt[None, :]
    distances = np.sqrt(dx**2 + dy**2)
    
    # Normalize by average bbox diagonal for scale invariance
    diag_test = np.sqrt(
        (bb_test[:, 2] - bb_test[:, 0])**2 + (bb_test[:, 3] - bb_test[:, 1])**2
    )  # (N,)
    diag_gt = np.sqrt(
        (bb_gt[:, 2] - bb_gt[:, 0])**2 + (bb_gt[:, 3] - bb_gt[:, 1])**2
    )  # (M,)
    
    avg_diag = (diag_test[:, None] + diag_gt[None, :]) / 2
    avg_diag = np.maximum(avg_diag, 1e-6)
    
    return distances / avg_diag


class SORTTracker:
    """
    Improved SORT (Simple Online and Realtime Tracking) implementation.

    Uses CASCADED association:
    1. First pass: IoU-based matching (Hungarian algorithm)
    2. Second pass: Centroid-distance matching for unmatched pairs
    
    This dramatically reduces track ID fragmentation and prevents
    the same person from getting a new ID due to brief detection gaps.

    Attributes:
        max_age: Maximum frames a track can survive without updates
        min_hits: Minimum hits before a track is reported
        iou_threshold: Minimum IoU for detection-to-track association
        max_centroid_dist: Maximum normalized centroid distance for fallback matching
    """

    def __init__(self, max_age=90, min_hits=3, iou_threshold=0.20,
                 max_centroid_dist=1.5):
        """
        Initialize the SORT tracker.

        Args:
            max_age: Max frames to keep a track without detections (90 = ~3 sec at 30fps)
            min_hits: Min detections before track is confirmed
            iou_threshold: Min IoU for matching detections to tracks
            max_centroid_dist: Max normalized centroid distance for fallback matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_centroid_dist = max_centroid_dist
        self.tracks = []
        self.frame_count = 0

        # Reset ID counter
        KalmanBoxTracker._id_counter = 0

        print(f"[Tracker] Improved SORT initialized - max_age={max_age}, "
              f"min_hits={min_hits}, iou_threshold={iou_threshold}, "
              f"max_centroid_dist={max_centroid_dist}")

    def set_parameters(self, max_age, min_hits, iou_threshold):
        """Update tracker parameters at runtime."""
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        print(f"[Tracker] Parameters updated - max_age={max_age}, "
              f"min_hits={min_hits}, iou_threshold={iou_threshold}")
    
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Improved pipeline with cascaded matching:
        1. Predict new positions for all existing tracks
        2. Stage 1: IoU-based association (primary)
        3. Stage 2: Centroid-distance association (fallback for unmatched)
        4. Update matched tracks
        5. Create new tracks for unmatched detections
        6. Remove dead tracks
        
        Args:
            detections: numpy array (N, 5) of [x1, y1, x2, y2, confidence]
                        or (N, 4) of [x1, y1, x2, y2]
                        or empty array if no detections
            
        Returns:
            List of active tracks (KalmanBoxTracker objects)
            Each track has: .id, .bbox, .trail, .get_centroid(), etc.
        """
        self.frame_count += 1
        
        # ============================================
        # Step 1: Predict new positions for all tracks
        # ============================================
        for track in self.tracks:
            track.predict()
        
        # ============================================
        # Step 2+3: CASCADED Association
        # ============================================
        if len(detections) == 0:
            detections = np.empty((0, 5))
        elif detections.ndim == 1:
            detections = detections.reshape(1, -1)
        
        matched, unmatched_dets, unmatched_trks = self._cascaded_associate(
            detections, self.tracks
        )
        
        # ============================================
        # Step 4: Update matched tracks with detections
        # ============================================
        for det_idx, trk_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx])
        
        # ============================================
        # Step 5: Create new tracks for unmatched detections
        # ============================================
        for det_idx in unmatched_dets:
            new_track = KalmanBoxTracker(detections[det_idx])
            self.tracks.append(new_track)
        
        # ============================================
        # Step 6: Remove dead tracks
        # ============================================
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]
        
        # ============================================
        # Return confirmed tracks (enough hits to be reliable)
        # ============================================
        active_tracks = []
        for track in self.tracks:
            # Report track if it has enough hits OR has been recently updated
            # During early frames, be more lenient
            if (track.hit_streak >= self.min_hits or 
                self.frame_count <= self.min_hits):
                if track.time_since_update == 0:  # Only report if recently updated
                    active_tracks.append(track)
        
        return active_tracks
    
    def _cascaded_associate(self, detections, tracks):
        """
        Cascaded association: IoU first, then centroid-distance fallback.
        
        Stage 1: Match using IoU (bounding box overlap) — works for slow movement
        Stage 2: For remaining unmatched, try centroid distance — works for fast movement
        
        This two-stage approach dramatically reduces track fragmentation.
        
        Args:
            detections: (N, 5) array of detections
            tracks: List of KalmanBoxTracker objects
            
        Returns:
            matched: List of (detection_idx, track_idx) pairs
            unmatched_dets: List of unmatched detection indices
            unmatched_trks: List of unmatched track indices
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        # Get predicted bounding boxes from tracks
        trk_bboxes = np.array([t.get_bbox() for t in tracks])
        det_bboxes = detections[:, :4]
        
        # ========================================
        # STAGE 1: IoU-based matching (primary)
        # ========================================
        iou_matrix = iou_batch(det_bboxes, trk_bboxes)
        cost_matrix = 1 - iou_matrix
        cost_matrix[iou_matrix < self.iou_threshold] = 1e5
        
        matched_s1 = []
        unmatched_dets_s1 = list(range(len(detections)))
        unmatched_trks_s1 = list(range(len(tracks)))
        
        if min(cost_matrix.shape) > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] >= self.iou_threshold:
                    matched_s1.append((row, col))
                    if row in unmatched_dets_s1:
                        unmatched_dets_s1.remove(row)
                    if col in unmatched_trks_s1:
                        unmatched_trks_s1.remove(col)
        
        # ========================================
        # STAGE 2: Centroid-distance fallback
        # For remaining unmatched detections & tracks
        # ========================================
        if len(unmatched_dets_s1) > 0 and len(unmatched_trks_s1) > 0:
            rem_det_bboxes = det_bboxes[unmatched_dets_s1]
            rem_trk_bboxes = trk_bboxes[unmatched_trks_s1]
            
            dist_matrix = centroid_distance_batch(rem_det_bboxes, rem_trk_bboxes)
            
            # Only allow matches within max_centroid_dist
            dist_cost = dist_matrix.copy()
            dist_cost[dist_matrix > self.max_centroid_dist] = 1e5
            
            if min(dist_cost.shape) > 0:
                row_indices, col_indices = linear_sum_assignment(dist_cost)
                
                matched_s2_dets = []
                matched_s2_trks = []
                
                for row, col in zip(row_indices, col_indices):
                    if dist_matrix[row, col] <= self.max_centroid_dist:
                        # Map back to original indices
                        orig_det = unmatched_dets_s1[row]
                        orig_trk = unmatched_trks_s1[col]
                        matched_s1.append((orig_det, orig_trk))
                        matched_s2_dets.append(orig_det)
                        matched_s2_trks.append(orig_trk)
                
                # Remove from unmatched lists
                for d in matched_s2_dets:
                    unmatched_dets_s1.remove(d)
                for t in matched_s2_trks:
                    unmatched_trks_s1.remove(t)
        
        return matched_s1, unmatched_dets_s1, unmatched_trks_s1
    
    def reset(self):
        """Reset all tracks and counters."""
        self.tracks = []
        self.frame_count = 0
        KalmanBoxTracker._id_counter = 0
        print("[Tracker] Reset complete")
    
    def get_all_tracks(self):
        """Get all tracks including unconfirmed ones."""
        return self.tracks
