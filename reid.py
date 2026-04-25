"""
reid.py - Improved Person Re-Identification Module

Solves the "unique person" problem: when the same person leaves
the frame and re-enters, the tracker assigns a NEW track ID.
This module uses appearance-based features (color histograms +
spatial descriptors + LBP texture) to recognize them as the SAME person.

Pipeline:
1. Extract appearance features from each person crop
2. Compare against a gallery of previously seen persons
3. If match found → reuse existing unique ID
4. If no match → assign new unique ID

Feature vector (per person):
- HSV color histograms for upper body and lower body (separated)
- Spatial color grid means (4x2 grid of average colors)
- LBP texture features for clothing texture matching
- Body proportion features (aspect ratio, relative height)
"""

import cv2
import numpy as np
from collections import OrderedDict


class PersonReID:
    """
    Person Re-Identification using color histogram + spatial + texture features.
    
    Maintains a gallery of known person appearances and matches
    new detections against them using cosine similarity.
    
    This is a lightweight, model-free approach that works well
    for short-term re-identification (same session, same camera).
    """

    def __init__(self, similarity_threshold=0.60, max_gallery_size=100,
                 feature_update_rate=0.3):
        """
        Args:
            similarity_threshold: Min cosine similarity to consider a match.
                                  Lower = more matches (risk false positives).
                                  Higher = fewer matches (risk false negatives).
            max_gallery_size: Maximum persons stored in gallery.
            feature_update_rate: EMA rate for updating stored features.
        """
        self.similarity_threshold = similarity_threshold
        self.max_gallery_size = max_gallery_size
        self.feature_update_rate = feature_update_rate

        # Gallery: {unique_id: {'features': np.array, 'track_ids': set, 'last_seen': int,
        #                        'bbox_area': float}}
        self.gallery = OrderedDict()
        self.next_unique_id = 1

        # Mapping: track_id → unique_id  (once assigned, never changes)
        self.track_to_unique = {}

        # Tracks that are pending (need N frames before ReID matching)
        # {track_id: {'frames': int, 'features_list': [np.array, ...]}}
        self.pending_tracks = {}
        self.min_frames_for_reid = 3  # Reduced from 5 for faster matching

        # Global frame counter
        self.frame_count = 0

        print("[ReID] Improved Person Re-ID initialized | "
              f"threshold={similarity_threshold}, gallery_max={max_gallery_size}")

    def _compute_lbp(self, gray_img, grid_x=4, grid_y=4):
        """
        Compute simplified Local Binary Pattern features for texture matching.
        
        This helps distinguish people by their clothing texture,
        not just color (e.g., striped shirt vs solid shirt).
        
        Args:
            gray_img: Grayscale image
            grid_x, grid_y: Grid divisions for spatial LBP
            
        Returns:
            Feature vector from LBP histogram
        """
        h, w = gray_img.shape
        if h < 8 or w < 8:
            return np.zeros(grid_x * grid_y * 8)
        
        features = []
        cell_h = h // grid_y
        cell_w = w // grid_x
        
        for gy in range(grid_y):
            for gx in range(grid_x):
                cell = gray_img[gy * cell_h:(gy + 1) * cell_h,
                                gx * cell_w:(gx + 1) * cell_w]
                
                if cell.size == 0 or cell.shape[0] < 3 or cell.shape[1] < 3:
                    features.append(np.zeros(8))
                    continue
                
                # Simplified LBP: compute gradients in 8 directions
                # using Sobel-like features (fast approximation of LBP)
                gx_grad = cv2.Sobel(cell, cv2.CV_64F, 1, 0, ksize=3)
                gy_grad = cv2.Sobel(cell, cv2.CV_64F, 0, 1, ksize=3)
                
                magnitude = np.sqrt(gx_grad**2 + gy_grad**2)
                angle = np.arctan2(gy_grad, gx_grad)
                
                # Create 8-bin orientation histogram
                bins = np.linspace(-np.pi, np.pi, 9)
                hist, _ = np.histogram(angle, bins=bins, weights=magnitude)
                
                # Normalize
                norm = np.linalg.norm(hist)
                if norm > 1e-6:
                    hist = hist / norm
                
                features.append(hist)
        
        return np.concatenate(features)

    def extract_features(self, frame, bbox):
        """
        Extract robust appearance features from a person crop.
        
        Uses a combination of:
        - HSV color histograms (upper/lower body, split horizontally)
        - Spatial color grid (4x2 grid of mean colors)
        - LBP texture features (clothing texture)
        - Body shape features (aspect ratio)
        
        Args:
            frame: Full BGR frame
            bbox: [x1, y1, x2, y2] person bounding box
            
        Returns:
            Feature vector (np.array) or None if crop is too small
        """
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 15:
            return None

        # Resize to standard size for consistent feature extraction
        crop_resized = cv2.resize(crop, (64, 128))

        # Convert to HSV (more robust to lighting changes than BGR)
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)

        features = []

        # === 1. Color histograms for upper and lower body ===
        # Split at 40% height (upper body = shirt, lower body = pants)
        mid = int(128 * 0.4)
        upper = hsv[:mid, :, :]
        lower = hsv[mid:, :, :]

        for region in [upper, lower]:
            # Hue histogram (16 bins, range 0-180)
            h_hist = cv2.calcHist([region], [0], None, [16], [0, 180])
            h_hist = cv2.normalize(h_hist, h_hist).flatten()

            # Saturation histogram (8 bins)
            s_hist = cv2.calcHist([region], [1], None, [8], [0, 256])
            s_hist = cv2.normalize(s_hist, s_hist).flatten()

            # Value histogram (8 bins)
            v_hist = cv2.calcHist([region], [2], None, [8], [0, 256])
            v_hist = cv2.normalize(v_hist, v_hist).flatten()

            features.extend([h_hist, s_hist, v_hist])

        # === 2. Spatial color grid (4 rows × 2 cols = 8 cells) ===
        for i in range(4):
            for j in range(2):
                cell = crop_resized[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32]
                cell_hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
                cell_mean = cell_hsv.mean(axis=(0, 1)) / np.array([180.0, 255.0, 255.0])
                features.append(cell_mean)

        # === 3. LBP texture features ===
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        lbp_features = self._compute_lbp(gray, grid_x=2, grid_y=4)
        features.append(lbp_features)

        # === 4. Body shape features ===
        bw = x2 - x1
        bh = y2 - y1
        aspect = bw / max(bh, 1)
        area = bw * bh
        features.append(np.array([aspect]))

        return np.concatenate(features)

    def _cosine_similarity(self, a, b):
        """Compute cosine similarity between two feature vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _size_compatible(self, bbox, gallery_area):
        """Check if bbox area is compatible with stored gallery area (within 3x)."""
        if gallery_area is None or gallery_area <= 0:
            return True
        x1, y1, x2, y2 = bbox[:4]
        current_area = (x2 - x1) * (y2 - y1)
        if current_area <= 0:
            return True
        ratio = max(current_area, gallery_area) / max(min(current_area, gallery_area), 1)
        return ratio < 3.0  # Allow up to 3x size difference

    def get_unique_id(self, frame, bbox, track_id):
        """
        Get the true unique person ID for a given track.
        
        Uses a staged approach:
        1. If track already has a unique ID → return it
        2. If track is new, collect features for N frames first
        3. After N frames, match against gallery
        4. If match found → reuse that unique ID
        5. If no match → assign new unique ID
        
        Args:
            frame: Full BGR frame
            bbox: [x1, y1, x2, y2] bounding box
            track_id: Tracker-assigned ID (may change for same person)
            
        Returns:
            (unique_id, is_new_person)
            unique_id: Stable person ID that persists across re-entries
            is_new_person: True only for genuinely new persons
        """
        self.frame_count += 1

        # Already mapped? Return cached result
        if track_id in self.track_to_unique:
            # Periodically update features for this person
            if self.frame_count % 10 == 0:
                self._update_features(frame, bbox, track_id)
            return self.track_to_unique[track_id], False

        # Extract features for this frame
        features = self.extract_features(frame, bbox)
        if features is None:
            # Can't extract → assign new unique ID immediately
            uid = self._assign_new_uid(track_id, None, bbox)
            return uid, True

        # Pending: collect features over several frames before matching
        if track_id not in self.pending_tracks:
            self.pending_tracks[track_id] = {
                'frames': 1,
                'features_list': [features]
            }
            # Return temporary unique ID (will be finalized later)
            uid = self._assign_new_uid(track_id, None, bbox)
            return uid, True

        pending = self.pending_tracks[track_id]
        pending['frames'] += 1
        pending['features_list'].append(features)

        # Not enough frames yet? Keep the temporary ID
        if pending['frames'] < self.min_frames_for_reid:
            if track_id in self.track_to_unique:
                return self.track_to_unique[track_id], False
            uid = self._assign_new_uid(track_id, None, bbox)
            return uid, True

        # === Enough frames collected! Do the real match ===
        # Average the features for robustness
        avg_features = np.mean(pending['features_list'], axis=0)
        del self.pending_tracks[track_id]

        # Search gallery for best match
        best_sim = 0.0
        best_uid = None

        for uid, entry in self.gallery.items():
            # Size compatibility check
            if not self._size_compatible(bbox, entry.get('bbox_area')):
                continue
            
            sim = self._cosine_similarity(avg_features, entry['features'])
            if sim > best_sim:
                best_sim = sim
                best_uid = uid

        if best_sim >= self.similarity_threshold and best_uid is not None:
            # ==============================
            # MATCH FOUND → same person!
            # ==============================
            # Remove the temporary unique ID if one was assigned
            old_temp_uid = self.track_to_unique.get(track_id)
            if old_temp_uid is not None and old_temp_uid in self.gallery:
                # Merge: remove the temporary gallery entry
                del self.gallery[old_temp_uid]

            self.track_to_unique[track_id] = best_uid
            self.gallery[best_uid]['track_ids'].add(track_id)
            self.gallery[best_uid]['last_seen'] = self.frame_count

            # Update features with EMA
            alpha = self.feature_update_rate
            self.gallery[best_uid]['features'] = (
                alpha * avg_features + (1 - alpha) * self.gallery[best_uid]['features']
            )

            print(f"[ReID] Track #{track_id} RE-IDENTIFIED as Person #{best_uid} "
                  f"(similarity={best_sim:.3f})")
            return best_uid, False
        else:
            # ==============================
            # NO MATCH → genuinely new person
            # ==============================
            # Check if we already have a temporary UID
            if track_id in self.track_to_unique:
                uid = self.track_to_unique[track_id]
                # Add features to gallery under this UID
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                bbox_area = (x2 - x1) * (y2 - y1)
                if uid not in self.gallery:
                    self.gallery[uid] = {
                        'features': avg_features,
                        'track_ids': {track_id},
                        'last_seen': self.frame_count,
                        'bbox_area': bbox_area
                    }
                else:
                    self.gallery[uid]['features'] = avg_features
                    self.gallery[uid]['last_seen'] = self.frame_count
                    self.gallery[uid]['bbox_area'] = bbox_area
            else:
                uid = self._assign_new_uid(track_id, avg_features, bbox)

            # Enforce gallery size limit
            while len(self.gallery) > self.max_gallery_size:
                self.gallery.popitem(last=False)

            return uid, True

    def _assign_new_uid(self, track_id, features, bbox=None):
        """Assign a new unique ID and optionally store features."""
        uid = self.next_unique_id
        self.next_unique_id += 1
        self.track_to_unique[track_id] = uid

        if features is not None:
            bbox_area = 0
            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                bbox_area = (x2 - x1) * (y2 - y1)
            
            self.gallery[uid] = {
                'features': features,
                'track_ids': {track_id},
                'last_seen': self.frame_count,
                'bbox_area': bbox_area
            }

        return uid

    def _update_features(self, frame, bbox, track_id):
        """Update stored features for an existing tracked person."""
        if track_id not in self.track_to_unique:
            return

        features = self.extract_features(frame, bbox)
        if features is None:
            return

        uid = self.track_to_unique[track_id]
        if uid in self.gallery:
            alpha = self.feature_update_rate
            self.gallery[uid]['features'] = (
                alpha * features + (1 - alpha) * self.gallery[uid]['features']
            )
            self.gallery[uid]['last_seen'] = self.frame_count
            
            # Update bbox area
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            self.gallery[uid]['bbox_area'] = (x2 - x1) * (y2 - y1)

    def get_unique_count(self):
        """Get total number of truly unique persons identified."""
        return len(self.gallery)

    def reset(self):
        """Reset the entire ReID system."""
        self.gallery.clear()
        self.track_to_unique.clear()
        self.pending_tracks.clear()
        self.next_unique_id = 1
        self.frame_count = 0
        print("[ReID] Reset complete")
