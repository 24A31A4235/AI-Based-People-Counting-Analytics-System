"""
counter.py - ReID-Aware Line Crossing Counter Module

Major accuracy improvements:
1. Uses ReID stable IDs to prevent the same person being double-counted
2. Trajectory-based crossing validation (not just 2-frame side check)
3. Requires person to be "settled" on one side before counting
4. Minimum distance from line requirement (prevents jitter-based counts)
5. Multi-frame direction consistency check
6. Stronger cooldown with distance-based reset
7. Ignores oscillating tracks near the line

Key Logic:
- Maintains a position history for each tracked person
- Accepts an optional stable_id_map (track_id → ReID stable_id)
- A crossing is only counted when:
  a) Person has been tracked for at least MIN_TRACK_AGE frames
  b) Person was clearly on one side (settled for N frames)
  c) Person is now clearly on the other side
  d) The average movement direction is consistent
  e) Person is not in cooldown
  f) The stable_id hasn't already crossed in this direction (dedup!)

FIX (2026-11-22): when a validation check fails, the recorded side is
NOT overwritten, so the crossing is re-checked on later frames. The old
code updated track_sides on every failed check, which permanently lost
slow crossings (the side flips at 35px, but Check 3 needs 52.5px past
the line). Check 4 also now uses the positions from BEFORE the side
flip instead of a trailing window that slides past the line.
"""

import time
import numpy as np
from collections import defaultdict
from datetime import datetime


class LineCrossingCounter:
    """
    High-accuracy people line-crossing counter with ReID deduplication.
    
    Uses trajectory analysis to validate crossings and prevents
    false counts from jitter, oscillation, brief detections,
    AND from the same person being recounted after track ID reassignment.
    """

    # Tuning constants
    MIN_TRACK_AGE = 5          # Min frames a track must exist before counting
    MIN_SETTLE_FRAMES = 7      # Min frames person must be settled on one side
    MIN_DISTANCE_RATIO = 0.03  # Min distance from line as ratio of frame width
    DIRECTION_CHECK_FRAMES = 4 # Frames to check for consistent direction

    def __init__(self, line_x, buffer_zone=35, cooldown_frames=30, frame_width=1280):
        """
        Initialize the counter.
        
        Args:
            line_x: X-coordinate of the vertical counting line
            buffer_zone: Pixel buffer around line to prevent jitter counts
            cooldown_frames: Minimum frames between counts for same person
            frame_width: Frame width (used for distance ratio calculations)
        """
        self.line_x = line_x
        self.buffer_zone = buffer_zone
        self.cooldown_frames = cooldown_frames
        self.frame_width = max(frame_width, 1)

        # Counters
        self.in_count = 0   # People entering (left → right)
        self.out_count = 0  # People exiting  (right → left)

        # Track state: {track_id: 'left' | 'right'}
        # Only stores "settled" side (not 'on_line')
        self.track_sides = {}

        # Position history: {track_id: [(cx, frame_num), ...]}
        self.track_positions = {}

        # Settle counter: {track_id: frames_on_current_side}
        self.track_settle = {}

        # Cooldown tracking: {track_id: frames_remaining}
        self.cooldown = defaultdict(int)

        # Set of IDs that have crossed (for visual indication)
        self.counted_ids = set()

        # ================================================
        # ReID DEDUPLICATION: Track crossings by stable_id
        # ================================================
        # {stable_id: set('IN', 'OUT')} — which directions this person has crossed
        self.stable_id_crossings = defaultdict(set)
        # Cooldown by stable_id (not just track_id)
        self.stable_cooldown = defaultdict(int)

        # Per-track crossing direction log: {track_id: ['IN', 'OUT', ...]}
        self.crossing_history = defaultdict(list)

        # Event log for analysis
        self.events = []

        # Visual feedback
        self.line_flash = False
        self.flash_timer = 0

        # Frame counter
        self.frame_num = 0

    def get_side(self, x):
        """
        Determine which side of the line a point is on.
        
        Returns 'left', 'right', or 'on_line' (within buffer zone).
        """
        if x < self.line_x - self.buffer_zone:
            return 'left'
        elif x > self.line_x + self.buffer_zone:
            return 'right'
        else:
            return 'on_line'

    def _get_min_distance(self):
        """Get minimum distance from line required for a valid crossing."""
        return max(self.buffer_zone * 1.5,
                   self.frame_width * self.MIN_DISTANCE_RATIO)

    def _check_direction_consistency(self, positions, direction):
        """
        Verify that recent positions show consistent movement
        in the expected direction.
        
        Args:
            positions: List of (cx, frame_num) tuples
            direction: 'left_to_right' or 'right_to_left'
            
        Returns:
            True if direction is consistent over recent frames
        """
        if len(positions) < self.DIRECTION_CHECK_FRAMES:
            return False

        recent_x = [p[0] for p in positions[-self.DIRECTION_CHECK_FRAMES:]]

        if direction == 'left_to_right':
            # Most recent positions should be moving rightward
            moves_right = sum(1 for i in range(1, len(recent_x))
                              if recent_x[i] >= recent_x[i - 1])
            return moves_right >= len(recent_x) // 2
        else:
            # Most recent positions should be moving leftward
            moves_left = sum(1 for i in range(1, len(recent_x))
                             if recent_x[i] <= recent_x[i - 1])
            return moves_left >= len(recent_x) // 2

    def _is_settled(self, track_id, side):
        """Check if a track has been settled on a side for enough frames."""
        return self.track_settle.get(track_id, 0) >= self.MIN_SETTLE_FRAMES

    def update(self, tracks, stable_id_map=None):
        """
        Process tracked objects and check for line crossings.
        
        Improved with ReID deduplication:
        - If stable_id_map is provided, crossings are tracked by stable_id
        - Same physical person won't be double-counted even if track ID changes
        
        Args:
            tracks: List of track objects with .id, .get_centroid(), .age attrs
            stable_id_map: Optional dict {track_id: stable_id} from ReID
            
        Returns:
            (in_count, out_count, crossed_ids)
        """
        self.frame_num += 1
        crossed_ids = set()

        # Decrease cooldown timers
        for tid in list(self.cooldown.keys()):
            if self.cooldown[tid] > 0:
                self.cooldown[tid] -= 1
            else:
                del self.cooldown[tid]

        for sid in list(self.stable_cooldown.keys()):
            if self.stable_cooldown[sid] > 0:
                self.stable_cooldown[sid] -= 1
            else:
                del self.stable_cooldown[sid]

        # Update flash timer
        if self.flash_timer > 0:
            self.flash_timer -= 1
        else:
            self.line_flash = False

        # Precompute minimum distance
        min_distance = self._get_min_distance()

        # Process each active track
        active_ids = set()

        for track in tracks:
            track_id = track.id
            active_ids.add(track_id)

            # Resolve stable_id (use ReID if available, else fall back to track_id)
            stable_id = track_id
            if stable_id_map and track_id in stable_id_map:
                stable_id = stable_id_map[track_id]

            # Get centroid position
            cx, cy = track.get_centroid()
            current_side = self.get_side(cx)

            # Record position history
            if track_id not in self.track_positions:
                self.track_positions[track_id] = []
            self.track_positions[track_id].append((cx, self.frame_num))

            # Keep only last 60 positions
            if len(self.track_positions[track_id]) > 60:
                self.track_positions[track_id] = self.track_positions[track_id][-60:]

            # Skip if person is in the buffer zone
            if current_side == 'on_line':
                # Reset settle counter when on line
                self.track_settle[track_id] = 0
                continue

            # Track settle duration
            prev_settled_side = self.track_sides.get(track_id)
            if prev_settled_side == current_side:
                self.track_settle[track_id] = self.track_settle.get(track_id, 0) + 1
            else:
                self.track_settle[track_id] = 1

            # === New track: record initial side ===
            if track_id not in self.track_sides:
                self.track_sides[track_id] = current_side
                continue

            previous_side = self.track_sides[track_id]

            # No change in side
            if previous_side == current_side:
                continue

            # === Potential crossing detected ===

            # Check 1: Track must be old enough
            positions = self.track_positions.get(track_id, [])
            if len(positions) < self.MIN_TRACK_AGE:
                # Do NOT overwrite the recorded side: the crossing is
                # re-checked on later frames once the track is old enough.
                continue

            # Check 2: Must not be in cooldown (both track and stable).
            # The recorded side is left untouched so the crossing is
            # re-checked after the cooldown expires instead of lost.
            if self.cooldown.get(track_id, 0) > 0:
                continue
            if self.stable_cooldown.get(stable_id, 0) > 0:
                continue

            # Check 3: Verify distance from line
            distance_from_line = abs(cx - self.line_x)
            if distance_from_line < min_distance:
                # Too close to line — might be jitter. Do NOT overwrite
                # the recorded side: on later frames the person is
                # further past the line and the crossing is re-checked.
                # (Overwriting here permanently lost slow crossings:
                # the side flips at 35px but this check needs 52.5px.)
                continue

            # Check 4: Verify the person was "settled" on the old side.
            # Use the positions recorded BEFORE the side flip. A trailing
            # window (e.g. the last 7 of positions[:-4]) slides forward
            # as the person walks, so for a slow walker it is already
            # past the line by the time Check 3 passes, and the crossing
            # could never be confirmed. Positions before the flip are
            # the true evidence of where the person came from.
            flip_idx = next((i for i, p in enumerate(positions)
                             if self.get_side(p[0]) == current_side),
                            len(positions))
            old_positions = [p[0] for p in positions[:flip_idx]]
            if old_positions:
                avg_old_x = float(np.mean(old_positions))
                if previous_side == 'left' and avg_old_x >= self.line_x - min_distance:
                    # Wasn't clearly on the left side before
                    continue
                elif previous_side == 'right' and avg_old_x <= self.line_x + min_distance:
                    # Wasn't clearly on the right side before
                    continue

            # Check 5: Direction consistency
            if previous_side == 'left' and current_side == 'right':
                direction = 'left_to_right'
            elif previous_side == 'right' and current_side == 'left':
                direction = 'right_to_left'
            else:
                continue

            if not self._check_direction_consistency(positions, direction):
                # Movement is not consistent — might be oscillating.
                # Recorded side is left untouched for re-checking.
                continue

            # ========================================
            # Check 6 (NEW): ReID deduplication
            # Prevent same physical person from being counted twice
            # ========================================
            dir_label = "IN" if direction == 'left_to_right' else "OUT"
            
            if dir_label in self.stable_id_crossings.get(stable_id, set()):
                # This stable person already crossed in this direction!
                # Don't count again — just update state
                self.track_sides[track_id] = current_side
                self.cooldown[track_id] = self.cooldown_frames
                print(f"[Counter] DEDUP: Person #{stable_id} (track #{track_id}) "
                      f"already counted {dir_label} — skipping")
                continue

            # ========================================
            # ✅ VALID CROSSING CONFIRMED!
            # ========================================

            if direction == 'left_to_right':
                self.in_count += 1
            else:
                self.out_count += 1

            crossed_ids.add(track_id)

            # Record the crossing
            self.track_sides[track_id] = current_side
            self.cooldown[track_id] = self.cooldown_frames
            self.stable_cooldown[stable_id] = self.cooldown_frames
            self.counted_ids.add(track_id)
            self.crossing_history[track_id].append(dir_label)
            
            # Record stable_id crossing direction for dedup
            self.stable_id_crossings[stable_id].add(dir_label)

            # Reset position history after crossing (clean slate)
            self.track_positions[track_id] = [(cx, self.frame_num)]
            self.track_settle[track_id] = 1

            # Visual feedback
            self.line_flash = True
            self.flash_timer = 8

            # Log the event
            event = {
                'track_id': track_id,
                'stable_id': stable_id,
                'direction': dir_label,
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'position': (cx, cy)
            }
            self.events.append(event)
            print(f"[Counter] Person #{stable_id} (track #{track_id}) crossed {dir_label} | "
                  f"Total IN: {self.in_count}, OUT: {self.out_count}")

        # Clean up stale tracks (gone for more than 120 frames)
        stale_ids = set(self.track_positions.keys()) - active_ids
        for stale_id in stale_ids:
            positions = self.track_positions.get(stale_id, [])
            if positions and (self.frame_num - positions[-1][1]) > 120:
                self.track_positions.pop(stale_id, None)
                self.track_settle.pop(stale_id, None)
                # Keep track_sides and counted_ids for persistent state

        return self.in_count, self.out_count, crossed_ids

    def set_line_position(self, line_x):
        """Update the counting line position."""
        self.line_x = line_x

    def set_frame_width(self, width):
        """Update the frame width (used for distance calculations)."""
        self.frame_width = max(width, 1)

    def reset(self):
        """Reset all counters and tracking data."""
        self.in_count = 0
        self.out_count = 0
        self.track_sides.clear()
        self.track_positions.clear()
        self.track_settle.clear()
        self.cooldown.clear()
        self.stable_cooldown.clear()
        self.counted_ids.clear()
        self.crossing_history.clear()
        self.stable_id_crossings.clear()
        self.events.clear()
        self.line_flash = False
        self.flash_timer = 0
        self.frame_num = 0
        print("[Counter] Counters reset")

    def get_counts(self):
        """Get current IN and OUT counts."""
        return self.in_count, self.out_count

    def get_net_count(self):
        """Get the net count of people currently inside."""
        return max(0, self.in_count - self.out_count)

    def get_events(self):
        """Get the event log."""
        return self.events.copy()

    def is_flashing(self):
        """Check if the line should be flashing (visual feedback)."""
        return self.line_flash

    def has_crossed(self, track_id):
        """Check if a specific track ID has ever crossed the line."""
        return track_id in self.counted_ids
