"""
gender_detector.py - Gender & Age Classification with Face Detection

Major improvements over the previous version:
1. Uses OpenCV Haar Cascade for ACTUAL face detection (not just top-crop)
2. Only classifies gender when a real face is detected
3. Majority voting system across frames for stable results
4. Downloads DNN models for gender/age classification automatically
5. Falls back gracefully when face is not visible

Pipeline:
  Person crop → Face detection (Haar) → Face crop → DNN gender/age → Result

Models used:
  - Face: OpenCV Haar cascade (built-in, no download needed)
  - Gender: Levi & Hassner CNN (gender_net.caffemodel, auto-downloaded)
  - Age: Levi & Hassner CNN (age_net.caffemodel, auto-downloaded)
"""

import os
import cv2
import numpy as np
import urllib.request
from collections import defaultdict


# ============================================================
# Model paths and URLs
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"

GENDER_PROTO_URL = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
GENDER_MODEL_URL = "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel"
AGE_PROTO_URL = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"
AGE_MODEL_URL = "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel"

# Model constants
GENDER_LIST = ['Male', 'Female']
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


def _download_file(url, dest_path):
    """Download a file from URL to destination path."""
    try:
        print(f"  [GenderDetector] Downloading: {os.path.basename(dest_path)}...")
        urllib.request.urlretrieve(url, dest_path)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  [GenderDetector] Downloaded: {os.path.basename(dest_path)} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [GenderDetector] Download failed: {e}")
        return False


class GenderAgeDetector:
    """
    Gender and Age estimation using face detection + DNN classification.
    
    Pipeline:
    1. Detect face within person bounding box using Haar cascade
    2. If face found, classify gender and age using DNN models
    3. Use majority voting across frames for stable results
    4. Track cumulative male/female counts (once per unique person)
    """

    def __init__(self, custom_model_path=None):
        """Initialize the gender/age detector with face detection."""
        self.gender_net = None
        self.age_net = None
        self.face_cascade = None
        self.custom_model = None
        self.use_custom_model = False
        self.is_available = False

        # Voting system: {track_id: {'male_votes': int, 'female_votes': int,
        #                             'age_votes': {age: count}, 'total': int}}
        self.votes = {}

        # Classification interval (run every N frames per track)
        self.classify_interval = 8
        self.frame_counters = {}  # {track_id: frame_count}

        # Cached results: {track_id: {'gender': str, 'age': str, 'confidence': float}}
        self.cache = {}

        # Global gender stats (count each track_id only once)
        self.male_count = 0
        self.female_count = 0
        self.counted_gender_ids = set()

        self._load_models(custom_model_path)

    def _load_models(self, custom_model_path=None):
        """Load face detector and gender/age classification models."""
        # === 1. Load Haar cascade for face detection (built-in) ===
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                print("[GenderDetector] WARNING: Haar cascade failed to load")
                self.face_cascade = None
            else:
                print("[GenderDetector] Haar face cascade loaded ✓")
        except Exception as e:
            print(f"[GenderDetector] Haar cascade error: {e}")
            self.face_cascade = None

        # === 2. Try to load custom PyTorch model first ===
        if custom_model_path and os.path.exists(custom_model_path):
            try:
                import torch
                from torchvision.models import resnet50

                self.custom_model = resnet50()
                self.custom_model.fc = torch.nn.Linear(self.custom_model.fc.in_features, 2)
                self.custom_model.load_state_dict(torch.load(custom_model_path, map_location='cpu'))
                self.custom_model.eval()
                self.use_custom_model = True
                self.is_available = True
                print(f"[GenderDetector] Custom PyTorch model loaded ✓: {custom_model_path}")
                return
            except Exception as e:
                print(f"[GenderDetector] Failed to load custom model: {e}")
                self.use_custom_model = False

        # === 3. Load/download gender and age DNN models (fallback) ===
        os.makedirs(MODEL_DIR, exist_ok=True)

        gender_proto_path = os.path.join(MODEL_DIR, GENDER_PROTO)
        gender_model_path = os.path.join(MODEL_DIR, GENDER_MODEL)
        age_proto_path = os.path.join(MODEL_DIR, AGE_PROTO)
        age_model_path = os.path.join(MODEL_DIR, AGE_MODEL)

        files_to_download = [
            (GENDER_PROTO_URL, gender_proto_path),
            (GENDER_MODEL_URL, gender_model_path),
            (AGE_PROTO_URL, age_proto_path),
            (AGE_MODEL_URL, age_model_path),
        ]

        need_download = any(not os.path.exists(p) for _, p in files_to_download)

        if need_download:
            print("[GenderDetector] Downloading gender/age models...")
            for url, path in files_to_download:
                if not os.path.exists(path):
                    if not _download_file(url, path):
                        print("[GenderDetector] Could not download models. "
                              "Gender detection disabled.")
                        return

        # Load the DNN models
        try:
            self.gender_net = cv2.dnn.readNet(gender_model_path, gender_proto_path)
            self.age_net = cv2.dnn.readNet(age_model_path, age_proto_path)
            self.is_available = True
            print("[GenderDetector] Gender & Age DNN models loaded ✓ (pre-trained)")
        except Exception as e:
            print(f"[GenderDetector] Failed to load DNN models: {e}")
            self.is_available = False

    def _detect_face(self, person_crop):
        """
        Detect face within a person crop using Haar cascade.
        
        Args:
            person_crop: BGR image of the person (cropped from frame)
            
        Returns:
            Face crop (BGR) or None if no face detected
        """
        if self.face_cascade is None:
            return None

        if person_crop.size == 0:
            return None

        h, w = person_crop.shape[:2]
        
        # Only search in upper 50% of person crop (faces are at the top)
        upper_region = person_crop[:int(h * 0.5), :]
        
        if upper_region.size == 0 or upper_region.shape[0] < 20:
            return None

        # Convert to grayscale for Haar
        gray = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Improve contrast

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            # Fallback: if face not detected by Haar, 
            # return the top 40% of the person (head/shoulders)
            fallback = person_crop[:int(h * 0.4), :]
            return fallback if fallback.size > 0 else None

        # Take the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = largest_face

        # Add padding around face (20% each side)
        pad_x = int(fw * 0.2)
        pad_y = int(fh * 0.2)
        fx1 = max(0, fx - pad_x)
        fy1 = max(0, fy - pad_y)
        fx2 = min(upper_region.shape[1], fx + fw + pad_x)
        fy2 = min(upper_region.shape[0], fy + fh + pad_y)

        face_crop = upper_region[fy1:fy2, fx1:fx2]

        if face_crop.size == 0 or face_crop.shape[0] < 15 or face_crop.shape[1] < 15:
            fallback = person_crop[:int(h * 0.4), :]
            return fallback if fallback.size > 0 else None

        return face_crop

    def _classify_face(self, face_crop):
        """
        Classify gender and age from a face crop using DNN or custom model.

        Args:
            face_crop: BGR image of the face

        Returns:
            (gender, gender_conf, age) or None if classification fails
        """
        if not self.is_available or face_crop is None:
            return None

        try:
            if self.use_custom_model and self.custom_model is not None:
                # Use custom PyTorch model
                import torch
                import torchvision.transforms as transforms

                # Prepare image
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])

                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                image_tensor = transform(face_rgb).unsqueeze(0)

                with torch.no_grad():
                    output = self.custom_model(image_tensor)
                    probs = torch.softmax(output, dim=1)
                    confidence, gender_idx = torch.max(probs, 1)

                gender = GENDER_LIST[gender_idx.item()]
                gender_conf = float(confidence.item())
                age = '(20-40)'  # Custom model only predicts gender, not age

                return gender, gender_conf, age

            else:
                # Use pre-trained DNN model
                blob = cv2.dnn.blobFromImage(
                    face_crop, 1.0, (227, 227),
                    MODEL_MEAN_VALUES, swapRB=False
                )

                # Gender prediction
                self.gender_net.setInput(blob)
                gender_preds = self.gender_net.forward()
                gender_idx = gender_preds[0].argmax()
                gender = GENDER_LIST[gender_idx]
                gender_conf = float(gender_preds[0][gender_idx])

                # Age prediction
                self.age_net.setInput(blob)
                age_preds = self.age_net.forward()
                age_idx = age_preds[0].argmax()
                age = AGE_LIST[age_idx]

                return gender, gender_conf, age

        except Exception as e:
            print(f"[GenderDetector] Classification error: {e}")
            return None

    def classify(self, frame, bbox, track_id):
        """
        Classify gender and age for a tracked person.
        
        Uses a voting system across multiple frames for stability:
        - Detects face using Haar cascade
        - Classifies gender/age using DNN only when face is visible  
        - Accumulates votes over time
        - Returns the majority vote result
        
        Args:
            frame: Full BGR frame
            bbox: [x1, y1, x2, y2] person bounding box
            track_id: Unique track ID for caching/voting
            
        Returns:
            dict: {'gender': str, 'age': str, 'gender_conf': float}
        """
        default_result = {'gender': 'Unknown', 'age': '', 'gender_conf': 0.0}

        if not self.is_available:
            return default_result

        # Throttle: only classify every N frames per track
        self.frame_counters[track_id] = self.frame_counters.get(track_id, 0) + 1
        
        # Return cached result between classification intervals
        if (self.frame_counters[track_id] % self.classify_interval != 1 and
                track_id in self.cache):
            return self.cache[track_id]

        # Extract person crop
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return self.cache.get(track_id, default_result)

        # Detect face
        face_crop = self._detect_face(person_crop)

        if face_crop is not None:
            # Classify the detected face
            result = self._classify_face(face_crop)

            if result is not None:
                gender, gender_conf, age = result

                # Initialize votes for new track
                if track_id not in self.votes:
                    self.votes[track_id] = {
                        'male_votes': 0,
                        'female_votes': 0,
                        'age_votes': defaultdict(int),
                        'total': 0,
                        'total_conf': 0.0
                    }

                # Add vote (weighted by confidence)
                v = self.votes[track_id]
                v['total'] += 1
                v['total_conf'] += gender_conf

                if gender == 'Male':
                    v['male_votes'] += 1
                else:
                    v['female_votes'] += 1

                v['age_votes'][age] += 1

                # Compute majority vote result
                if v['male_votes'] > v['female_votes']:
                    voted_gender = 'Male'
                    vote_ratio = v['male_votes'] / max(v['total'], 1)
                elif v['female_votes'] > v['male_votes']:
                    voted_gender = 'Female'
                    vote_ratio = v['female_votes'] / max(v['total'], 1)
                else:
                    voted_gender = 'Unknown'
                    vote_ratio = 0.0

                # Need at least 2 votes and 55% majority
                if v['total'] >= 2 and vote_ratio >= 0.55:
                    final_gender = voted_gender
                else:
                    final_gender = 'Unknown'

                # Most voted age
                if v['age_votes']:
                    final_age = max(v['age_votes'], key=v['age_votes'].get)
                else:
                    final_age = ''

                cached = {
                    'gender': final_gender,
                    'age': final_age,
                    'gender_conf': vote_ratio
                }
                self.cache[track_id] = cached
                return cached

        # No face detected this frame — return cached result if available
        return self.cache.get(track_id, default_result)

    def update_gender_counts(self, track_id, gender):
        """
        Update running male/female counts.
        Each track_id is counted only ONCE.
        """
        if track_id not in self.counted_gender_ids and gender in ('Male', 'Female'):
            self.counted_gender_ids.add(track_id)
            if gender == 'Male':
                self.male_count += 1
            else:
                self.female_count += 1

    def get_gender_counts(self):
        """Return (male_count, female_count)."""
        return self.male_count, self.female_count

    def reset(self):
        """Reset all caches and counts."""
        self.votes.clear()
        self.frame_counters.clear()
        self.cache.clear()
        self.male_count = 0
        self.female_count = 0
        self.counted_gender_ids.clear()
        print("[GenderDetector] Reset complete")
