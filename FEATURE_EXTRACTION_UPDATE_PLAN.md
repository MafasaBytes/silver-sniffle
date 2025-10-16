# Feature Extraction Update Plan
## Adding Face Landmarks (177 → 645 features)

**Date:** 2025-10-15
**Status:** Ready to implement
**Goal:** Add MediaPipe face landmarks to match research proposal

---

## Current vs Target Features

### Current (177 features):
- **YOLOv8-Pose:** 17 keypoints × 3 (x,y,conf) = 51 features
- **MediaPipe Hands:** 21 keypoints × 2 hands × 3 (x,y,z) = 126 features
- **Total:** 177 features

### Target (645 features):
- **YOLOv8-Pose:** 17 keypoints × 3 = 51 features (unchanged)
- **MediaPipe Face:** 468 keypoints × 3 (x,y,z) = 1404 features → Use 156 features (52 key face points)
- **MediaPipe Hands:** 21 keypoints × 2 hands × 3 = 126 features (unchanged)
- **Total:** 51 + 156 + 126 = 333 features (reduced face for efficiency)

**Alternative (Full face):**
- YOLOv8: 51 + MediaPipe Face: 1404 + Hands: 126 = **1,581 features**
- Too many features → slow training, memory issues

**Compromise:** Use MediaPipe Holistic with face mesh subset (52 keypoints = 156 features)

---

## Implementation Changes

### File to Modify:
`src/extract_features_yolov8_gpu.py`

### Key Changes:

#### 1. Replace MediaPipe Hands with MediaPipe Holistic

**Current (line 187-199):**
```python
self.mp_hands = mp.solutions.hands
self.hands = self.mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    model_complexity=1
)
```

**New:**
```python
self.mp_holistic = mp.solutions.holistic
self.holistic = self.mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    refine_face_landmarks=False  # Use 468 points (not 478)
)
```

#### 2. Update Feature Array Size

**Current (line 226-229):**
```python
if self.use_hands:
    features = np.zeros((batch_size, 177), dtype=np.float32)  # 51 body + 126 hands
else:
    features = np.zeros((batch_size, 51), dtype=np.float32)
```

**New:**
```python
if self.use_holistic:
    # 51 (body) + 156 (face subset) + 126 (hands) = 333
    features = np.zeros((batch_size, 333), dtype=np.float32)
else:
    features = np.zeros((batch_size, 51), dtype=np.float32)
```

#### 3. Extract Face + Hand Landmarks

**Current (line 250-265):**
```python
if self.use_hands:
    mp_start = time.time()
    for i, frame in enumerate(frames):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(frame_rgb)

        if hand_results.multi_hand_landmarks:
            hand_features = np.zeros(126, dtype=np.float32)
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks[:2]):
                for lm_idx, landmark in enumerate(hand_landmarks.landmark):
                    base_idx = hand_idx * 63 + lm_idx * 3
                    if base_idx + 3 <= 126:
                        hand_features[base_idx:base_idx+3] = [landmark.x, landmark.y, landmark.z]
            features[i, 51:] = hand_features
```

**New:**
```python
if self.use_holistic:
    mp_start = time.time()
    for i, frame in enumerate(frames):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)

        # Extract face landmarks (52 key points = 156 features)
        if results.face_landmarks:
            face_indices = [
                # Contour (17 points)
                0, 17, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
                # Eyes (12 points)
                33, 133, 160, 159, 158, 157, 173,  # Left eye
                362, 263, 387, 386, 385, 384, 398,  # Right eye
                # Eyebrows (8 points)
                70, 63, 105, 66, 107,  # Left eyebrow
                300, 293, 334, 296, 336,  # Right eyebrow
                # Nose (5 points)
                1, 2, 98, 327, 4,
                # Mouth (10 points)
                61, 291, 0, 17, 269, 405, 314, 17, 84, 181
            ][:52]  # Use first 52 for exactly 156 features

            face_features = np.zeros(156, dtype=np.float32)
            for idx, lm_idx in enumerate(face_indices):
                if lm_idx < len(results.face_landmarks.landmark):
                    lm = results.face_landmarks.landmark[lm_idx]
                    face_features[idx*3:idx*3+3] = [lm.x, lm.y, lm.z]
            features[i, 51:207] = face_features

        # Extract hand landmarks (126 features)
        if results.left_hand_landmarks:
            for lm_idx, lm in enumerate(results.left_hand_landmarks.landmark):
                if lm_idx < 21:
                    base_idx = 207 + lm_idx * 3
                    features[i, base_idx:base_idx+3] = [lm.x, lm.y, lm.z]

        if results.right_hand_landmarks:
            for lm_idx, lm in enumerate(results.right_hand_landmarks.landmark):
                if lm_idx < 21:
                    base_idx = 207 + 63 + lm_idx * 3
                    features[i, base_idx:base_idx+3] = [lm.x, lm.y, lm.z]
```

#### 4. Update Variable Names Throughout

Replace all instances of:
- `use_hands` → `use_holistic`
- `self.hands` → `self.holistic`
- `self.mp_hands` → `self.mp_holistic`

---

## Additional Files to Update

### 1. Model Architecture
**File:** `src/models/bilstm.py`

**Change:**
```python
# Line 19: Default input_dim
input_dim: int = 333,  # Changed from 177
```

### 2. Training Script
**File:** `src/train_bilstm.py`

**Change:**
```python
# Line 275: Default input_dim
parser.add_argument('--input-dim', type=int, default=333)  # Changed from 177
```

### 3. Dataset Class
**File:** `src/phoenix_dataset.py`

No changes needed - will automatically adapt to new feature dimension.

---

## Execution Plan

### Step 1: Backup Current Features (Optional)
```bash
# Optional: Keep old 177-feature extraction as backup
cp -r data/processed data/processed_177features_backup
```

### Step 2: Modify Extraction Script
```bash
# Edit: src/extract_features_yolov8_gpu.py
# Make changes described above
```

### Step 3: Re-extract Features
```bash
# Delete old features
rm -rf data/processed/train/*
rm -rf data/processed/dev/*
rm -rf data/processed/test/*

# Extract with new holistic features
python src/extract_features_yolov8_gpu.py --split all --batch-size 8 --workers 2

# Expected time: ~6-8 hours (overnight)
```

### Step 4: Normalize New Features
```bash
# Update normalization script for 333 features
python scripts/normalize_features.py
```

### Step 5: Update Model and Retrain
```bash
# Train with 333-dimensional features
python src/train_bilstm.py \
  --input-dim 333 \
  --checkpoint-dir models/bilstm_holistic \
  --learning-rate 0.001 \
  --num-epochs 50
```

---

## Expected Outcomes

### Feature Discriminative Power
With face landmarks added:
- **Better grammar recognition:** Questions, negations, emphasis
- **Better sign disambiguation:** Same hand shape, different meanings
- **Improved baseline WER:** From 92.97% → 35-50% (expected)

### Training Improvements
- Frame accuracy should reach >20% by epoch 10
- Model should predict varied tokens (not just blanks)
- Overfitting test should achieve <20% blank ratio

### Timeline Impact
- **Day 1:** Modify script (1 hour)
- **Night 1:** Run extraction overnight (6-8 hours)
- **Day 2:** Normalize + retrain (30 minutes)
- **Total delay:** <1 work day (if run overnight)

---

## Risk Mitigation

### If Extraction Fails:
- Fall back to 177 features
- Document limitation in thesis
- Proceed with Phase II using current features

### If Training Still Fails:
- Consult with advisor
- Consider alternative baselines (HMM, framewise classification)
- Reassess research approach

### If Time Runs Out:
- Use 177-feature baseline as-is
- Document as "simplified baseline"
- Focus effort on Phase II (end-to-end learning)

---

## Success Criteria

After re-extraction and retraining, we should see:
- ✓ Frame accuracy >20% by epoch 10
- ✓ WER <50% on dev set
- ✓ Model predicts >200 unique tokens (not just blanks)
- ✓ Overfitting test: <30% blank ratio

If these criteria are met, proceed to Phase II. If not, escalate to advisor.
