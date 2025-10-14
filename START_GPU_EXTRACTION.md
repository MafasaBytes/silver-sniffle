# Start GPU Extraction - Quick Guide

**Date**: October 14, 2025
**Status**: READY TO RUN
**Script**: `src/extract_features_yolov8_gpu.py`

---

## ✅ What's Ready

1. **YOLOv8 GPU** - Already installed (`ultralytics`)
2. **MediaPipe** - Already installed
3. **CUDA 12.6** - Working with RTX 4070
4. **Extraction script** - Complete with resume capability

---

## 🎯 Quick Start Commands

### Option 1: TEST FIRST (5-10 minutes) - RECOMMENDED
```bash
# Test on first 50 sequences to verify everything works
python src/extract_features_yolov8_gpu.py --split train --checkpoint-interval 50 &
# Then kill it after you see it working (Ctrl+C)
```

### Option 2: FASTEST (1.5-2 hours) - Body pose only
```bash
python src/extract_features_yolov8_gpu.py --model yolov8n-pose.pt --no-hands --split train
```

### Option 3: COMPLETE (3-4 hours) - Body + hands - RECOMMENDED
```bash
python src/extract_features_yolov8_gpu.py --model yolov8m-pose.pt --split train
```

### Option 4: ALL SPLITS (3.5-4.5 hours)
```bash
python src/extract_features_yolov8_gpu.py --model yolov8m-pose.pt --split all
```

---

## 📊 Expected Performance

| Model | Hands | FPS | Time (4,376 samples) | Quality |
|-------|-------|-----|----------------------|---------|
| yolov8n | No | ~85 | 1.5 hours | Good |
| yolov8m | No | ~60 | 2 hours | Better |
| yolov8n | Yes | ~40 | 3 hours | Good |
| **yolov8m** | **Yes** | **~30** | **3-4 hours** | **Best** ✓ |

---

## 🔍 How to Test First

### 1. Quick GPU Check
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output**:
```
CUDA: True
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
```

### 2. Test YOLOv8 Model Loading
```bash
python -c "from ultralytics import YOLO; model = YOLO('yolov8m-pose.pt'); model.to('cuda'); print('YOLOv8 ready on GPU!')"
```

**Expected**: Model downloads (~6MB) if not cached, then prints "YOLOv8 ready on GPU!"

### 3. Test Extraction on One Sample
```python
# Create test script
python -c "
from src.extract_features_yolov8_gpu import YOLOv8GPUExtractor
from pathlib import Path

extractor = YOLOv8GPUExtractor(yolo_model='yolov8m-pose.pt', use_hands=True)

# Test on first train sample
data_root = Path('data/raw_data/phoenix-2014-signerindependent-SI5')
test_folder = data_root / 'features' / 'fullFrame-210x260px' / 'train' / '01April_2010_Thursday_heute_default-0'

if test_folder.exists():
    result = extractor.process_video_sequence(test_folder)
    print(f'Success! Extracted {result[\"num_frames\"]} frames at {result[\"fps\"]:.1f} FPS')
    print(f'Feature shape: {result[\"feature_shape\"]}')
else:
    print(f'Test folder not found: {test_folder}')
"
```

---

## 📁 Output Structure

```
data/processed/train/
├── *.npy                                # Feature files (177 dims per frame)
├── extraction_metrics_yolov8.csv        # Per-sample statistics
├── extraction_summary_yolov8.json       # Overall performance
├── checkpoint_yolov8_100.json           # Checkpoints every 100 samples
├── checkpoint_yolov8_200.json
├── ...
└── failed_samples_yolov8.json           # Any failures (hopefully empty)
```

---

## 🖥️ Monitoring Progress

### Terminal 1: Run Extraction
```bash
cd C:\Users\Masia\OneDrive\Desktop\sign-language-recognition
python src/extract_features_yolov8_gpu.py --model yolov8m-pose.pt --split train
```

### Terminal 2: Monitor GPU
```bash
# Option A: nvidia-smi
nvidia-smi -l 1

# Option B: PowerShell
while ($true) { cls; nvidia-smi; Sleep 1 }
```

### Terminal 3: Check Progress
```bash
# Count extracted files
(Get-ChildItem data\processed\train\*.npy).Count

# Watch it increase
while ($true) { cls; Write-Host "Extracted: $((Get-ChildItem data\processed\train\*.npy).Count)/4376"; Sleep 5 }
```

---

## 🎯 Feature Dimensions

### YOLOv8-Pose (Body)
- 17 keypoints × 3 values (x, y, confidence) = **51 dimensions**
- Keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles

### MediaPipe Hands (Optional)
- 2 hands × 21 landmarks × 3 coords (x, y, z) = **126 dimensions**
- Total landmarks per hand: thumb (5), index (4), middle (4), ring (4), pinky (4)

### Total Features
- **Body only**: 51 dimensions
- **Body + hands**: 51 + 126 = **177 dimensions**

---

## ⚠️ Troubleshooting

### Issue: CUDA Out of Memory
**Symptom**: RuntimeError: CUDA out of memory
**Solution**:
```bash
# Use smaller model
python src/extract_features_yolov8_gpu.py --model yolov8n-pose.pt --split train

# Or disable hands
python src/extract_features_yolov8_gpu.py --model yolov8m-pose.pt --no-hands --split train
```

### Issue: Model Download Stuck
**Symptom**: Hangs at "Downloading yolov8m-pose.pt"
**Solution**:
```bash
# Pre-download model
python -c "from ultralytics import YOLO; YOLO('yolov8m-pose.pt')"

# Then run extraction
```

### Issue: Slow Extraction (<10 FPS)
**Check**:
- GPU utilization: `nvidia-smi` should show 70-90%
- CPU bottleneck: Check if MediaPipe hands is slowing down
**Solution**:
```bash
# Disable hands if bottleneck
python src/extract_features_yolov8_gpu.py --model yolov8m-pose.pt --no-hands --split train
```

---

## 🔄 Resume Capability

If extraction is interrupted:
```bash
# Just rerun the same command
python src/extract_features_yolov8_gpu.py --model yolov8m-pose.pt --split train

# It will automatically skip already-extracted .npy files
```

---

## ✅ Validation After Extraction

### 1. Check File Count
```bash
# Should be 4,376 for train
(Get-ChildItem data\processed\train\*.npy).Count
```

### 2. Check Feature Dimensions
```python
import numpy as np
sample = np.load('data/processed/train/01April_2010_Thursday_heute_default-0.npy')
print(f'Shape: {sample.shape}')  # Should be (num_frames, 177) or (num_frames, 51)
```

### 3. Review Summary
```bash
Get-Content data\processed\train\extraction_summary_yolov8.json
```

**Expected**:
```json
{
  "split": "train",
  "total_samples": 4376,
  "newly_extracted": 4376,
  "failed": 0,
  "average_fps": 30-60,
  "device": "cuda"
}
```

---

## 🚀 Recommended Execution

**For tonight (overnight run)**:
```bash
# Start extraction with medium model + hands
python src/extract_features_yolov8_gpu.py --model yolov8m-pose.pt --split train

# Expected completion: 3-4 AM
# Wake up to features ready for training!
```

**Monitor in separate terminal**:
```bash
nvidia-smi -l 1
```

---

## 📅 Timeline

| Time | Event |
|------|-------|
| Tonight 23:00 | Start extraction |
| Tonight 23:05 | Verify GPU at 70-90% |
| Tomorrow 02:00 | ~90% complete |
| Tomorrow 03:00 | **Train extraction COMPLETE** |
| Tomorrow 03:15 | Extract dev split (111 samples, ~5 min) |
| Tomorrow 03:20 | Extract test split (180 samples, ~8 min) |
| Tomorrow 03:30 | **ALL EXTRACTIONS COMPLETE** |
| Tomorrow 09:00 | Begin baseline model training |

---

## 🎯 Command to Execute NOW

**RECOMMENDED** (complete features, overnight):
```bash
python src/extract_features_yolov8_gpu.py --model yolov8m-pose.pt --split train
```

**FASTEST** (body only, if in hurry):
```bash
python src/extract_features_yolov8_gpu.py --model yolov8n-pose.pt --no-hands --split train
```

---

*Created: October 14, 2025*
*GPU: RTX 4070 with CUDA 12.6*
*Expected: 30-60 FPS with GPU acceleration*
*Time: 3-4 hours for complete extraction*
