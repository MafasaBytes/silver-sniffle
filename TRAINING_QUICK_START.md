# Training Quick Start Guide

**Status:** Dataset validated and ready for BiLSTM training

## Validated Dataset Statistics

**Extraction Complete:** 4,667/4,667 sequences (100% success)

| Split | Sequences | Frames | Avg Length |
|-------|-----------|--------|------------|
| Train | 4,376 | 612,027 | 139.9 ± 43.5 |
| Dev | 111 | 16,460 | 148.3 ± 40.0 |
| Test | 180 | 26,891 | 149.4 ± 44.9 |

**Feature Dimensions:** (num_frames, 177)
- YOLOv8-Pose: 51 body keypoints
- MediaPipe Hands: 126 hand landmarks

**Vocabulary:** 1,120 tokens (1,117 signs + 3 special tokens)

## Recommended Training Configuration

Based on validation analysis, use these settings for initial training:

```python
# Training hyperparameters
MAX_SEQUENCE_LENGTH = 250  # Keeps 99.4% of data
BATCH_SIZE = 16            # Can likely increase to 24-32
HIDDEN_DIM = 256           # Start conservative, can try 512
NUM_LAYERS = 2             # BiLSTM layers
BIDIRECTIONAL = True
DROPOUT = 0.3

# Model architecture
INPUT_DIM = 177            # Feature dimension
VOCAB_SIZE = 1120          # Including special tokens
BLANK_TOKEN_ID = 1         # For CTC loss

# Optimization
LEARNING_RATE = 1e-3       # Start here, tune as needed
WEIGHT_DECAY = 1e-5
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Data loading
NUM_WORKERS = 2            # Validated in extraction
PIN_MEMORY = True
```

## Dataset Loader Example

```python
from src.phoenix_dataset import create_dataloaders

# Create dataloaders
train_loader, dev_loader, test_loader = create_dataloaders(
    data_root="data/raw_data/phoenix-2014-signerindependent-SI5",
    features_root="data/processed",
    batch_size=16,
    num_workers=2,
    max_sequence_length=250
)

# Batch structure
for batch in train_loader:
    features = batch['features']        # (batch, max_len, 177)
    targets = batch['targets']          # (batch, max_target_len)
    feature_lengths = batch['feature_lengths']  # (batch,)
    target_lengths = batch['target_lengths']    # (batch,)
    break
```

## GPU Memory Estimates

**Feature memory per batch (batch_size=16, max_len=250):**
- Features: 16 × 250 × 177 × 4 bytes = 2.83 MB
- Targets: ~0.5 MB (short sequences)
- **Total batch data:** ~3.5 MB

**Available for model (8GB VRAM):**
- Feature data: ~3.5 MB
- Model parameters: ~20-50 MB (BiLSTM)
- Optimizer states: ~40-100 MB (Adam)
- Gradients: ~20-50 MB
- Activations: ~100-500 MB (depends on batch size)
- **Total estimated:** ~200-700 MB
- **Headroom:** 7.3-7.8 GB free (excellent!)

You can likely increase batch size to 24-32 without issues.

## Key Validation Results

- Zero NaN values detected
- Zero failed extractions
- Zero data leakage between splits
- Dimensions correct: (frames, 177)
- Dataset loader tested and working

## Performance Expectations

**Baseline targets from research proposal:**
- WER < 25% (target from Phase I CNN-HMM baseline: ~40% WER)
- Real-time factor < 1.0
- Inference FPS > 30

**Training time estimates:**
- ~1,094 batches per epoch (train)
- At ~0.5 sec/batch: ~9 min/epoch
- 100 epochs: ~15 hours training time

## Files and Locations

**Data:**
- Features: `data/processed/{train,dev,test}/*.npy`
- Vocabulary: `data/processed/train/vocabulary.txt`
- Annotations: `data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/*.SI5.corpus.csv`

**Code:**
- Dataset: `src/phoenix_dataset.py` (validated)
- Model: `src/train_bilstm.py` (to be created)

**Validation artifacts:**
- Full report: `VALIDATION_REPORT.md`
- JSON stats: `validation_report.json`
- Plots: `validation_plots.png`, `truncation_analysis.png`

## Next Steps

1. Implement BiLSTM model in `src/train_bilstm.py`
2. Set up TensorBoard logging
3. Implement WER/SER evaluation metrics
4. Run baseline training experiment
5. Tune hyperparameters based on initial results

**Dataset is ready. You can start training immediately!**
