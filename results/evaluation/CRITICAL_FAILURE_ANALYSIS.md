# CRITICAL FAILURE ANALYSIS
## Model Predicting 100% BLANK Tokens

**Date:** 2025-10-15
**Status:** 🔴 CRITICAL - Model completely broken
**Attempts:** 2 full training runs (unnormalized + normalized features)

---

## Executive Summary

The BiLSTM-CTC model has **completely failed to learn** despite two training attempts:

1. **First attempt (unnormalized features):** WER 92.97%, frame acc 3.9%
2. **Second attempt (normalized features):** WER 92.80%, frame acc 4.6%

**Root cause:** Model outputs **99.3% BLANK tokens** (CTC blank=1). It learned to predict blank for every single frame instead of actual sign tokens.

---

## Diagnostic Evidence

### Model Output Analysis

```
Sample 1 (173 frames):
  Reference: [871, 828, 133, 457, 750, ...]  (14 signs)
  Predictions: [1, 1, 1, 1, 1, 1, 1, ...]    (all BLANKs)
  BLANK percentage: 100.0%

Sample 2 (147 frames):
  Reference: [812, 57, 508, 683, 251, ...]   (11 signs)
  Predictions: [1, 1, 1, 1, 1, 1, 1, ...]    (all BLANKs)
  BLANK percentage: 100.0%

Dev Set (111 sequences, 16,460 frames):
  Token 1 (BLANK): 16,346 frames (99.3%)
  Token 1042:         63 frames ( 0.4%)
  Token 1043:         48 frames ( 0.3%)
  Other tokens:        3 frames ( 0.02%)
```

The model predicts only 5 unique tokens across the entire dev set, with BLANK dominating 99.3% of predictions.

---

## Why CTC Loss Converged Despite Failure

### CTC Loss Behavior

Training loss decreased from 16.1 → 4.1 (seemingly normal), but this is **deceptive**:

- **CTC loss can decrease even when model predicts all blanks**
- Loss measures alignment quality, not prediction accuracy
- Model found a "valid" but useless alignment: blank everywhere

### Frame Accuracy vs Loss

```
                  Unnormalized    Normalized
Train Loss:       4.06           4.39
Dev Loss:         4.81           4.39
Frame Accuracy:   3.9%           4.6%   ← Key indicator
```

Frame accuracy remained catastrophically low (<5%) in both attempts, indicating the model never learned meaningful patterns.

---

## Ruled Out Causes

✓ **Feature normalization:** Tried both unnormalized and normalized - no improvement
✓ **Target validity:** Validated targets are correct (no PAD/BLANK contamination)
✓ **Data loading:** Confirmed features and targets load correctly
✓ **Model architecture:** Standard BiLSTM-CTC (3M params) - sufficient capacity

---

## Probable Root Causes

### 1. **CTC Loss Configuration Issue** (HIGH PROBABILITY)

**Hypothesis:** CTC loss parameters may be misconfigured.

**Evidence:**
- Model immediately converges to "predict blank" solution
- This is a known CTC pathology when loss function has incorrect settings

**Suspects:**
```python
# From train_bilstm.py - need to verify:
criterion = nn.CTCLoss(blank=1, reduction='mean', zero_infinity=True)

# Potential issues:
# - Is blank=1 correct? (PAD=0, BLANK=1, UNK=2, signs=3-1119)
# - Should zero_infinity be True or False?
# - Is 'mean' reduction appropriate?
```

**Action:** Review CTC loss initialization against PyTorch docs and literature.

---

### 2. **Input/Output Length Mismatch** (MEDIUM PROBABILITY)

**Hypothesis:** CTC requires `output_length >= target_length`. If violated, model can't align properly.

**Evidence from debug output:**
```
Sample 1:
  Input frames: 173
  Output length: 173  (after BiLSTM)
  Target length: 14
  Ratio: 173/14 = 12.4x

Sample 2:
  Input frames: 147
  Output length: 147
  Target length: 11
  Ratio: 147/11 = 13.4x
```

The ratios look reasonable (10-15x), but need to verify:
- Are output lengths computed correctly from LSTM?
- Does pack/unpack preserve correct lengths?
- Are there edge cases where output < target?

**Action:** Add assertion to check `output_length >= target_length` for all samples.

---

### 3. **Learning Rate / Optimization Issue** (MEDIUM PROBABILITY)

**Hypothesis:** Model gets stuck in local minimum (all blanks) due to optimization issues.

**Evidence:**
- LR=0.001 may be too high for CTC loss
- Second attempt with LR=0.001 also failed
- LR scheduler reduced to 0.00005 but didn't help

**Observations:**
- Koller et al. (2015) baseline paper doesn't specify LR
- CTC loss is known to be sensitive to LR

**Action:** Try much lower initial LR (0.0001 or 0.00003) + longer warmup.

---

### 4. **Feature Quality Issue** (LOW PROBABILITY - but not ruled out)

**Hypothesis:** Extracted features may not contain temporal discriminative information.

**Evidence:**
- Features are spatial keypoint coordinates (YOLOv8 + MediaPipe)
- No temporal features (velocity, acceleration)
- Model may not be able to distinguish signs from static poses

**Counter-evidence:**
- Koller et al. (2015) used similar hand-crafted features successfully
- Features passed normalization checks

**Action:** Compute feature statistics per-sign class and check discriminative power.

---

### 5. **Vocabulary/CTC Blank Indexing** (LOW PROBABILITY)

**Hypothesis:** Mismatch between vocabulary indexing and CTC blank token.

**Current setup:**
```
PAD:   0  (padding token)
BLANK: 1  (CTC blank)
UNK:   2  (unknown token)
Signs: 3-1119  (actual vocabulary)
```

CTC loss configured with `blank=1`, which matches BLANK token. This looks correct.

**Action:** Double-check that no signs have token ID = 1 (would conflict with blank).

---

## Recommended Next Steps (Prioritized)

### URGENT: Verify CTC Loss Configuration

**Step 1:** Check PyTorch CTC loss docs and verify our configuration:
```python
# Current:
criterion = nn.CTCLoss(blank=1, reduction='mean', zero_infinity=True)

# Questions:
# 1. Is blank=1 the right index?
# 2. Should we use reduction='sum' instead of 'mean'?
# 3. What does zero_infinity do and is it needed?
# 4. Are log_probs in correct format (log softmax)?
```

**Step 2:** Add validation checks before CTC loss:
```python
# Verify shapes and ranges
assert (log_probs.shape[0] >= targets.shape[1]).all(), "Output too short!"
assert (output_lengths >= target_lengths).all(), "Output shorter than target!"
assert torch.isfinite(log_probs).all(), "Non-finite log probs!"
```

**Step 3:** Test CTC loss with synthetic data:
```python
# Create simple test case:
# - 3 classes + 1 blank
# - Simple repeating pattern
# - Verify loss decreases and predictions are correct
```

---

### Step 4: Try Lower Learning Rate with Warm-up

If CTC configuration is correct, try:

```python
# Option A: Much lower LR
learning_rate = 0.0001  (10x lower)

# Option B: LR warm-up
# Start at 1e-5, increase to 1e-3 over 5 epochs
# Then use ReduceLROnPlateau

# Option C: Different optimizer
# Try AdamW with different hyperparameters
```

---

### Step 5: Simplify Problem

Create a **minimal test case**:

1. **Subset data:** Use only 100 training samples
2. **Reduce vocabulary:** Top 50 most frequent signs only
3. **Shorter sequences:** Truncate to max 100 frames
4. **Train to overfit:** Should achieve near-0 train loss if model can learn at all

If model can't overfit this simple case, it's definitely a configuration bug, not a data/optimization issue.

---

##Comparison with Literature

### Koller et al. (2015) Baseline

Their CNN-HMM baseline achieved:
- **WER: ~40%** on PHOENIX 2014
- Used hand-crafted features (similar to ours)
- Different architecture (HMM not CTC)

### Expected BiLSTM-CTC Performance

Based on literature, BiLSTM-CTC on PHOENIX should achieve:
- **WER: 35-50%** (baseline)
- **Frame accuracy: 25-40%** (rough estimate)

Our results:
- **WER: 92.80%** (catastrophic)
- **Frame accuracy: 4.6%** (10x worse than expected)

This massive gap confirms we have a fundamental bug, not just suboptimal hyperparameters.

---

## Impact on Thesis Timeline

### Current Status
- Week 3: Stuck on Phase I baseline
- 2 failed training attempts
- Feature extraction working ✓
- Model architecture implemented ✓
- Training pipeline working ✓
- But model not learning ✗

### Revised Timeline

**Week 3 (remaining 3-4 days):**
- Day 1: Debug CTC loss configuration
- Day 2: Test with minimal dataset
- Day 3: Retrain with fixes
- Day 4: Validate baseline (35-45% WER target)

**Risk Level: HIGH**
- If CTC configuration issue → fixable in 1-2 days
- If fundamental feature/architecture issue → may need to pivot approach

**Contingency Plan:**
- If BiLSTM-CTC continues to fail, consider:
  1. Switching to simpler loss (e.g., framewise cross-entropy)
  2. Using pre-trained models/features
  3. Consulting with advisor for alternative baseline

---

## Key Takeaways

1. **CTC loss converging ≠ model learning** - Loss can decrease while model predicts useless outputs
2. **Frame accuracy is the key metric** - Should have monitored this more carefully from start
3. **Always validate model outputs** - We should have checked predictions much earlier
4. **Feature normalization was a red herring** - Real issue is model/loss configuration

---

## References

- Training logs: `models/bilstm_baseline/training_metrics.json`
- Model outputs analysis: `scripts/debug_model_outputs.py`
- CTC loss docs: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
- Koller et al. (2015): "Continuous Sign Language Recognition: Towards Large Vocabulary Statistical Recognition Systems"
