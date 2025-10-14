# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a Master's thesis project for developing a **lightweight real-time sign language recognition system** optimized for edge devices in educational settings. The system aims to achieve >85% accuracy while operating on consumer-grade hardware using spatial-temporal feature fusion.

**Key Research Goals:**
- Deploy on devices with 8GB VRAM constraints
- Achieve <25% Word Error Rate on RWTH-PHOENIX-Weather 2014 dataset
- Maintain 30+ FPS real-time inference
- Keep model size under 100MB

## Dataset

The project uses the **RWTH-PHOENIX-Weather 2014** dataset located in `data/raw_data/`:
- Contains ~8,257 annotated sequences of German Sign Language
- Over 1,000-sign vocabulary
- Dataset size: 53GB, resolution: 210x260px
- Two variants available:
  - `phoenix-2014-multisigner/`: Multiple signer setup
  - `phoenix-2014-signerindependent-SI5/`: Signer-independent subset (SI5)

**Citation requirements**: When using this data, cite both:
1. Koller, Forster, & Ney (2015) - Continuous sign language recognition paper
2. Koller, Zargaran, & Ney (2017) - Re-Sign CVPR paper

## Architecture Design (Planned)

The system follows a three-phase development approach:

### Phase I: Baseline Development
- CNN-HMM baseline following Koller et al. (2015)
- MediaPipe Holistic feature extraction (33 pose + 42 hand + 468 face keypoints = ~500 landmarks)
- Offline feature caching for training efficiency
- Target: ~40% WER baseline

### Phase II: Architecture Optimization
**Core Components:**
- **Visual backbone**: MobileNetV3 (lightweight CNN)
- **Temporal modeling**: Bidirectional LSTM with selective attention
- **Sequence learning**: Connectionist Temporal Classification (CTC) loss
- **Knowledge distillation**: Transfer from I3D/SlowFast teacher models
  - Loss function: `L = 0.7 * L_soft + 0.3 * L_hard`
  - Temperature: T=3.0
  - Target: 95% of teacher accuracy, 90% parameter reduction

**Memory optimization techniques:**
- Mixed-precision (FP16) training (50% memory reduction)
- Gradient checkpointing (70% activation memory savings)
- Dynamic sequence truncation
- Feature quantization

### Phase III: Real-time Deployment
- TensorRT compilation (2-3x speedup)
- Sliding window processing (32-frame buffer, 8-frame stride)
- TensorFlow.js for browser deployment

## Evaluation Metrics

**Recognition Performance:**
- Word Error Rate (WER) - primary metric
- Sign Error Rate (SER) - isolated signs
- BLEU scores - translation quality

**Computational Efficiency:**
- Real-time factor (RTF)
- End-to-end latency
- Memory footprint
- Frames-per-second (FPS)

**User Experience:**
- System Usability Scale (SUS) questionnaire
- Task completion rates
- Qualitative feedback from deaf/hard-of-hearing students (5-10 participants)

## Experimental Design

All experiments should follow this ablation study structure:
1. Baseline CNN-HMM vs. proposed architecture
2. With/without knowledge distillation
3. Temporal window sizes: 16, 32, 64 frames
4. Impact of attention mechanisms
5. Optimization technique comparisons

**Reproducibility requirements:**
- Fixed random seeds
- 5-fold cross-validation on training set
- Report results as mean ± standard deviation

## Python Environment

The project uses a virtual environment (`venv/`) for dependency management.

**Setup:**
```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

Note: `requirements.txt` is currently empty - dependencies need to be added as development progresses.

## Specialized Agents

This repository has 10 specialized agents configured in `.claude/agents/`:
- `senior-ai-research-advisor` - Research planning and methodology
- `neural-network-architecture-specialist` - Model architecture design
- `knowledge-distillation-quantization-specialist` - Model compression
- `sequence-learning-rnn-expert` - LSTM/RNN temporal modeling
- `visual-feature-extraction-specialist` - MediaPipe feature extraction
- `real-time-inference-optimizer` - Deployment optimization
- `experiment-validation-expert` - Metrics and evaluation
- `hci-accessibility-researcher` - User studies with deaf/hard-of-hearing community
- `scientific-visualization-specialist` - Results visualization
- `scientific-writing-consultant` - Academic writing and thesis structure

Use these agents for specialized tasks in their respective domains.

## Research Proposal

See `research-proposal.md` for the complete thesis proposal including:
- Detailed methodology for each phase
- Literature review references
- Thesis chapter structure
- Educational accessibility considerations
- Ethical considerations and bias analysis

## Development Workflow

When implementing features:
1. **Feature extraction**: Start with MediaPipe Holistic pipeline for landmark extraction
2. **Model development**: Build incrementally following Phase I → Phase II → Phase III
3. **Memory constraints**: Always profile memory usage - target is 8GB VRAM
4. **Performance testing**: Continuously measure FPS and latency
5. **User studies**: Consult `hci-accessibility-researcher` agent for ethical protocols

## Key Technical Constraints

- **Hardware**: Consumer-grade GPUs with 8GB VRAM
- **Model size**: < 100MB
- **Inference speed**: 30+ FPS
- **Accuracy threshold**: > 85% (< 25% WER)
- **Dataset**: RWTH-PHOENIX-Weather 2014 (German Sign Language)
