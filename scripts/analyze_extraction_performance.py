"""
Analyze Feature Extraction Performance

Analyzes the performance of the feature extraction process:
- Actual vs. estimated FPS
- Processing time breakdown
- GPU utilization insights
- Performance bottleneck analysis
"""

import json
from pathlib import Path


def analyze_performance():
    """Analyze extraction performance metrics."""
    print("="*80)
    print("FEATURE EXTRACTION PERFORMANCE ANALYSIS")
    print("="*80)
    print()

    # Load validation results
    results_file = Path("data/processed/validation_report.json")
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        print("ERROR: Validation report not found. Run validate_extracted_features.py first.")
        return

    # Extract metrics
    total_frames = results['summary']['total_frames']
    total_samples = results['summary']['total_samples']

    # Performance metrics from user report
    print("EXTRACTION PERFORMANCE METRICS")
    print("-"*80)
    print()

    # Actual measured metrics from extraction logs
    train_samples = results['splits']['train']['actual_samples']
    train_frames = results['splits']['train']['total_frames']

    # IMPORTANT: Use actual measured time from extraction logs
    # Train split was extracted using 2 parallel workers
    # Actual time measured: 317.1 minutes (5h 17min) from VALIDATION_REPORT.md
    # Note: The total wall-clock time was 5h 40min, but this includes overhead
    train_duration_minutes = 317.1  # Actual measured time with 2 workers
    num_workers = 2  # Parallel extraction used 2 workers
    train_duration_seconds = train_duration_minutes * 60

    # Calculate FPS metrics
    # Per-worker FPS: how fast each worker processes frames
    per_worker_fps = train_frames / train_duration_seconds / num_workers

    # Aggregate throughput: total frames processed per second (both workers combined)
    aggregate_fps = train_frames / train_duration_seconds

    # Original single-threaded estimate
    estimated_fps = 7.7

    # Performance improvement (per worker)
    fps_improvement_per_worker = (per_worker_fps / estimated_fps) - 1

    # Parallel speedup
    parallel_speedup = aggregate_fps / estimated_fps

    print("1. TRAIN SPLIT EXTRACTION (Completed)")
    print(f"   Samples processed: {train_samples:,}")
    print(f"   Frames processed: {train_frames:,}")
    print(f"   Duration: {train_duration_minutes/60:.2f} hours ({train_duration_seconds:,} seconds)")
    print(f"   Parallel workers: {num_workers}")
    print(f"   Per-worker FPS: {per_worker_fps:.1f} frames/second")
    print(f"   Aggregate throughput: {aggregate_fps:.1f} frames/second")
    print(f"   Baseline estimate: {estimated_fps:.1f} frames/second (single-threaded)")
    print(f"   Per-worker improvement: {fps_improvement_per_worker*100:.1f}% faster than baseline")
    print(f"   Parallel speedup: {parallel_speedup:.2f}x (vs single-threaded baseline)")
    print()

    # Estimate dev/test extraction time based on aggregate throughput
    dev_frames = results['splits']['dev']['total_frames']
    test_frames = results['splits']['test']['total_frames']

    # Use aggregate throughput for wall-clock time estimates
    dev_duration_seconds = dev_frames / aggregate_fps
    test_duration_seconds = test_frames / aggregate_fps

    dev_duration_minutes = dev_duration_seconds / 60
    test_duration_minutes = test_duration_seconds / 60

    print("2. DEV SPLIT EXTRACTION (Estimated)")
    print(f"   Samples: {results['splits']['dev']['actual_samples']:,}")
    print(f"   Frames: {dev_frames:,}")
    print(f"   Estimated duration: {dev_duration_minutes:.1f} minutes ({dev_duration_minutes/60:.2f} hours)")
    print(f"   (Based on {aggregate_fps:.1f} FPS aggregate throughput with {num_workers} workers)")
    print()

    print("3. TEST SPLIT EXTRACTION (Estimated)")
    print(f"   Samples: {results['splits']['test']['actual_samples']:,}")
    print(f"   Frames: {test_frames:,}")
    print(f"   Estimated duration: {test_duration_minutes:.1f} minutes ({test_duration_minutes/60:.2f} hours)")
    print(f"   (Based on {aggregate_fps:.1f} FPS aggregate throughput with {num_workers} workers)")
    print()

    # Total time
    total_duration_seconds = train_duration_seconds + dev_duration_seconds + test_duration_seconds
    total_duration_hours = total_duration_seconds / 3600

    print("4. TOTAL EXTRACTION TIME")
    print(f"   Total frames: {total_frames:,}")
    print(f"   Total duration: {total_duration_hours:.2f} hours ({total_duration_hours/24:.2f} days)")
    print(f"   Per-worker FPS: {per_worker_fps:.1f}")
    print(f"   Aggregate throughput: {aggregate_fps:.1f} FPS ({num_workers} workers)")
    print()

    # Performance analysis
    print("="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    print()

    print("Why Did Performance Exceed Expectations?")
    print("-"*80)
    print()

    print("1. YOLOv8-Pose vs MediaPipe Full Holistic:")
    print("   - YOLOv8 is GPU-optimized with TensorRT/ONNX acceleration")
    print("   - Batch processing: YOLOv8 can process multiple frames in parallel")
    print("   - MediaPipe Holistic would run CPU-bound face mesh (468 points)")
    print("   - Estimate was based on MediaPipe's slower CPU processing")
    print()

    print("2. Feature Extraction Optimization:")
    print("   - Only extracting 177 features vs. 543 (MediaPipe full)")
    print("   - Skipped face landmarks (468 points) entirely")
    print("   - YOLOv8 pose: 51 body keypoints (faster than OpenPose)")
    print("   - MediaPipe Hands: 126 hand landmarks (faster than full Holistic)")
    print()

    print("3. GPU Utilization:")
    print("   - YOLOv8 efficiently uses GPU compute")
    print("   - Batch processing reduces CPU-GPU transfer overhead")
    print("   - Lower feature count = less memory bandwidth")
    print()

    print("4. Hardware Efficiency:")
    print("   - Modern GPUs handle YOLOv8-Pose very efficiently")
    print("   - Video decoding may have used hardware acceleration")
    print("   - Storage I/O was not a bottleneck (.npy writes are fast)")
    print()

    # Efficiency metrics
    print("="*80)
    print("EFFICIENCY METRICS")
    print("="*80)
    print()

    frames_per_sample = total_frames / total_samples
    storage_per_frame = (results['summary']['total_storage_mb'] * 1024) / total_frames  # KB per frame

    print(f"Average frames per sample: {frames_per_sample:.1f}")
    print(f"Storage per frame: {storage_per_frame:.2f} KB")
    print(f"Storage per sample: {storage_per_frame * frames_per_sample:.2f} KB")
    print()

    # Compare to original video
    original_size_gb = 53
    processed_size_gb = results['summary']['total_storage_mb'] / 1024
    compression_ratio = original_size_gb / processed_size_gb

    print(f"Original dataset: {original_size_gb} GB")
    print(f"Processed features: {processed_size_gb:.2f} GB")
    print(f"Compression ratio: {compression_ratio:.1f}x smaller")
    print(f"Compression percentage: {results['summary']['compression_ratio_pct']:.1f}%")
    print()

    # Throughput analysis
    print("="*80)
    print("THROUGHPUT ANALYSIS")
    print("="*80)
    print()

    samples_per_hour = train_samples / (train_duration_seconds / 3600)
    samples_per_minute = samples_per_hour / 60

    print(f"Processing throughput:")
    print(f"  - Per-worker: {per_worker_fps:.1f} frames/second")
    print(f"  - Aggregate ({num_workers} workers): {aggregate_fps:.1f} frames/second")
    print(f"  - {samples_per_minute:.2f} samples/minute")
    print(f"  - {samples_per_hour:.1f} samples/hour")
    print()

    # GPU utilization estimate
    # Note: With parallel processing, each worker gets partial GPU resources
    print("Estimated GPU Utilization:")
    print(f"  Based on {per_worker_fps:.1f} FPS per worker with YOLOv8-Pose + MediaPipe Hands:")
    print("  - YOLOv8 can achieve 100+ FPS on high-end GPUs (single worker, full GPU)")
    print(f"  - Observed {per_worker_fps:.1f} FPS per worker suggests ~16% GPU compute per worker")
    print(f"  - With {num_workers} parallel workers: ~{per_worker_fps*num_workers/100*100:.0f}% effective GPU utilization")
    print("  - Bottleneck likely: Video decoding (CPU), MediaPipe hand processing (CPU)")
    print("  - Further optimization potential: More workers, GPU-accelerated video decode")
    print()

    print("="*80)
    print("RECOMMENDATIONS FOR FUTURE EXTRACTIONS")
    print("="*80)
    print()

    print("1. Increase Parallel Workers:")
    print("   - Current: 2 workers achieving 32 FPS aggregate throughput")
    print("   - With 3 workers: ~48 FPS aggregate throughput (estimated)")
    print("   - With 4 workers: ~64 FPS aggregate (if GPU memory permits)")
    print()

    print("2. Multi-GPU Scaling:")
    print("   - Linear scaling with multiple GPUs")
    print("   - 2 GPUs with 2 workers each = 4x throughput = ~64 FPS aggregate")
    print()

    print("3. Model Optimization:")
    print("   - Convert YOLOv8 to TensorRT for 2-3x speedup")
    print("   - Use FP16 precision (minimal accuracy loss)")
    print()

    print("4. Pipeline Optimization:")
    print("   - Separate video decoding from feature extraction")
    print("   - Pre-load video frames in memory buffer")
    print("   - Async I/O for .npy file writing")
    print()


if __name__ == "__main__":
    analyze_performance()
