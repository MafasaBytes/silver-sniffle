"""
Batch Feature Extraction with Performance Profiling
Extracts MediaPipe features from multiple videos and profiles performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import psutil
import os
from mediapipe_feature_extractor import MediaPipeFeatureExtractor


class BatchFeatureExtractor:
    """Batch extract features with performance profiling."""

    def __init__(self, data_root="data/raw_data/phoenix-2014-signerindependent-SI5"):
        self.data_root = Path(data_root)
        self.features_root = self.data_root / "features" / "fullFrame-210x260px"
        self.annotations_root = self.data_root / "annotations" / "manual"
        self.extractor = MediaPipeFeatureExtractor()

    def extract_batch(self, split="train", num_samples=100, output_dir="data/processed"):
        """
        Extract features from a batch of samples.

        Args:
            split: train, dev, or test
            num_samples: number of samples to process
            output_dir: directory to save extracted features

        Returns:
            performance_metrics: dict with timing and memory stats
        """
        # Load corpus
        corpus_file = self.annotations_root / f"{split}.SI5.corpus.csv"
        df = pd.read_csv(corpus_file, delimiter="|")

        # Sample
        if num_samples < len(df):
            df = df.sample(num_samples, random_state=42)

        output_dir = Path(output_dir) / split
        output_dir.mkdir(exist_ok=True, parents=True)

        # Performance tracking
        metrics = {
            "sample_id": [],
            "num_frames": [],
            "processing_time": [],
            "fps": [],
            "memory_used_mb": [],
            "feature_shape": [],
        }

        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"\nExtracting features from {len(df)} samples ({split} split)...")
        print("=" * 70)

        total_frames = 0
        total_time = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
            sample_id = row["id"]
            frame_folder = self.features_root / split / row["folder"].replace("/*.png", "")

            if not frame_folder.exists():
                print(f"Warning: Folder not found: {frame_folder}")
                continue

            try:
                # Extract features
                result = self.extractor.process_video_frames(frame_folder)

                # Save features
                feature_file = output_dir / f"{sample_id}.npy"
                np.save(feature_file, result['features'])

                # Track metrics
                metrics["sample_id"].append(sample_id)
                metrics["num_frames"].append(result["num_frames"])
                metrics["processing_time"].append(result["processing_time"])
                metrics["fps"].append(result["fps"])
                metrics["memory_used_mb"].append(result["memory_used_mb"])
                metrics["feature_shape"].append(str(result["feature_shape"]))

                total_frames += result["num_frames"]
                total_time += result["processing_time"]

            except Exception as e:
                print(f"\nError processing {sample_id}: {str(e)}")
                continue

        # Calculate aggregate metrics
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory

        metrics_df = pd.DataFrame(metrics)

        summary = {
            "total_samples": len(metrics_df),
            "total_frames": total_frames,
            "total_processing_time": total_time,
            "average_fps": total_frames / total_time if total_time > 0 else 0,
            "mean_fps_per_video": metrics_df["fps"].mean(),
            "median_fps_per_video": metrics_df["fps"].median(),
            "min_fps": metrics_df["fps"].min(),
            "max_fps": metrics_df["fps"].max(),
            "mean_memory_per_video_mb": metrics_df["memory_used_mb"].mean(),
            "total_memory_increase_mb": total_memory_increase,
            "mean_frames_per_video": metrics_df["num_frames"].mean(),
            "median_frames_per_video": metrics_df["num_frames"].median(),
        }

        # Save metrics
        metrics_file = output_dir / "extraction_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)

        summary_file = output_dir / "extraction_summary.txt"
        with open(summary_file, "w") as f:
            f.write("Feature Extraction Performance Summary\n")
            f.write("=" * 70 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

        return summary, metrics_df

    def print_performance_report(self, summary, metrics_df):
        """Print detailed performance report."""
        print("\n" + "=" * 70)
        print("FEATURE EXTRACTION PERFORMANCE REPORT")
        print("=" * 70)

        print("\n--- Processing Statistics ---")
        print(f"Total samples processed: {summary['total_samples']}")
        print(f"Total frames processed: {summary['total_frames']}")
        print(f"Total processing time: {summary['total_processing_time']:.2f}s")

        print("\n--- Throughput Metrics ---")
        print(f"Average FPS (overall): {summary['average_fps']:.2f}")
        print(f"Mean FPS per video: {summary['mean_fps_per_video']:.2f}")
        print(f"Median FPS per video: {summary['median_fps_per_video']:.2f}")
        print(f"Min FPS: {summary['min_fps']:.2f}")
        print(f"Max FPS: {summary['max_fps']:.2f}")

        print("\n--- Memory Usage ---")
        print(f"Mean memory per video: {summary['mean_memory_per_video_mb']:.2f} MB")
        print(f"Total memory increase: {summary['total_memory_increase_mb']:.2f} MB")

        print("\n--- Video Characteristics ---")
        print(f"Mean frames per video: {summary['mean_frames_per_video']:.1f}")
        print(f"Median frames per video: {summary['median_frames_per_video']:.1f}")

        print("\n--- Performance Assessment ---")
        if summary['average_fps'] >= 30:
            print("✓ PASS: Achieving 30+ FPS target for real-time processing")
        else:
            fps_deficit = 30 - summary['average_fps']
            print(f"✗ FAIL: Below 30 FPS target by {fps_deficit:.2f} FPS")
            print("  Recommendations:")
            print("  - Consider reducing MediaPipe model_complexity (currently 1)")
            print("  - Use static_image_mode=False (already enabled)")
            print("  - Implement frame sampling/skipping strategy")
            print("  - Profile for bottlenecks in image loading vs processing")

        print("\n" + "=" * 70)


def main():
    """Main batch extraction function."""
    batch_extractor = BatchFeatureExtractor()

    # Extract features from 100 samples
    summary, metrics_df = batch_extractor.extract_batch(
        split="train",
        num_samples=100,
        output_dir="data/processed"
    )

    # Print report
    batch_extractor.print_performance_report(summary, metrics_df)

    print("\nFeatures and metrics saved to: data/processed/train/")
    print("Batch extraction complete!")


if __name__ == "__main__":
    main()
