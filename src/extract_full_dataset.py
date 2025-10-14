"""
Full Dataset Feature Extraction with Resume Capability
Extracts MediaPipe features for entire RWTH-PHOENIX-Weather 2014 SI5 dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import psutil
import os
import json
from datetime import datetime
from mediapipe_feature_extractor import MediaPipeFeatureExtractor

# Get project root directory (works from any directory)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR


class FullDatasetExtractor:
    """Extract features from entire dataset with checkpointing and resume."""

    def __init__(self, data_root=None):
        # Use PROJECT_ROOT for default paths
        if data_root is None:
            data_root = PROJECT_ROOT / "data" / "raw_data" / "phoenix-2014-signerindependent-SI5"
        self.data_root = Path(data_root)
        self.features_root = self.data_root / "features" / "fullFrame-210x260px"
        self.annotations_root = self.data_root / "annotations" / "manual"
        self.extractor = MediaPipeFeatureExtractor()

    def extract_split(self, split="train", output_dir=None, checkpoint_interval=100):
        """
        Extract features for an entire split with resume capability.

        Args:
            split: train, dev, or test
            output_dir: directory to save extracted features
            checkpoint_interval: save checkpoint every N samples

        Returns:
            performance_metrics: dict with timing and memory stats
        """
        # Use PROJECT_ROOT for default output
        if output_dir is None:
            output_dir = PROJECT_ROOT / "data" / "processed"

        # Load corpus
        corpus_file = self.annotations_root / f"{split}.SI5.corpus.csv"
        df = pd.read_csv(corpus_file, delimiter="|")

        output_dir = Path(output_dir) / split
        output_dir.mkdir(exist_ok=True, parents=True)

        # Check which samples already exist (for resume)
        existing_files = set([f.stem for f in output_dir.glob("*.npy")])
        samples_to_process = [
            (idx, row) for idx, row in df.iterrows()
            if row["id"] not in existing_files
        ]

        total_samples = len(df)
        already_extracted = len(existing_files)
        remaining = len(samples_to_process)

        print(f"\n{'='*80}")
        print(f"EXTRACTING FEATURES: {split.upper()} SPLIT")
        print(f"{'='*80}")
        print(f"Total samples in split: {total_samples}")
        print(f"Already extracted: {already_extracted}")
        print(f"Remaining to process: {remaining}")
        print(f"{'='*80}\n")

        if remaining == 0:
            print("✓ All samples already extracted!")
            return self._load_existing_metrics(output_dir)

        # Performance tracking
        metrics = {
            "sample_id": [],
            "num_frames": [],
            "processing_time": [],
            "fps": [],
            "memory_used_mb": [],
            "feature_shape": [],
            "status": [],  # 'success' or 'failed'
            "error_message": [],
        }

        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Track overall stats
        total_frames = 0
        total_time = 0
        start_time = time.time()
        failed_samples = []

        # Progress bar with explicit settings for background processes
        import sys
        pbar = tqdm(
            samples_to_process,
            desc=f"Extracting {split}",
            unit="video",
            file=sys.stderr,
            ncols=100,
            mininterval=1.0,
            disable=False
        )

        for sample_idx, (idx, row) in enumerate(pbar):
            sample_id = row["id"]
            frame_folder = self.features_root / split / row["folder"].replace("/*.png", "")

            if not frame_folder.exists():
                error_msg = f"Folder not found: {frame_folder}"
                failed_samples.append({"id": sample_id, "error": error_msg})
                metrics["sample_id"].append(sample_id)
                metrics["num_frames"].append(0)
                metrics["processing_time"].append(0)
                metrics["fps"].append(0)
                metrics["memory_used_mb"].append(0)
                metrics["feature_shape"].append("(0, 0)")
                metrics["status"].append("failed")
                metrics["error_message"].append(error_msg)
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
                metrics["status"].append("success")
                metrics["error_message"].append("")

                total_frames += result["num_frames"]
                total_time += result["processing_time"]

                # Update progress bar with stats
                avg_fps = total_frames / total_time if total_time > 0 else 0
                pbar.set_postfix({
                    'FPS': f'{avg_fps:.1f}',
                    'Failed': len(failed_samples),
                    'Progress': f'{sample_idx + 1}/{remaining}'
                })

            except Exception as e:
                error_msg = str(e)
                failed_samples.append({"id": sample_id, "error": error_msg})
                metrics["sample_id"].append(sample_id)
                metrics["num_frames"].append(0)
                metrics["processing_time"].append(0)
                metrics["fps"].append(0)
                metrics["memory_used_mb"].append(0)
                metrics["feature_shape"].append("(0, 0)")
                metrics["status"].append("failed")
                metrics["error_message"].append(error_msg)
                print(f"\nError processing {sample_id}: {error_msg}")
                continue

            # Checkpoint every N samples
            if (sample_idx + 1) % checkpoint_interval == 0:
                self._save_checkpoint(output_dir, metrics, split, sample_idx + 1)
                # Print heartbeat for background processes
                elapsed = time.time() - start_time
                avg_time_per_sample = elapsed / (sample_idx + 1)
                remaining_samples = remaining - (sample_idx + 1)
                eta_seconds = remaining_samples * avg_time_per_sample
                eta_hours = eta_seconds / 3600
                print(f"\n[CHECKPOINT] Processed {sample_idx + 1}/{remaining} | "
                      f"Elapsed: {elapsed/3600:.2f}h | ETA: {eta_hours:.2f}h | "
                      f"Avg FPS: {avg_fps:.1f}\n", flush=True)

        pbar.close()

        # Calculate final metrics
        end_time = time.time()
        total_elapsed = end_time - start_time

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory

        metrics_df = pd.DataFrame(metrics)
        success_df = metrics_df[metrics_df["status"] == "success"]

        summary = {
            "split": split,
            "total_samples": total_samples,
            "already_extracted": already_extracted,
            "newly_extracted": len(success_df),
            "failed": len(failed_samples),
            "total_frames": total_frames,
            "total_processing_time": total_time,
            "total_wall_time": total_elapsed,
            "average_fps": total_frames / total_time if total_time > 0 else 0,
            "mean_fps_per_video": success_df["fps"].mean() if len(success_df) > 0 else 0,
            "median_fps_per_video": success_df["fps"].median() if len(success_df) > 0 else 0,
            "min_fps": success_df["fps"].min() if len(success_df) > 0 else 0,
            "max_fps": success_df["fps"].max() if len(success_df) > 0 else 0,
            "mean_memory_per_video_mb": success_df["memory_used_mb"].mean() if len(success_df) > 0 else 0,
            "total_memory_increase_mb": total_memory_increase,
            "mean_frames_per_video": success_df["num_frames"].mean() if len(success_df) > 0 else 0,
            "median_frames_per_video": success_df["num_frames"].median() if len(success_df) > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        }

        # Save final metrics
        metrics_file = output_dir / "extraction_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)

        summary_file = output_dir / "extraction_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Save failed samples if any
        if failed_samples:
            failed_file = output_dir / "failed_samples.json"
            with open(failed_file, "w") as f:
                json.dump(failed_samples, f, indent=2)

        return summary, metrics_df, failed_samples

    def _save_checkpoint(self, output_dir, metrics, split, sample_count):
        """Save checkpoint of current progress."""
        checkpoint_file = output_dir / f"checkpoint_{sample_count}.json"
        checkpoint_data = {
            "split": split,
            "samples_processed": sample_count,
            "timestamp": datetime.now().isoformat(),
            "total_metrics": len(metrics["sample_id"])
        }
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

    def _load_existing_metrics(self, output_dir):
        """Load metrics from already extracted split."""
        summary_file = output_dir / "extraction_summary.json"
        metrics_file = output_dir / "extraction_metrics.csv"

        if summary_file.exists() and metrics_file.exists():
            with open(summary_file, "r") as f:
                summary = json.load(f)
            metrics_df = pd.read_csv(metrics_file)
            return summary, metrics_df, []
        else:
            return None, None, []

    def print_summary(self, summary, failed_samples):
        """Print extraction summary."""
        print(f"\n{'='*80}")
        print(f"EXTRACTION COMPLETE: {summary['split'].upper()} SPLIT")
        print(f"{'='*80}")
        print(f"\n--- Dataset Statistics ---")
        print(f"Total samples: {summary['total_samples']}")
        print(f"Already extracted: {summary['already_extracted']}")
        print(f"Newly extracted: {summary['newly_extracted']}")
        print(f"Failed: {summary['failed']}")

        print(f"\n--- Processing Performance ---")
        print(f"Total frames processed: {summary['total_frames']:,}")
        print(f"Processing time: {summary['total_processing_time'] / 3600:.2f} hours")
        print(f"Wall clock time: {summary['total_wall_time'] / 3600:.2f} hours")
        print(f"Average FPS: {summary['average_fps']:.2f}")
        print(f"Mean FPS per video: {summary['mean_fps_per_video']:.2f}")

        print(f"\n--- Video Statistics ---")
        print(f"Mean frames per video: {summary['mean_frames_per_video']:.1f}")
        print(f"Median frames per video: {summary['median_frames_per_video']:.1f}")

        print(f"\n--- Memory Usage ---")
        print(f"Mean memory per video: {summary['mean_memory_per_video_mb']:.2f} MB")
        print(f"Total memory increase: {summary['total_memory_increase_mb']:.2f} MB")

        if failed_samples:
            print(f"\n--- Failed Samples ({len(failed_samples)}) ---")
            for sample in failed_samples[:5]:  # Show first 5
                print(f"  - {sample['id']}: {sample['error']}")
            if len(failed_samples) > 5:
                print(f"  ... and {len(failed_samples) - 5} more")

        print(f"\n{'='*80}\n")

    def extract_all_splits(self, output_dir=None):
        """Extract features for all splits (train, dev, test)."""
        all_summaries = {}
        all_failed = {}

        for split in ["train", "dev", "test"]:
            print(f"\n\n{'#'*80}")
            print(f"# PROCESSING {split.upper()} SPLIT")
            print(f"{'#'*80}\n")

            summary, metrics_df, failed = self.extract_split(
                split=split,
                output_dir=output_dir,
                checkpoint_interval=100
            )

            if summary:
                self.print_summary(summary, failed)
                all_summaries[split] = summary
                all_failed[split] = failed

        # Generate overall report
        self._generate_final_report(all_summaries, all_failed, output_dir)

        return all_summaries, all_failed

    def _generate_final_report(self, all_summaries, all_failed, output_dir):
        """Generate final extraction report across all splits."""
        output_dir = Path(output_dir)
        report_file = output_dir / "FULL_DATASET_EXTRACTION_REPORT.txt"

        with open(report_file, "w") as f:
            f.write("="*80 + "\n")
            f.write("FULL DATASET FEATURE EXTRACTION REPORT\n")
            f.write("RWTH-PHOENIX-Weather 2014 SI5\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            total_samples = 0
            total_extracted = 0
            total_failed = 0
            total_frames = 0
            total_time = 0

            for split in ["train", "dev", "test"]:
                if split in all_summaries:
                    summary = all_summaries[split]
                    f.write(f"--- {split.upper()} SPLIT ---\n")
                    f.write(f"Total samples: {summary['total_samples']}\n")
                    f.write(f"Successfully extracted: {summary['already_extracted'] + summary['newly_extracted']}\n")
                    f.write(f"Failed: {summary['failed']}\n")
                    f.write(f"Processing time: {summary['total_processing_time'] / 3600:.2f} hours\n")
                    f.write(f"Average FPS: {summary['average_fps']:.2f}\n\n")

                    total_samples += summary['total_samples']
                    total_extracted += (summary['already_extracted'] + summary['newly_extracted'])
                    total_failed += summary['failed']
                    total_frames += summary['total_frames']
                    total_time += summary['total_processing_time']

            f.write("="*80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("="*80 + "\n")
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Successfully extracted: {total_extracted}\n")
            f.write(f"Failed: {total_failed}\n")
            f.write(f"Success rate: {total_extracted/total_samples*100:.2f}%\n")
            f.write(f"Total frames processed: {total_frames:,}\n")
            f.write(f"Total processing time: {total_time / 3600:.2f} hours\n")
            f.write(f"Overall average FPS: {total_frames / total_time if total_time > 0 else 0:.2f}\n")
            f.write("="*80 + "\n")

        print(f"\n✓ Final report saved to: {report_file}\n")


def main():
    """Main extraction function."""
    extractor = FullDatasetExtractor()

    print("\n" + "="*80)
    print("FULL DATASET FEATURE EXTRACTION")
    print("RWTH-PHOENIX-Weather 2014 SI5")
    print("="*80)
    print("\nThis will extract MediaPipe Holistic features for:")
    print("  - Train: 4,376 samples")
    print("  - Dev: 111 samples")
    print("  - Test: 180 samples")
    print("  - Total: 4,667 samples")
    print("\nEstimated time: ~10 hours")
    print("="*80 + "\n")

    # Extract all splits
    all_summaries, all_failed = extractor.extract_all_splits()

    print("\n" + "="*80)
    print("FULL DATASET EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\nAll features saved to: {PROJECT_ROOT / 'data' / 'processed'}/")
    print("Review FULL_DATASET_EXTRACTION_REPORT.txt for complete statistics.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
