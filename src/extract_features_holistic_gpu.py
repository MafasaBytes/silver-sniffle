"""
GPU-Accelerated Feature Extraction with MediaPipe Holistic
Uses YOLOv8-Pose + MediaPipe Holistic (face + hands)

Features extracted:
- YOLOv8-Pose: 17 keypoints × 3 (x,y,conf) = 51 features
- MediaPipe Face: 52 key points × 3 (x,y,z) = 156 features
- MediaPipe Hands: 21 × 2 hands × 3 (x,y,z) = 126 features
- Total: 333 features (was 177)
"""

import os
import torch
import cv2
import numpy as np
import pandas as pd
import json
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import time
import argparse
from datetime import datetime
import warnings
import logging
import sys
from multiprocessing import Process, Queue, Event
import threading
from queue import Empty
import signal

warnings.filterwarnings('ignore')
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['GLOG_minloglevel'] = '2'

try:
    import colorama
    colorama.init()
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False


# Key face landmark indices for efficient representation
FACE_KEY_INDICES = [
    # Face contour (17 points)
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
    # Left eye (6 points)
    33, 160, 158, 133, 153, 144,
    # Right eye (6 points)
    362, 385, 387, 263, 373, 380,
    # Left eyebrow (5 points)
    70, 63, 105, 66, 107,
    # Right eyebrow (5 points)
    336, 296, 334, 293, 300,
    # Nose (6 points)
    168, 6, 197, 195, 5, 4,
    # Mouth outer (7 points)
    61, 146, 91, 181, 84, 17, 314
]  # Total: 52 points × 3 = 156 features


def setup_logging(output_dir):
    """Setup logging configuration."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"holistic_extraction_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    return logging.getLogger(__name__)


class HolisticExtractor:
    """Optimized GPU extractor with MediaPipe Holistic."""

    def __init__(self, yolo_model='yolov8m-pose.pt', device='cuda',
                 batch_size=4, logger=None):
        """
        Initialize extractor with MediaPipe Holistic.

        Args:
            yolo_model: YOLOv8 model size
            device: 'cuda' or 'cpu'
            batch_size: Number of frames to process at once
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info("="*80)
        self.logger.info("MEDIAPIPE HOLISTIC FEATURE EXTRACTOR (333 features)")
        self.logger.info("="*80)

        # GPU setup
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size

        if self.device == 'cuda':
            gpu_props = torch.cuda.get_device_properties(0)
            self.logger.info(f"GPU: {gpu_props.name} ({gpu_props.total_memory/1e9:.1f}GB)")
            self.logger.info(f"Batch size: {batch_size}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            self.logger.warning("CUDA not available, using CPU")

        # Load YOLOv8-Pose
        self.logger.info(f"Loading {yolo_model}...")
        self.pose_model = YOLO(yolo_model)
        self.pose_model.overrides['conf'] = 0.25
        self.pose_model.overrides['iou'] = 0.45
        self.pose_model.overrides['max_det'] = 1

        if self.device == 'cuda':
            self.pose_model.to('cuda')
        self.logger.info(f"YOLOv8 loaded on {self.device}")

        # MediaPipe Holistic (CPU)
        self.logger.info("Initializing MediaPipe Holistic...")
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            refine_face_landmarks=False
        )
        self.logger.info("MediaPipe Holistic ready (face + hands)")
        self.logger.info("Feature dimensions: 51 (body) + 156 (face) + 126 (hands) = 333")
        self.logger.info("="*80)

        self.stats = {
            'total_sequences': 0,
            'total_frames': 0,
            'total_time': 0,
            'yolo_time': 0,
            'holistic_time': 0,
            'failed_sequences': []
        }

    def extract_batch(self, frames):
        """
        Extract features from a batch of frames using MediaPipe Holistic.

        Args:
            frames: List of frames (BGR format)

        Returns:
            Array of features (batch_size, 333)
        """
        batch_size = len(frames)
        features = np.zeros((batch_size, 333), dtype=np.float32)

        # YOLOv8 batch inference (GPU) - Body pose
        yolo_start = time.time()
        with torch.no_grad():
            results = self.pose_model(frames, stream=False, verbose=False)

        if self.device == 'cuda':
            torch.cuda.synchronize()

        self.stats['yolo_time'] += time.time() - yolo_start

        # Extract YOLOv8 keypoints
        for i, result in enumerate(results):
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                kpts = result.keypoints.data[0].cpu().numpy()  # (17, 3)
                features[i, :51] = kpts.flatten()

        # MediaPipe Holistic (CPU) - Face + Hands
        holistic_start = time.time()
        for i, frame in enumerate(frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(frame_rgb)

            # Extract face landmarks (52 key points = 156 features)
            if results.face_landmarks:
                face_features = np.zeros(156, dtype=np.float32)
                for idx, lm_idx in enumerate(FACE_KEY_INDICES):
                    if lm_idx < len(results.face_landmarks.landmark):
                        lm = results.face_landmarks.landmark[lm_idx]
                        face_features[idx*3:idx*3+3] = [lm.x, lm.y, lm.z]
                features[i, 51:207] = face_features

            # Extract left hand landmarks (63 features)
            if results.left_hand_landmarks:
                for lm_idx, lm in enumerate(results.left_hand_landmarks.landmark):
                    if lm_idx < 21:
                        base_idx = 207 + lm_idx * 3
                        features[i, base_idx:base_idx+3] = [lm.x, lm.y, lm.z]

            # Extract right hand landmarks (63 features)
            if results.right_hand_landmarks:
                for lm_idx, lm in enumerate(results.right_hand_landmarks.landmark):
                    if lm_idx < 21:
                        base_idx = 207 + 63 + lm_idx * 3
                        features[i, base_idx:base_idx+3] = [lm.x, lm.y, lm.z]

        self.stats['holistic_time'] += time.time() - holistic_start

        return features

    def process_video_sequence(self, frame_folder):
        """
        Process all frames in a video sequence with batching.

        Args:
            frame_folder: Path to folder containing frame PNG files

        Returns:
            Dictionary with features array and metadata
        """
        frame_files = sorted(frame_folder.glob("*.png"))
        if not frame_files:
            raise ValueError(f"No PNG files found in {frame_folder}")

        all_features = []
        start_time = time.time()

        num_frames = len(frame_files)
        num_batches = (num_frames + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_frames)
            batch_files = frame_files[start_idx:end_idx]

            # Load batch of frames
            batch_frames = []
            for frame_file in batch_files:
                frame = cv2.imread(str(frame_file))
                if frame is not None:
                    batch_frames.append(frame)

            if not batch_frames:
                continue

            # Extract features for batch
            batch_features = self.extract_batch(batch_frames)
            all_features.append(batch_features)

        if not all_features:
            raise ValueError("No features extracted")

        features_array = np.vstack(all_features)
        processing_time = time.time() - start_time

        return {
            'features': features_array,
            'num_frames': len(features_array),
            'processing_time': processing_time,
            'fps': len(features_array) / processing_time if processing_time > 0 else 0,
            'feature_shape': features_array.shape
        }

    def extract_dataset(self, split='train', output_dir=None, checkpoint_interval=100):
        """Extract features for entire dataset split."""
        data_root = PROJECT_ROOT / "data" / "raw_data" / "phoenix-2014-signerindependent-SI5"
        features_root = data_root / "features" / "fullFrame-210x260px"
        annotations_root = data_root / "annotations" / "manual"

        if output_dir is None:
            output_dir = PROJECT_ROOT / "data" / "processed_holistic" / split
        else:
            output_dir = Path(output_dir) / split

        output_dir.mkdir(exist_ok=True, parents=True)

        # Load corpus
        corpus_file = annotations_root / f"{split}.SI5.corpus.csv"
        print(f"Loading corpus: {corpus_file}")
        df = pd.read_csv(corpus_file, delimiter="|")

        # Check existing files
        existing_files = set([f.stem for f in output_dir.glob("*.npy")])
        samples_to_process = [
            (idx, row) for idx, row in df.iterrows()
            if row["id"] not in existing_files
        ]

        total_samples = len(df)
        already_extracted = len(existing_files)
        remaining = len(samples_to_process)

        print(f"\n{'='*80}")
        print(f"EXTRACTING: {split.upper()} SPLIT")
        print(f"{'='*80}")
        print(f"Total samples: {total_samples}")
        print(f"Already extracted: {already_extracted}")
        print(f"Remaining: {remaining}")
        print(f"Output features: 333 (body 51 + face 156 + hands 126)")
        print(f"Batch size: {self.batch_size}")
        print(f"{'='*80}\n")

        if remaining == 0:
            print("All samples already extracted!")
            return self._load_existing_metrics(output_dir)

        # Metrics tracking
        metrics = []
        total_frames = 0
        total_time = 0
        start_time = time.time()
        failed_samples = []

        pbar = tqdm(
            samples_to_process,
            desc=f"[{split.upper()}]",
            unit="seq",
            ncols=120,
            file=sys.stdout,
            dynamic_ncols=True,
            ascii=True if sys.platform == 'win32' else False
        )

        for sample_idx, (idx, row) in enumerate(pbar):
            sample_id = row["id"]
            frame_folder = features_root / split / row["folder"].replace("/*.png", "")

            if not frame_folder.exists():
                error_msg = f"Folder not found: {frame_folder}"
                failed_samples.append({'id': sample_id, 'error': error_msg})
                continue

            try:
                result = self.process_video_sequence(frame_folder)
                feature_file = output_dir / f"{sample_id}.npy"
                np.save(feature_file, result['features'])

                metrics.append({
                    'sample_id': sample_id,
                    'num_frames': result['num_frames'],
                    'processing_time': result['processing_time'],
                    'fps': result['fps'],
                    'feature_shape': str(result['feature_shape']),
                    'status': 'success',
                    'error_message': ''
                })

                total_frames += result['num_frames']
                total_time += result['processing_time']

                avg_fps = total_frames / total_time if total_time > 0 else 0

                if self.device == 'cuda':
                    gpu_memory = torch.cuda.memory_allocated() / 1e9
                    pbar.set_postfix({
                        'FPS': f'{avg_fps:.1f}',
                        'GPU': f'{gpu_memory:.1f}GB',
                        'Failed': len(failed_samples)
                    })
                else:
                    pbar.set_postfix({
                        'FPS': f'{avg_fps:.1f}',
                        'Failed': len(failed_samples)
                    })

                if self.device == 'cuda' and (sample_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                error_msg = str(e)
                failed_samples.append({'id': sample_id, 'error': error_msg})
                metrics.append({
                    'sample_id': sample_id,
                    'num_frames': 0,
                    'processing_time': 0,
                    'fps': 0,
                    'feature_shape': "(0, 0)",
                    'status': 'failed',
                    'error_message': error_msg
                })
                continue

            # Checkpoint
            if (sample_idx + 1) % checkpoint_interval == 0:
                elapsed = time.time() - start_time
                eta_seconds = (remaining - (sample_idx + 1)) * (elapsed / (sample_idx + 1))
                print(f"\n[CHECKPOINT] {sample_idx + 1}/{remaining} | "
                      f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_seconds/60:.1f}m | "
                      f"Avg FPS: {avg_fps:.1f}\n")

        pbar.close()

        if self.device == 'cuda':
            torch.cuda.empty_cache()

        # Calculate final metrics
        end_time = time.time()
        total_elapsed = end_time - start_time

        metrics_df = pd.DataFrame(metrics)
        success_df = metrics_df[metrics_df['status'] == 'success']

        summary = {
            'split': split,
            'total_samples': total_samples,
            'already_extracted': already_extracted,
            'newly_extracted': len(success_df),
            'failed': len(failed_samples),
            'total_frames': total_frames,
            'total_processing_time': total_time,
            'total_wall_time': total_elapsed,
            'average_fps': total_frames / total_time if total_time > 0 else 0,
            'batch_size': self.batch_size,
            'feature_dims': 333,
            'device': self.device,
            'timestamp': datetime.now().isoformat()
        }

        # Save metrics
        metrics_file = output_dir / "extraction_metrics_holistic.csv"
        metrics_df.to_csv(metrics_file, index=False)

        summary_file = output_dir / "extraction_summary_holistic.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        if failed_samples:
            failed_file = output_dir / "failed_samples_holistic.json"
            with open(failed_file, 'w') as f:
                json.dump(failed_samples, f, indent=2)

        self._print_summary(summary, failed_samples)

        return summary, metrics_df, failed_samples

    def _load_existing_metrics(self, output_dir):
        """Load metrics from already extracted split."""
        summary_file = output_dir / "extraction_summary_holistic.json"
        metrics_file = output_dir / "extraction_metrics_holistic.csv"

        if summary_file.exists() and metrics_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            metrics_df = pd.read_csv(metrics_file)
            return summary, metrics_df, []

        return None, None, []

    def _print_summary(self, summary, failed_samples):
        """Print extraction summary."""
        print(f"\n{'='*80}")
        print(f"EXTRACTION COMPLETE: {summary['split'].upper()}")
        print(f"{'='*80}")

        print("\nResults:")
        print(f"  Newly extracted: {summary['newly_extracted']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Total frames: {summary['total_frames']:,}")
        print(f"  Feature dims: {summary['feature_dims']}")

        print("\nPerformance:")
        print(f"  Average FPS: {summary['average_fps']:.1f}")
        print(f"  Total time: {summary['total_wall_time']/60:.1f} minutes")
        print(f"  Device: {summary['device']}")
        print(f"  Batch size: {summary['batch_size']}")

        if failed_samples:
            print(f"\nFailed Samples ({len(failed_samples)}):")
            for sample in failed_samples[:3]:
                print(f"  - {sample['id']}: {sample['error'][:50]}")
            if len(failed_samples) > 3:
                print(f"  ... and {len(failed_samples) - 3} more")

        print(f"{'='*80}\n")


def main():
    """Main extraction function."""
    parser = argparse.ArgumentParser(
        description="MediaPipe Holistic feature extraction (333 features)"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8m-pose.pt",
        help="YOLOv8 model size"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "dev", "test", "all"],
        help="Which split to extract"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/processed_holistic)"
    )

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "data" / "processed_holistic")

    logger = setup_logging(Path(args.output_dir))

    extractor = HolisticExtractor(
        yolo_model=args.model,
        device=device,
        batch_size=args.batch_size,
        logger=logger
    )

    if args.split == "all":
        for split in ["train", "dev", "test"]:
            print(f"\n{'#'*80}")
            print(f"# PROCESSING {split.upper()} SPLIT")
            print(f"{'#'*80}\n")

            extractor.extract_dataset(
                split=split,
                output_dir=args.output_dir
            )
    else:
        extractor.extract_dataset(
            split=args.split,
            output_dir=args.output_dir
        )

    print(f"\n{'='*80}")
    print("ALL EXTRACTION COMPLETE!")
    print(f"{'='*80}")
    print(f"Features saved to: {args.output_dir}")
    print(f"Feature dimensions: 333 (body 51 + face 156 + hands 126)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
