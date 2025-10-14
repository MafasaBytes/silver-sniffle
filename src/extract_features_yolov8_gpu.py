"""
GPU-Accelerated Feature Extraction for RWTH-PHOENIX Dataset
Uses YOLOv8-Pose with batch processing for maximum GPU utilization

Key:
- Multi-GPU parallel processing
- Batch processing (8-16 frames at once)
- Efficient GPU memory management
- Proper tensor handling
- Optimized inference settings
- Clean logging system
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
from multiprocessing import Process, Queue, Manager, cpu_count, Event
import threading
from queue import Empty
import signal
warnings.filterwarnings('ignore')

# Enable Windows console compatibility
os.environ['PYTHONUNBUFFERED'] = '1'

os.environ['GLOG_minloglevel'] = '2'  # Suppress MediaPipe C++ warnings

# Initialize colorama for Windows terminal support
try:
    import colorama
    colorama.init()
except ImportError:
    pass  # colorama not required, tqdm will fallback

# Get project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

# Try to import GPU monitoring
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False


def setup_logging(output_dir):
    """Setup logging configuration."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"yolov8_extraction_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress verbose logs
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def get_gpu_info():
    """Get GPU information and availability."""
    gpu_info = {
        "available": torch.cuda.is_available(),
        "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": []
    }
    
    if gpu_info["available"]:
        for i in range(gpu_info["count"]):
            gpu_info["devices"].append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory / 1e9,
                "memory_allocated": torch.cuda.memory_allocated(i) / 1e9
            })
    
    return gpu_info


def monitor_gpu(stop_event, stats_queue, interval=5):
    """Monitor GPU usage in background thread."""
    if not HAS_GPUTIL:
        return
    
    while not stop_event.is_set():
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                stats = {
                    "timestamp": time.time(),
                    "gpus": []
                }
                for gpu in gpus:
                    stats["gpus"].append({
                        "id": gpu.id,
                        "load": gpu.load * 100,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "temperature": gpu.temperature
                    })
                stats_queue.put(stats)
        except:
            pass
        
        time.sleep(interval)


class YOLOv8Extractor:
    """Optimized GPU extractor with batch processing."""

    def __init__(self, yolo_model='yolov8m-pose.pt', use_hands=True, 
                 device='cuda', batch_size=4, logger=None):
        """
        Initialize extractor.

        Args:
            yolo_model: YOLOv8 model size
            use_hands: Whether to extract hand landmarks
            device: 'cuda' or 'cpu'
            batch_size: Number of frames to process at once
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("="*80)
        self.logger.info("OPTIMIZED YOLOV8 GPU FEATURE EXTRACTOR")
        self.logger.info("="*80)

        # GPU check and optimization
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        
        if self.device == 'cuda':
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_name = gpu_props.name
            gpu_memory = gpu_props.total_memory / 1e9
            self.logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            self.logger.info(f"Batch size: {batch_size}")
            
            # Enable cuDNN benchmarking for optimal performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            self.logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'

        # Load YOLOv8-Pose model with optimized settings
        self.logger.info(f"Loading {yolo_model}...")
        self.pose_model = YOLO(yolo_model)
        
        # Configure for batch inference
        self.pose_model.overrides['conf'] = 0.25  # Confidence threshold
        self.pose_model.overrides['iou'] = 0.45   # NMS IoU threshold
        self.pose_model.overrides['max_det'] = 1   # Max detections per image (1 person)
        self.pose_model.overrides['agnostic_nms'] = False
        
        if self.device == 'cuda':
            self.pose_model.to('cuda')
        self.logger.info(f"Model loaded on {self.device}")

        # MediaPipe Hands (optional, CPU only)
        self.use_hands = use_hands
        if use_hands:
            self.logger.info("Initializing MediaPipe Hands (CPU)...")
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,  # False for video sequences
                max_num_hands=2,
                min_detection_confidence=0.3,  # Lower threshold for better detection
                min_tracking_confidence=0.3,
                model_complexity=1  # Balance between speed and accuracy
            )
            self.logger.info("MediaPipe Hands ready")

        self.logger.info("="*80)

        # Performance tracking
        self.stats = {
            'total_sequences': 0,
            'total_frames': 0,
            'total_time': 0,
            'yolo_time': 0,
            'mediapipe_time': 0,
            'failed_sequences': []
        }

    def extract_batch(self, frames):
        """
        Extract features from a batch of frames.
        
        Args:
            frames: List of frames (BGR format)
            
        Returns:
            Array of features for each frame
        """
        batch_size = len(frames)
        
        # Initialize feature arrays
        if self.use_hands:
            features = np.zeros((batch_size, 177), dtype=np.float32)  # 51 body + 126 hands
        else:
            features = np.zeros((batch_size, 51), dtype=np.float32)   # 51 body only
        
        # YOLOv8 batch inference (GPU)
        yolo_start = time.time()
        with torch.no_grad():
            # Run batch inference
            results = self.pose_model(frames, stream=False, verbose=False)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()  # Ensure GPU operations complete
        
        self.stats['yolo_time'] += time.time() - yolo_start
        
        # Extract keypoints from results
        for i, result in enumerate(results):
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                # Get first person's keypoints (most confident)
                kpts = result.keypoints.data[0].cpu().numpy()  # Shape: (17, 3)
                features[i, :51] = kpts.flatten()
        
        # MediaPipe Hands (CPU, optional)
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
                            if base_idx + 3 <= 126:  # Bounds check
                                hand_features[base_idx:base_idx+3] = [landmark.x, landmark.y, landmark.z]
                    features[i, 51:] = hand_features
            
            self.stats['mediapipe_time'] += time.time() - mp_start
        
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
        
        # Process frames in batches
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
        
        # Concatenate all batch features
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
        """
        Extract features for entire dataset split with optimizations.
        """
        # Paths
        data_root = PROJECT_ROOT / "data" / "raw_data" / "phoenix-2014-signerindependent-SI5"
        features_root = data_root / "features" / "fullFrame-210x260px"
        annotations_root = data_root / "annotations" / "manual"

        if output_dir is None:
            output_dir = PROJECT_ROOT / "data" / "processed" / split
        else:
            output_dir = Path(output_dir) / split

        output_dir.mkdir(exist_ok=True, parents=True)

        # Load corpus
        corpus_file = annotations_root / f"{split}.SI5.corpus.csv"
        print(f"Loading corpus: {corpus_file}")
        df = pd.read_csv(corpus_file, delimiter="|")

        # Check existing files (resume capability)
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

        # Progress bar with Windows compatibility
        use_tqdm = False
        try:
            pbar = tqdm(
                samples_to_process,
                desc=f"[{split.upper()}]",
                unit="seq",
                ncols=120,
                file=sys.stdout,  # Explicit stdout for Windows
                dynamic_ncols=True,  # Adapt to terminal width
                ascii=True if sys.platform == 'win32' else False  # ASCII for Windows cmd
            )
            use_tqdm = True
        except Exception as e:
            print(f"WARNING: Could not initialize progress bar: {e}")
            print("Continuing without progress bar...")
            print(f"Processing {remaining} samples...\n")
            pbar = samples_to_process

        for sample_idx, (idx, row) in enumerate(pbar):
            sample_id = row["id"]
            frame_folder = features_root / split / row["folder"].replace("/*.png", "")

            if not frame_folder.exists():
                error_msg = f"Folder not found: {frame_folder}"
                failed_samples.append({'id': sample_id, 'error': error_msg})
                continue

            try:
                # Extract features
                result = self.process_video_sequence(frame_folder)

                # Save features
                feature_file = output_dir / f"{sample_id}.npy"
                np.save(feature_file, result['features'])

                # Track metrics
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

                # Update progress bar
                avg_fps = total_frames / total_time if total_time > 0 else 0

                # GPU memory monitoring
                if use_tqdm:
                    try:
                        if self.device == 'cuda':
                            gpu_memory = torch.cuda.memory_allocated() / 1e9
                            gpu_memory_max = torch.cuda.max_memory_allocated() / 1e9
                            pbar.set_postfix({
                                'FPS': f'{avg_fps:.1f}',
                                'GPU': f'{gpu_memory:.1f}/{gpu_memory_max:.1f}GB',
                                'Failed': len(failed_samples)
                            })
                        else:
                            pbar.set_postfix({
                                'FPS': f'{avg_fps:.1f}',
                                'Failed': len(failed_samples)
                            })
                    except Exception:
                        pass  # Ignore progress bar update errors
                else:
                    # Print progress without tqdm
                    if (sample_idx + 1) % 10 == 0:
                        print(f"[{split.upper()}] {sample_idx + 1}/{remaining} | FPS: {avg_fps:.1f} | Failed: {len(failed_samples)}")

                # Clear GPU cache periodically
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
                self._save_checkpoint(output_dir, metrics, split, sample_idx + 1)
                elapsed = time.time() - start_time
                eta_seconds = (remaining - (sample_idx + 1)) * (elapsed / (sample_idx + 1))
                print(f"\n[CHECKPOINT] {sample_idx + 1}/{remaining} | "
                      f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_seconds/60:.1f}m | "
                      f"Avg FPS: {avg_fps:.1f}\n")

        if use_tqdm:
            try:
                pbar.close()
            except Exception:
                pass

        # Clear GPU memory
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
            'use_hands': self.use_hands,
            'device': self.device,
            'timestamp': datetime.now().isoformat()
        }

        # Save metrics
        metrics_file = output_dir / "extraction_metrics_yolov8.csv"
        metrics_df.to_csv(metrics_file, index=False)

        summary_file = output_dir / "extraction_summary_yolov8.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        if failed_samples:
            failed_file = output_dir / "failed_samples_yolov8.json"
            with open(failed_file, 'w') as f:
                json.dump(failed_samples, f, indent=2)

        # Print summary
        self._print_summary(summary, failed_samples)

        return summary, metrics_df, failed_samples

    def _save_checkpoint(self, output_dir, metrics, split, sample_count):
        """Save checkpoint of current progress."""
        checkpoint_file = output_dir / f"checkpoint_yolov8_{sample_count}.json"
        checkpoint_data = {
            'split': split,
            'samples_processed': sample_count,
            'timestamp': datetime.now().isoformat()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def _load_existing_metrics(self, output_dir):
        """Load metrics from already extracted split."""
        summary_file = output_dir / "extraction_summary_yolov8.json"
        metrics_file = output_dir / "extraction_metrics_yolov8.csv"

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


def worker_process(worker_id, task_queue, result_queue, data_root, features_root, 
                   output_dir, model_name='yolov8m-pose.pt', use_hands=True, 
                   gpu_id=0, batch_size=8):
    """
    Worker process for parallel GPU extraction.
    
    Args:
        worker_id: Unique ID for this worker
        task_queue: Queue of (idx, row) tuples to process
        result_queue: Queue to put results
        data_root: Root of dataset
        features_root: Root of video frames
        output_dir: Where to save features
        model_name: YOLOv8 model name
        use_hands: Whether to use MediaPipe hands
        gpu_id: GPU device ID to use
        batch_size: Batch size for inference
    """
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, use device 0
    
    # Create worker logger
    worker_logger = logging.getLogger(f"Worker-{worker_id}")
    
    # Initialize extractor
    extractor = YOLOv8Extractor(
        yolo_model=model_name,
        use_hands=use_hands,
        device='cuda',
        batch_size=batch_size,
        logger=worker_logger
    )
    
    worker_logger.info(f"Started on GPU:{gpu_id}")
    
    processed_count = 0
    error_count = 0
    
    while True:
        try:
            # Non-blocking get with timeout
            task = task_queue.get(timeout=0.5)
            
            if task is None:  # Poison pill
                worker_logger.info(f"Received stop signal after {processed_count} samples")
                break
            
            idx, row = task
            sample_id = row["id"]
            split = row.get("split", "train")
            frame_folder = features_root / split / row["folder"].replace("/*.png", "")
            
            if not frame_folder.exists():
                result_queue.put({
                    "sample_id": sample_id,
                    "status": "failed",
                    "error": f"Folder not found: {frame_folder}",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id
                })
                error_count += 1
                continue
            
            try:
                # Extract features
                result = extractor.process_video_sequence(frame_folder)
                
                # Save features
                feature_file = output_dir / f"{sample_id}.npy"
                np.save(feature_file, result['features'])
                
                # Send result back
                result_queue.put({
                    "sample_id": sample_id,
                    "status": "success",
                    "num_frames": result["num_frames"],
                    "processing_time": result["processing_time"],
                    "fps": result["fps"],
                    "feature_shape": str(result["feature_shape"]),
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "error": ""
                })
                
                processed_count += 1
                
                # Clear GPU cache periodically
                if processed_count % 10 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                worker_logger.error(f"Failed to process {sample_id}: {str(e)}")
                result_queue.put({
                    "sample_id": sample_id,
                    "status": "failed",
                    "error": str(e),
                    "worker_id": worker_id,
                    "gpu_id": gpu_id
                })
                error_count += 1
                
        except Empty:
            continue
        except Exception as e:
            worker_logger.error(f"Worker error: {e}")
            continue
    
    worker_logger.info(f"Finished (Processed: {processed_count}, Errors: {error_count})")
    del extractor  # Cleanup GPU resources
    torch.cuda.empty_cache()


class ParallelYOLOv8Extractor:
    """Parallel YOLOv8 extractor with multi-GPU support."""
    
    def __init__(self, yolo_model='yolov8m-pose.pt', use_hands=True, 
                 num_workers=None, batch_size=8):
        """
        Initialize parallel extractor.
        
        Args:
            yolo_model: YOLOv8 model name
            use_hands: Whether to use MediaPipe hands
            num_workers: Number of worker processes (None = auto)
            batch_size: Batch size per worker
        """
        self.yolo_model = yolo_model
        self.use_hands = use_hands
        self.batch_size = batch_size
        
        # GPU configuration
        self.gpu_info = get_gpu_info()
        
        # Worker configuration
        if num_workers is None:
            if self.gpu_info["available"]:
                # Use 1-2 workers per GPU for optimal utilization
                self.num_workers = min(self.gpu_info["count"] * 2, 4)
            else:
                self.num_workers = 1
        else:
            self.num_workers = num_workers
        
        # Paths
        self.data_root = PROJECT_ROOT / "data" / "raw_data" / "phoenix-2014-signerindependent-SI5"
        self.features_root = self.data_root / "features" / "fullFrame-210x260px"
        self.annotations_root = self.data_root / "annotations" / "manual"
        
        # Process management
        self.workers = []
        self.should_stop = False
        self.stop_event = Event()
        
        # Setup signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Setup logging
        self.logger = setup_logging(PROJECT_ROOT / "data" / "processed")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info("Received shutdown signal. Cleaning up...")
        self.should_stop = True
        self.stop_event.set()
    
    def extract_split(self, split='train', output_dir=None, checkpoint_interval=50):
        """Extract features for a dataset split using parallel processing."""
        if output_dir is None:
            output_dir = PROJECT_ROOT / "data" / "processed"
        
        output_dir = Path(output_dir) / split
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load corpus
        corpus_file = self.annotations_root / f"{split}.SI5.corpus.csv"
        self.logger.info(f"Loading corpus: {corpus_file}")
        df = pd.read_csv(corpus_file, delimiter="|")
        df["split"] = split
        
        # Check existing files
        existing_files = set([f.stem for f in output_dir.glob("*.npy")])
        samples_to_process = [
            (idx, row) for idx, row in df.iterrows()
            if row["id"] not in existing_files
        ]
        
        total_samples = len(df)
        already_extracted = len(existing_files)
        remaining = len(samples_to_process)
        
        # Log extraction info
        self.logger.info("="*80)
        self.logger.info(f"EXTRACTION: {split.upper()} SPLIT")
        self.logger.info("="*80)
        self.logger.info(f"Total samples: {total_samples}")
        self.logger.info(f"Already extracted: {already_extracted}")
        self.logger.info(f"Remaining: {remaining}")
        self.logger.info(f"Workers: {self.num_workers}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Model: {self.yolo_model}")
        self.logger.info(f"Use hands: {self.use_hands}")
        
        if self.gpu_info["available"]:
            for gpu in self.gpu_info["devices"]:
                self.logger.info(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_total']:.1f}GB)")
        self.logger.info("="*80)
        
        if remaining == 0:
            self.logger.info("All samples already extracted!")
            return self._load_existing_metrics(output_dir)
        
        # Create queues
        task_queue = Queue(maxsize=self.num_workers * 10)
        result_queue = Queue()
        
        # Start GPU monitoring
        stats_queue = Queue()
        gpu_monitor_thread = None
        if self.gpu_info["available"] and HAS_GPUTIL:
            gpu_monitor_thread = threading.Thread(
                target=monitor_gpu,
                args=(self.stop_event, stats_queue),
                daemon=True
            )
            gpu_monitor_thread.start()
        
        # Start worker processes
        self.logger.info(f"Starting {self.num_workers} workers...")
        for i in range(self.num_workers):
            # Distribute workers across GPUs
            gpu_id = i % self.gpu_info["count"] if self.gpu_info["available"] else 0
            
            worker = Process(
                target=worker_process,
                args=(i, task_queue, result_queue, self.data_root, 
                      self.features_root, output_dir, self.yolo_model,
                      self.use_hands, gpu_id, self.batch_size)
            )
            worker.start()
            self.workers.append(worker)
        
        # Metrics tracking
        metrics = []
        total_frames = 0
        total_time = 0
        start_time = time.time()
        failed_samples = []
        processed_count = 0
        
        # GPU stats tracking
        gpu_stats = []
        
        # Feed tasks in background thread
        def feed_tasks():
            for task in samples_to_process:
                if self.should_stop:
                    break
                task_queue.put(task)
            # Send poison pills
            for _ in range(self.num_workers):
                task_queue.put(None)
        
        feeder_thread = threading.Thread(target=feed_tasks, daemon=True)
        feeder_thread.start()
        
        # Progress bar with Windows compatibility
        pbar = tqdm(
            total=remaining,
            desc=f"Extracting {split}",
            unit="video",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            file=sys.stdout,  # Explicit stdout for Windows
            ascii=True if sys.platform == 'win32' else False  # ASCII for Windows cmd
        )
        
        # Main collection loop
        last_gpu_check = time.time()
        while processed_count < remaining and not self.should_stop:
            try:
                # Collect GPU stats periodically
                if time.time() - last_gpu_check > 5:
                    while not stats_queue.empty():
                        try:
                            gpu_stat = stats_queue.get_nowait()
                            gpu_stats.append(gpu_stat)
                        except Empty:
                            break
                    last_gpu_check = time.time()
                
                # Get result with timeout
                result = result_queue.get(timeout=1.0)
                
                # Record metrics
                metrics.append({
                    'sample_id': result['sample_id'],
                    'status': result['status'],
                    'worker_id': result['worker_id'],
                    'gpu_id': result.get('gpu_id', -1),
                    'num_frames': result.get('num_frames', 0),
                    'processing_time': result.get('processing_time', 0),
                    'fps': result.get('fps', 0),
                    'feature_shape': result.get('feature_shape', ''),
                    'error_message': result.get('error', '')
                })
                
                if result['status'] == 'success':
                    total_frames += result['num_frames']
                    total_time += result['processing_time']
                else:
                    failed_samples.append({
                        'id': result['sample_id'],
                        'error': result['error']
                    })
                
                processed_count += 1
                
                # Update progress bar
                avg_fps = total_frames / total_time if total_time > 0 else 0
                gpu_load = 0
                if gpu_stats and gpu_stats[-1]['gpus']:
                    gpu_load = np.mean([g['load'] for g in gpu_stats[-1]['gpus']])
                
                pbar.set_postfix({
                    'FPS': f'{avg_fps:.1f}',
                    'GPU': f'{gpu_load:.0f}%' if self.gpu_info["available"] else 'CPU',
                    'Failed': len(failed_samples)
                })
                pbar.update(1)
                
                # Checkpoint
                if processed_count % checkpoint_interval == 0:
                    self._save_checkpoint(output_dir, metrics, split, processed_count)
                    elapsed = time.time() - start_time
                    eta_seconds = (remaining - processed_count) * (elapsed / processed_count)
                    
                    self.logger.info(
                        f"[CHECKPOINT] {processed_count}/{remaining} | "
                        f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_seconds/60:.1f}m | "
                        f"Avg FPS: {avg_fps:.1f}"
                    )
                    
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error collecting result: {e}")
                continue
        
        pbar.close()
        self.stop_event.set()
        
        # Wait for workers
        self.logger.info("Waiting for workers to finish...")
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                self.logger.warning(f"Terminating worker {worker.pid}")
                worker.terminate()
        
        # Calculate final metrics
        end_time = time.time()
        total_elapsed = end_time - start_time
        
        metrics_df = pd.DataFrame(metrics)
        success_df = metrics_df[metrics_df['status'] == 'success']
        
        # Calculate GPU statistics
        gpu_summary = {}
        if gpu_stats:
            all_gpu_loads = []
            all_gpu_memory = []
            for stat in gpu_stats:
                for gpu in stat['gpus']:
                    all_gpu_loads.append(gpu['load'])
                    all_gpu_memory.append(gpu['memory_used'] / gpu['memory_total'] * 100)
            
            gpu_summary = {
                'avg_gpu_load': np.mean(all_gpu_loads) if all_gpu_loads else 0,
                'max_gpu_load': np.max(all_gpu_loads) if all_gpu_loads else 0,
                'avg_memory_usage': np.mean(all_gpu_memory) if all_gpu_memory else 0,
                'max_memory_usage': np.max(all_gpu_memory) if all_gpu_memory else 0
            }
        
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
            'speedup_factor': total_time / total_elapsed if total_elapsed > 0 else 1,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'model': self.yolo_model,
            'use_hands': self.use_hands,
            'device': 'cuda' if self.gpu_info["available"] else 'cpu',
            **gpu_summary,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save metrics
        metrics_file = output_dir / "extraction_metrics_yolov8.csv"
        metrics_df.to_csv(metrics_file, index=False)
        
        summary_file = output_dir / "extraction_summary_yolov8.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if failed_samples:
            failed_file = output_dir / "failed_samples_yolov8.json"
            with open(failed_file, 'w') as f:
                json.dump(failed_samples, f, indent=2)
        
        # Save GPU stats
        if gpu_stats:
            gpu_stats_file = output_dir / "gpu_stats_yolov8.json"
            with open(gpu_stats_file, 'w') as f:
                json.dump(gpu_stats, f, indent=2)
        
        # Print summary
        self._print_summary(summary, failed_samples)
        
        return summary, metrics_df, failed_samples
    
    def _save_checkpoint(self, output_dir, metrics, split, sample_count):
        """Save checkpoint of current progress."""
        checkpoint_file = output_dir / f"checkpoint_yolov8_{sample_count}.json"
        checkpoint_data = {
            'split': split,
            'samples_processed': sample_count,
            'timestamp': datetime.now().isoformat()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def _load_existing_metrics(self, output_dir):
        """Load metrics from already extracted split."""
        summary_file = output_dir / "extraction_summary_yolov8.json"
        metrics_file = output_dir / "extraction_metrics_yolov8.csv"
        
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
        
        print("\nPerformance:")
        print(f"  Average FPS: {summary['average_fps']:.1f}")
        print(f"  Total time: {summary['total_wall_time']/60:.1f} minutes")
        print(f"  Speedup: {summary['speedup_factor']:.2f}x")
        print(f"  Workers: {summary['num_workers']}")
        print(f"  Device: {summary['device']}")
        print(f"  Batch size: {summary['batch_size']}")
        
        if 'avg_gpu_load' in summary:
            print("\nGPU Statistics:")
            print(f"  Average GPU load: {summary['avg_gpu_load']:.1f}%")
            print(f"  Peak GPU load: {summary['max_gpu_load']:.1f}%")
            print(f"  Average memory: {summary['avg_memory_usage']:.1f}%")
            print(f"  Peak memory: {summary['max_memory_usage']:.1f}%")
        
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
        description="GPU-accelerated feature extraction with YOLOv8-Pose"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8m-pose.pt",
        choices=["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt",
                 "yolov8l-pose.pt", "yolov8x-pose.pt"],
        help="YOLOv8 model size (default: yolov8m-pose.pt)"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "dev", "test", "all"],
        help="Which split to extract (default: train)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for GPU processing (default: 8)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of worker processes (default: auto based on GPU count)"
    )
    parser.add_argument(
        "--no-hands", action="store_true",
        help="Skip MediaPipe hand tracking (faster but less features)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/processed)"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=50,
        help="Save checkpoint every N samples (default: 50)"
    )
    parser.add_argument(
        "--parallel", action="store_true", default=True,
        help="Use parallel processing (default: True)"
    )
    parser.add_argument(
        "--no-parallel", dest="parallel", action="store_false",
        help="Disable parallel processing"
    )

    args = parser.parse_args()

    if args.parallel:
        # Use parallel extractor with multi-GPU support
        print("\n" + "="*80)
        print("PARALLEL YOLOV8 GPU FEATURE EXTRACTION")
        print("="*80)
        
        extractor = ParallelYOLOv8Extractor(
            yolo_model=args.model,
            use_hands=not args.no_hands,
            num_workers=args.workers,
            batch_size=args.batch_size
        )
        
        print(f"\nDataset: RWTH-PHOENIX-Weather 2014 SI5")
        print(f"Workers: {extractor.num_workers}")
        print(f"Batch size: {args.batch_size}")
        print(f"Model: {args.model}")
        print(f"Use hands: {not args.no_hands}")
        
        if extractor.gpu_info["available"]:
            print(f"\nGPU Information:")
            for gpu in extractor.gpu_info["devices"]:
                print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_total']:.1f}GB)")
        
        print(f"\nSamples to process:")
        print(f"  Train: 4,259 samples")
        print(f"  Dev: 106 samples")  
        print(f"  Test: 175 samples")
        print(f"  Total: 4,540 samples")
        print("="*80 + "\n")
        
        try:
            if args.split == "all":
                for split in ["train", "dev", "test"]:
                    print(f"\n{'#'*80}")
                    print(f"# PROCESSING {split.upper()} SPLIT")
                    print(f"{'#'*80}\n")
                    
                    extractor.extract_split(
                        split=split,
                        output_dir=args.output_dir,
                        checkpoint_interval=args.checkpoint_interval
                    )
            else:
                extractor.extract_split(
                    split=args.split,
                    output_dir=args.output_dir,
                    checkpoint_interval=args.checkpoint_interval
                )
            
            print("\n" + "="*80)
            print("EXTRACTION COMPLETE!")
            print("="*80)
            print(f"\nFeatures saved to: {PROJECT_ROOT / 'data' / 'processed'}")
            print(f"Logs saved to: {PROJECT_ROOT / 'data' / 'processed' / 'logs'}")
            print("="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Cleaning up...")
        except Exception as e:
            print(f"\n\nError: {e}")
            raise
    
    else:
        # Use single GPU extractor (original code)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger = setup_logging(PROJECT_ROOT / "data" / "processed")
        
        extractor = YOLOv8Extractor(
            yolo_model=args.model,
            use_hands=not args.no_hands,
            device=device,
            batch_size=args.batch_size,
            logger=logger
        )
        
        # Extract splits
        if args.split == "all":
            for split in ["train", "dev", "test"]:
                print(f"\n{'#'*80}")
                print(f"# PROCESSING {split.upper()} SPLIT")
                print(f"{'#'*80}\n")
                
                extractor.extract_dataset(
                    split=split,
                    output_dir=args.output_dir,
                    checkpoint_interval=args.checkpoint_interval
                )
        else:
            extractor.extract_dataset(
                split=args.split,
                output_dir=args.output_dir,
                checkpoint_interval=args.checkpoint_interval
            )
        
        print(f"\n{'='*80}")
        print("ALL EXTRACTION COMPLETE!")
        print(f"{'='*80}")
        print(f"Features saved to: {PROJECT_ROOT / 'data' / 'processed'}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()