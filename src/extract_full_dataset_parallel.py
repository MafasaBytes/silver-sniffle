"""
Parallel Full Dataset Feature Extraction with GPU Support
Extracts MediaPipe features using multiple worker processes with GPU acceleration.

Features:
- GPU-accelerated MediaPipe processing
- Intelligent GPU/CPU work distribution
- Clean logging with separate log file
- Real-time GPU monitoring
- Optimized queue management
- Better error handling
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
from multiprocessing import Process, Queue, Manager, cpu_count, Event
import signal
import sys
import logging
from queue import Empty
import threading
import warnings
warnings.filterwarnings('ignore')

# Try to import GPU monitoring libraries
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("GPUtil not found. Install with: pip install gputil")

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except:
    HAS_NVML = False

# Get project root directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR


class GPUMediaPipeFeatureExtractor:
    """GPU-optimized MediaPipe feature extractor."""
    
    def __init__(self, device_id=0, use_gpu=True):
        """Initialize MediaPipe with GPU support."""
        import mediapipe as mp
        
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configure for GPU usage
        if use_gpu:
            # Set environment variables for GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
            
            # MediaPipe GPU configuration
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,  # Video mode for better temporal consistency
                model_complexity=2,  # Higher complexity for better accuracy
                smooth_landmarks=True,
                enable_segmentation=False,  # Disable for speed
                smooth_segmentation=False,
                refine_face_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            # CPU optimized configuration (per research advisor recommendation)
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=True,     # Better for CPU frame-by-frame
                model_complexity=1,         # Lower complexity for speed
                smooth_landmarks=False,     # Disable smoothing for speed
                enable_segmentation=False,  # Disable segmentation
                refine_face_landmarks=False,  # Disable refinement for speed
                min_detection_confidence=0.3,  # Lower threshold for speed
                min_tracking_confidence=0.3    # Lower threshold for speed
            )
        
        self.use_gpu = use_gpu
        self.device_id = device_id
    
    def process_video_frames(self, frame_folder, batch_size=8):
        """Process video frames with batching for GPU efficiency."""
        import cv2
        from PIL import Image
        
        frame_files = sorted(frame_folder.glob("*.png"))
        if not frame_files:
            raise ValueError(f"No frames found in {frame_folder}")
        
        features = []
        start_time = time.time()
        memory_start = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Process in batches for GPU efficiency
        for i in range(0, len(frame_files), batch_size):
            batch_frames = frame_files[i:i+batch_size]
            batch_features = []
            
            for frame_file in batch_frames:
                # Load and process frame
                image = cv2.imread(str(frame_file))
                if image is None:
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.holistic.process(image_rgb)
                
                # Extract features
                frame_features = self._extract_features(results)
                batch_features.append(frame_features)
            
            if batch_features:
                features.extend(batch_features)
        
        if not features:
            raise ValueError("No features extracted")
        
        features_array = np.array(features, dtype=np.float32)
        processing_time = time.time() - start_time
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024
        
        return {
            "features": features_array,
            "num_frames": len(features),
            "processing_time": processing_time,
            "fps": len(features) / processing_time if processing_time > 0 else 0,
            "memory_used_mb": memory_end - memory_start,
            "feature_shape": features_array.shape,
            "device": f"GPU:{self.device_id}" if self.use_gpu else "CPU"
        }
    
    def _extract_features(self, results):
        """Extract features from MediaPipe results."""
        features = []
        
        # Pose landmarks (33 points × 4 coordinates)
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            features.extend([0.0] * (33 * 4))
        
        # Left hand landmarks (21 points × 3 coordinates)
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            features.extend([0.0] * (21 * 3))
        
        # Right hand landmarks (21 points × 3 coordinates)
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            features.extend([0.0] * (21 * 3))
        
        # Face landmarks (468 points × 3 coordinates)
        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
        else:
            features.extend([0.0] * (468 * 3))
        
        return features
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'holistic'):
            self.holistic.close()


def setup_logging(output_dir):
    """Setup logging configuration."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extraction_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress verbose logs from libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def get_gpu_info():
    """Get GPU information and availability."""
    gpu_info = {
        "available": False,
        "count": 0,
        "devices": []
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["count"] = torch.cuda.device_count()
            for i in range(gpu_info["count"]):
                gpu_info["devices"].append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory / 1e9,
                    "memory_allocated": torch.cuda.memory_allocated(i) / 1e9
                })
    except ImportError:
        pass
    
    if HAS_GPUTIL:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info["available"] = True
                gpu_info["count"] = len(gpus)
                gpu_info["devices"] = []
                for gpu in gpus:
                    gpu_info["devices"].append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal / 1024,
                        "memory_used": gpu.memoryUsed / 1024,
                        "memory_free": gpu.memoryFree / 1024,
                        "gpu_load": gpu.load * 100,
                        "temperature": gpu.temperature
                    })
        except:
            pass
    
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


def worker_process(worker_id, task_queue, result_queue, data_root, features_root, 
                   output_dir, use_gpu=True, gpu_id=0):
    """
    Enhanced worker process with GPU support.
    
    Args:
        worker_id: Unique ID for this worker
        task_queue: Queue of (idx, row) tuples to process
        result_queue: Queue to put results
        data_root: Root of dataset
        features_root: Root of video frames
        output_dir: Where to save features
        use_gpu: Whether to use GPU
        gpu_id: GPU device ID to use
    """
    # Set process name for monitoring
    import setproctitle
    setproctitle.setproctitle(f"SLR_Worker_{worker_id}")
    
    # Initialize GPU-enabled MediaPipe
    extractor = GPUMediaPipeFeatureExtractor(
        device_id=gpu_id,
        use_gpu=use_gpu
    )
    
    worker_logger = logging.getLogger(f"Worker-{worker_id}")
    worker_logger.info(f"Started (Device: {'GPU:' + str(gpu_id) if use_gpu else 'CPU'})")
    
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
                    "error": "Folder not found",
                    "worker_id": worker_id
                })
                error_count += 1
                continue
            
            try:
                # Extract features with GPU acceleration
                result = extractor.process_video_frames(frame_folder, batch_size=16)
                
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
                    "memory_used_mb": result["memory_used_mb"],
                    "feature_shape": str(result["feature_shape"]),
                    "worker_id": worker_id,
                    "device": result["device"],
                    "error": ""
                })
                
                processed_count += 1
                
            except Exception as e:
                worker_logger.error(f"Failed to process {sample_id}: {str(e)}")
                result_queue.put({
                    "sample_id": sample_id,
                    "status": "failed",
                    "error": str(e),
                    "worker_id": worker_id
                })
                error_count += 1
                
        except Empty:
            continue
        except Exception as e:
            worker_logger.error(f"Worker error: {e}")
            continue
    
    worker_logger.info(f"Finished (Processed: {processed_count}, Errors: {error_count})")
    del extractor  # Cleanup GPU resources


class OptimizedParallelExtractor:
    """Optimized parallel dataset extractor with GPU support."""
    
    def __init__(self, data_root=None, num_workers=None, use_gpu=True):
        # Use PROJECT_ROOT for default paths
        if data_root is None:
            data_root = PROJECT_ROOT / "data" / "raw_data" / "phoenix-2014-signerindependent-SI5"
        
        self.data_root = Path(data_root)
        self.features_root = self.data_root / "features" / "fullFrame-210x260px"
        self.annotations_root = self.data_root / "annotations" / "manual"
        
        # GPU configuration
        self.gpu_info = get_gpu_info()
        self.use_gpu = use_gpu and self.gpu_info["available"]
        
        # Worker configuration
        if num_workers is None:
            if self.use_gpu:
                # For GPU: use fewer workers (1-2 per GPU)
                self.num_workers = min(self.gpu_info["count"] * 2, 4) if self.gpu_info["count"] > 0 else 2
            else:
                # For CPU: use more workers
                self.num_workers = min(cpu_count() - 1, 8)
        else:
            self.num_workers = num_workers
        
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
    
    def extract_split(self, split="train", output_dir=None, checkpoint_interval=50):
        """
        Extract features for a split with GPU acceleration.
        
        Args:
            split: train, dev, or test
            output_dir: directory to save extracted features
            checkpoint_interval: save checkpoint every N samples
            
        Returns:
            Performance metrics dictionary
        """
        if output_dir is None:
            output_dir = PROJECT_ROOT / "data" / "processed"
        
        # Load corpus
        corpus_file = self.annotations_root / f"{split}.SI5.corpus.csv"
        df = pd.read_csv(corpus_file, delimiter="|")
        df["split"] = split
        
        output_dir = Path(output_dir) / split
        output_dir.mkdir(exist_ok=True, parents=True)
        
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
        self.logger.info(f"GPU: {self.use_gpu}")
        if self.use_gpu:
            for gpu in self.gpu_info["devices"]:
                self.logger.info(f"  GPU {gpu['id']}: {gpu['name']}")
        self.logger.info("="*80)
        
        if remaining == 0:
            self.logger.info("All samples already extracted!")
            return self._load_existing_metrics(output_dir)
        
        # Create queues with larger buffer
        task_queue = Queue(maxsize=self.num_workers * 10)
        result_queue = Queue()
        
        # Start GPU monitoring
        stats_queue = Queue()
        gpu_monitor_thread = None
        if self.use_gpu and HAS_GPUTIL:
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
            gpu_id = i % self.gpu_info["count"] if self.use_gpu else 0
            
            worker = Process(
                target=worker_process,
                args=(i, task_queue, result_queue, self.data_root, 
                      self.features_root, output_dir, self.use_gpu, gpu_id)
            )
            worker.start()
            self.workers.append(worker)
        
        # Metrics tracking
        metrics = {
            "sample_id": [],
            "num_frames": [],
            "processing_time": [],
            "fps": [],
            "memory_used_mb": [],
            "feature_shape": [],
            "status": [],
            "error_message": [],
            "worker_id": [],
            "device": []
        }
        
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
        
        # Progress bar with better formatting
        pbar = tqdm(
            total=remaining,
            desc=f"[{split.upper()}]",
            unit="video",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
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
                
                # Get result with short timeout
                result = result_queue.get(timeout=0.1)
                
                # Record metrics
                metrics["sample_id"].append(result["sample_id"])
                metrics["worker_id"].append(result["worker_id"])
                metrics["status"].append(result["status"])
                metrics["device"].append(result.get("device", "unknown"))
                
                if result["status"] == "success":
                    metrics["num_frames"].append(result["num_frames"])
                    metrics["processing_time"].append(result["processing_time"])
                    metrics["fps"].append(result["fps"])
                    metrics["memory_used_mb"].append(result["memory_used_mb"])
                    metrics["feature_shape"].append(result["feature_shape"])
                    metrics["error_message"].append("")
                    
                    total_frames += result["num_frames"]
                    total_time += result["processing_time"]
                else:
                    failed_samples.append({
                        "id": result["sample_id"],
                        "error": result["error"]
                    })
                    # Add empty metrics for failed samples
                    for key in ["num_frames", "processing_time", "fps", "memory_used_mb"]:
                        metrics[key].append(0)
                    metrics["feature_shape"].append("(0, 0)")
                    metrics["error_message"].append(result["error"])
                
                processed_count += 1
                
                # Update progress bar
                avg_fps = total_frames / total_time if total_time > 0 else 0
                gpu_load = 0
                if gpu_stats and gpu_stats[-1]["gpus"]:
                    gpu_load = np.mean([g["load"] for g in gpu_stats[-1]["gpus"]])
                
                pbar.set_postfix({
                    'FPS': f'{avg_fps:.1f}',
                    'GPU': f'{gpu_load:.0f}%' if self.use_gpu else 'CPU',
                    'Fail': len(failed_samples)
                })
                pbar.update(1)
                
                # Checkpoint
                if processed_count % checkpoint_interval == 0:
                    self._save_checkpoint(output_dir, metrics, split, processed_count)
                    elapsed = time.time() - start_time
                    eta_seconds = (remaining - processed_count) * (elapsed / processed_count)
                    
                    self.logger.info(
                        f"[CHECKPOINT] {processed_count}/{remaining} | "
                        f"Elapsed: {elapsed/3600:.2f}h | ETA: {eta_seconds/3600:.2f}h | "
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
        success_df = metrics_df[metrics_df["status"] == "success"]
        
        # Calculate GPU statistics if available
        gpu_summary = {}
        if gpu_stats:
            all_gpu_loads = []
            all_gpu_memory = []
            for stat in gpu_stats:
                for gpu in stat["gpus"]:
                    all_gpu_loads.append(gpu["load"])
                    all_gpu_memory.append(gpu["memory_used"] / gpu["memory_total"] * 100)
            
            gpu_summary = {
                "avg_gpu_load": np.mean(all_gpu_loads),
                "max_gpu_load": np.max(all_gpu_loads),
                "avg_memory_usage": np.mean(all_gpu_memory),
                "max_memory_usage": np.max(all_gpu_memory)
            }
        
        summary = {
            "split": split,
            "total_samples": total_samples,
            "already_extracted": already_extracted,
            "newly_extracted": len(success_df),
            "failed": len(failed_samples),
            "total_frames": total_frames,
            "total_processing_time": total_time,
            "total_wall_time": total_elapsed,
            "num_workers": self.num_workers,
            "use_gpu": self.use_gpu,
            "speedup_factor": total_time / total_elapsed if total_elapsed > 0 else 1,
            "average_fps": total_frames / total_time if total_time > 0 else 0,
            "mean_fps_per_video": success_df["fps"].mean() if len(success_df) > 0 else 0,
            "median_fps_per_video": success_df["fps"].median() if len(success_df) > 0 else 0,
            "mean_memory_per_video_mb": success_df["memory_used_mb"].mean() if len(success_df) > 0 else 0,
            **gpu_summary,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save metrics
        metrics_file = output_dir / "extraction_metrics_gpu.csv"
        metrics_df.to_csv(metrics_file, index=False)
        
        summary_file = output_dir / "extraction_summary_gpu.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        if failed_samples:
            failed_file = output_dir / "failed_samples_gpu.json"
            with open(failed_file, "w") as f:
                json.dump(failed_samples, f, indent=2)
        
        # Save GPU stats if available
        if gpu_stats:
            gpu_stats_file = output_dir / "gpu_stats.json"
            with open(gpu_stats_file, "w") as f:
                json.dump(gpu_stats, f, indent=2)
        
        return summary, metrics_df, failed_samples
    
    def _save_checkpoint(self, output_dir, metrics, split, sample_count):
        """Save checkpoint of current progress."""
        checkpoint_file = output_dir / f"checkpoint_{sample_count}.json"
        checkpoint_data = {
            "split": split,
            "samples_processed": sample_count,
            "num_workers": self.num_workers,
            "use_gpu": self.use_gpu,
            "timestamp": datetime.now().isoformat()
        }
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def _load_existing_metrics(self, output_dir):
        """Load metrics from already extracted split."""
        summary_file = output_dir / "extraction_summary_gpu.json"
        metrics_file = output_dir / "extraction_metrics_gpu.csv"
        
        if not summary_file.exists():
            summary_file = output_dir / "extraction_summary_parallel.json"
            metrics_file = output_dir / "extraction_metrics_parallel.csv"
        
        if summary_file.exists() and metrics_file.exists():
            with open(summary_file, "r") as f:
                summary = json.load(f)
            metrics_df = pd.read_csv(metrics_file)
            return summary, metrics_df, []
        
        return None, None, []
    
    def print_summary(self, summary, failed_samples):
        """Print extraction summary with GPU info."""
        print("\n" + "="*80)
        print(f"EXTRACTION COMPLETE: {summary['split'].upper()}")
        print("="*80)
        
        print("\nDataset Statistics")
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  Already extracted: {summary['already_extracted']}")
        print(f"  Newly extracted: {summary['newly_extracted']}")
        print(f"  Failed: {summary['failed']}")
        
        print("\nPerformance")
        print(f"  Workers: {summary['num_workers']}")
        print(f"  Device: {'GPU' if summary.get('use_gpu') else 'CPU'}")
        print(f"  Total frames: {summary['total_frames']:,}")
        print(f"  Wall time: {summary['total_wall_time']/3600:.2f} hours")
        print(f"  Speedup: {summary['speedup_factor']:.2f}x")
        print(f"  Average FPS: {summary['average_fps']:.1f}")
        
        if 'avg_gpu_load' in summary:
            print("\nGPU Statistics")
            print(f"  Average GPU load: {summary['avg_gpu_load']:.1f}%")
            print(f"  Peak GPU load: {summary['max_gpu_load']:.1f}%")
            print(f"  Average memory: {summary['avg_memory_usage']:.1f}%")
            print(f"  Peak memory: {summary['max_memory_usage']:.1f}%")
        
        if failed_samples:
            print(f"\nFailed Samples ({len(failed_samples)})")
            for sample in failed_samples[:3]:
                print(f"  - {sample['id']}: {sample['error'][:50]}")
            if len(failed_samples) > 3:
                print(f"  ... and {len(failed_samples) - 3} more")
        
        print("="*80 + "\n")
    
    def extract_all_splits(self, output_dir=None):
        """Extract features for all splits."""
        all_summaries = {}
        all_failed = {}
        
        for split in ["train", "dev", "test"]:
            if self.should_stop:
                self.logger.info("Extraction stopped by user")
                break
            
            print(f"\n{'#'*80}")
            print(f"# {split.upper()} SPLIT")
            print(f"{'#'*80}\n")
            
            summary, metrics_df, failed = self.extract_split(
                split=split,
                output_dir=output_dir,
                checkpoint_interval=50
            )
            
            if summary:
                self.print_summary(summary, failed)
                all_summaries[split] = summary
                all_failed[split] = failed
        
        return all_summaries, all_failed


def main():
    """Main function with GPU support."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimized MediaPipe feature extraction with GPU support"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of worker processes (default: auto-detect)"
    )
    parser.add_argument(
        "--split", type=str, default="all",
        choices=["train", "dev", "test", "all"],
        help="Which split to extract (default: all)"
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Disable GPU acceleration"
    )
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = OptimizedParallelExtractor(
        num_workers=args.workers,
        use_gpu=not args.no_gpu
    )
    
    print("\n" + "="*80)
    print("OPTIMIZED PARALLEL FEATURE EXTRACTION")
    print("="*80)
    print(f"\nDataset: RWTH-PHOENIX-Weather 2014 SI5")
    print(f"Workers: {extractor.num_workers}")
    print(f"GPU: {'Enabled' if extractor.use_gpu else 'Disabled'}")
    
    if extractor.use_gpu:
        print(f"\nGPU Information:")
        for gpu in extractor.gpu_info["devices"]:
            print(f"  GPU {gpu['id']}: {gpu['name']}")
            if 'memory_total' in gpu:
                print(f"    Memory: {gpu['memory_free']:.1f}/{gpu['memory_total']:.1f} GB free")
            if 'gpu_load' in gpu:
                print(f"    Load: {gpu['gpu_load']:.1f}%")
    
    print(f"\nSamples to process:")
    print(f"  Train: 4,376 samples")
    print(f"  Dev: 111 samples")
    print(f"  Test: 180 samples")
    print(f"  Total: 4,667 samples")
    
    if extractor.use_gpu:
        print(f"\nEstimated time: ~{5 / extractor.num_workers:.1f} hours (GPU)")
    else:
        print(f"\nEstimated time: ~{10 / max(extractor.num_workers / 2, 1):.1f} hours (CPU)")
    
    print("="*80 + "\n")
    
    try:
        if args.split == "all":
            all_summaries, all_failed = extractor.extract_all_splits()
        else:
            summary, metrics_df, failed = extractor.extract_split(split=args.split)
            extractor.print_summary(summary, failed)
        
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


if __name__ == "__main__":
    main()