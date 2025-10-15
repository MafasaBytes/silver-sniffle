#!/usr/bin/env python3
"""
Run YOLOv8 GPU-accelerated feature extraction with proper monitoring.

This script:
1. Checks GPU availability
2. Runs the parallel YOLOv8 extraction
3. Monitors progress in real-time
4. Handles interruptions gracefully
"""

import subprocess
import sys
import os
from pathlib import Path
import time
import threading
import signal

# Get project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

# Global flag for clean shutdown
should_stop = False


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global should_stop
    print("\n\nReceived shutdown signal. Stopping extraction...")
    should_stop = True


def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return True
        else:
            print("✗ No GPU available. Extraction will be slower.")
            return False
    except ImportError:
        print("✗ PyTorch not installed. Please install it first.")
        return False


def run_extraction():
    """Run the YOLOv8 extraction process."""
    # Command to run
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "extract_features_yolov8_gpu.py"),
        "--split", "all",
        "--batch-size", "16",  # Increased batch size for better GPU utilization
        "--workers", "2",      # 2 workers for optimal GPU sharing
        "--parallel",          # Enable parallel processing
        "--checkpoint-interval", "50"
    ]
    
    print("\nStarting YOLOv8 extraction with command:")
    print(" ".join(cmd))
    print("\n" + "="*80 + "\n")
    
    # Run the extraction
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Stream output in real-time
    try:
        for line in iter(process.stdout.readline, ''):
            if should_stop:
                process.terminate()
                break
            print(line, end='', flush=True)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Terminating extraction...")
        process.terminate()
    
    # Wait for process to finish
    process.wait()
    
    return process.returncode


def run_monitor():
    """Run the monitoring script in a separate thread."""
    def monitor_thread():
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "extraction-monitor.py")
        ]
        subprocess.run(cmd, capture_output=True)
    
    monitor = threading.Thread(target=monitor_thread, daemon=True)
    monitor.start()


def main():
    """Main function."""
    print("="*80)
    print("YOLOV8 GPU FEATURE EXTRACTION")
    print("="*80)
    
    # Setup signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Check dependencies
    try:
        import ultralytics
        print("✓ Ultralytics YOLO installed")
    except ImportError:
        print("✗ Ultralytics not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    
    try:
        import GPUtil
        print("✓ GPUtil installed (GPU monitoring enabled)")
    except ImportError:
        print("! GPUtil not installed. GPU monitoring will be limited.")
        print("  Install with: pip install gputil")
    
    print("\n" + "="*80 + "\n")
    
    # Ask user to confirm
    response = input("Start extraction? (y/n): ")
    if response.lower() != 'y':
        print("Extraction cancelled.")
        return
    
    print("\nTIP: Run extraction-monitor.py in another terminal to see real-time progress!")
    print("\n" + "="*80 + "\n")
    
    # Run extraction
    start_time = time.time()
    return_code = run_extraction()
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("EXTRACTION FINISHED")
    print("="*80)
    print(f"Total time: {elapsed_time/3600:.2f} hours")
    print(f"Exit code: {return_code}")
    
    if return_code == 0:
        print("\n✓ Extraction completed successfully!")
        print(f"  Features saved to: {PROJECT_ROOT / 'data' / 'processed'}")
        print(f"  Logs saved to: {PROJECT_ROOT / 'data' / 'processed' / 'logs'}")
    else:
        print("\n✗ Extraction failed or was interrupted.")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
