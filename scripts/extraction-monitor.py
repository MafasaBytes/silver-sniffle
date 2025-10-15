#!/usr/bin/env python3
"""
Real-time monitoring dashboard for feature extraction progress.
Run this in a separate terminal while extraction is running.
"""

import os
import sys
import time
import json
import psutil
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from collections import deque
import threading

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data" / "processed"


class ExtractionMonitor:
    """Real-time monitoring dashboard for feature extraction."""

    def __init__(self):
        self.running = True
        self.data_dir = DATA_DIR
        self.stats_history = deque(maxlen=60)  # Keep 60 seconds of history
        self.gpu_history = deque(maxlen=60)
        self.fps_history = deque(maxlen=100)

        # Expected totals
        self.expected_totals = {
            "train": 4376,
            "dev": 111,
            "test": 180
        }

    def get_extraction_status(self):
        """Get current extraction status from files."""
        status = {}

        for split in ["train", "dev", "test"]:
            split_dir = self.data_dir / split
            status[split] = {
                "completed": 0,
                "expected": self.expected_totals[split],
                "latest_summary": None,
                "latest_checkpoint": None
            }

            if split_dir.exists():
                # Count completed files
                npy_files = list(split_dir.glob("*.npy"))
                status[split]["completed"] = len(npy_files)

                # Check for latest summary (try multiple file patterns)
                summary_files = [
                    split_dir / "extraction_summary_yolov8.json",
                    split_dir / "extraction_summary_gpu.json",
                    split_dir / "extraction_summary_parallel.json"
                ]
                
                for summary_file in summary_files:
                    if summary_file.exists():
                        break
                else:
                    summary_file = None

                if summary_file and summary_file.exists():
                    try:
                        with open(summary_file, "r") as f:
                            status[split]["latest_summary"] = json.load(f)
                    except:
                        pass

                # Find latest checkpoint
                checkpoints = sorted(split_dir.glob("checkpoint_*.json"))
                if checkpoints:
                    try:
                        with open(checkpoints[-1], "r") as f:
                            status[split]["latest_checkpoint"] = json.load(f)
                    except:
                        pass

        return status

    def get_system_stats(self):
        """Get current system statistics."""
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage(str(self.data_dir)).percent if self.data_dir.exists() else 0,
            "processes": []
        }

        # Find worker processes
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                pinfo = proc.info
                if 'python' in pinfo.get('name', '').lower():
                    try:
                        cmdline = proc.cmdline()
                        # Look for extraction scripts
                        if any('extract_full_dataset' in arg or 
                               'extract_features_yolov8' in arg or
                               'SLR_Worker' in arg 
                               for arg in cmdline):
                            stats["processes"].append({
                                "pid": pinfo['pid'],
                                "name": pinfo['name'],
                                "cpu": proc.cpu_percent(interval=0.1),
                                "memory": proc.memory_percent()
                            })
                    except:
                        pass
            except:
                pass

        return stats

    def get_gpu_stats(self):
        """Get GPU statistics."""
        if not HAS_GPUTIL:
            return []

        try:
            gpus = GPUtil.getGPUs()
            gpu_stats = []
            for gpu in gpus:
                gpu_stats.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load * 100,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0,
                    "temperature": gpu.temperature
                })
            return gpu_stats
        except:
            return []

    def format_time(self, seconds):
        """Format seconds to human-readable time."""
        if seconds is None or seconds < 0:
            return "N/A"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def create_progress_bar(self, percentage, width=40):
        """Create a text-based progress bar."""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def run_text_mode(self):
        """Simple text-based monitoring."""
        print("=" * 80)
        print("FEATURE EXTRACTION MONITOR")
        print("Windows-compatible text mode")
        print("=" * 80)
        print("\nPress Ctrl+C to quit\n")
        time.sleep(2)

        while self.running:
            try:
                self.clear_screen()

                extraction_status = self.get_extraction_status()
                system_stats = self.get_system_stats()
                gpu_stats = self.get_gpu_stats()

                # Header
                print("=" * 80)
                print(" " * 20 + "FEATURE EXTRACTION MONITOR")
                print(" " * 25 + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print("=" * 80)
                print()

                # Extraction Progress
                print("EXTRACTION PROGRESS:")
                print("-" * 80)

                total_completed = 0
                total_expected = 0

                for split in ["train", "dev", "test"]:
                    status = extraction_status[split]
                    completed = status["completed"]
                    expected = status["expected"]
                    total_completed += completed
                    total_expected += expected

                    percentage = (completed / expected * 100) if expected > 0 else 0
                    bar = self.create_progress_bar(percentage, 30)

                    # Color-coded status
                    if percentage == 100:
                        status_symbol = "✓"
                        status_text = "COMPLETE"
                    elif percentage > 0:
                        status_symbol = "⟳"
                        status_text = "RUNNING "
                    else:
                        status_symbol = "○"
                        status_text = "PENDING "

                    print(f"{status_symbol} {split.upper():5s}: {bar} {completed:4d}/{expected:4d} ({percentage:5.1f}%) {status_text}")

                    # Show checkpoint age if active
                    if status["latest_checkpoint"]:
                        try:
                            checkpoint_time = datetime.fromisoformat(status["latest_checkpoint"]["timestamp"])
                            age = (datetime.now() - checkpoint_time).total_seconds()
                            if age < 300:  # Less than 5 minutes old
                                print(f"          Last checkpoint: {int(age)}s ago")
                        except:
                            pass

                print()
                overall_percentage = (total_completed / total_expected * 100) if total_expected > 0 else 0
                overall_bar = self.create_progress_bar(overall_percentage, 30)
                print(f"  TOTAL : {overall_bar} {total_completed:4d}/{total_expected:4d} ({overall_percentage:5.1f}%)")

                # System Resources
                print()
                print("=" * 80)
                print("SYSTEM RESOURCES:")
                print("-" * 80)

                cpu_bar = self.create_progress_bar(system_stats["cpu_percent"], 20)
                mem_bar = self.create_progress_bar(system_stats["memory_percent"], 20)
                disk_bar = self.create_progress_bar(system_stats["disk_usage"], 20)

                print(f"CPU Usage    : {cpu_bar} {system_stats['cpu_percent']:5.1f}%")
                print(f"Memory Usage : {mem_bar} {system_stats['memory_percent']:5.1f}%")
                print(f"Disk Usage   : {disk_bar} {system_stats['disk_usage']:5.1f}%")

                # Worker processes
                if system_stats["processes"]:
                    print(f"\nActive Workers: {len(system_stats['processes'])}")
                    for proc in system_stats["processes"][:3]:  # Show first 3
                        print(f"  PID {proc['pid']}: CPU={proc['cpu']:.1f}% | Memory={proc['memory']:.1f}%")
                else:
                    print("\nActive Workers: 0 (No extraction running)")

                # GPU Stats
                if gpu_stats:
                    print()
                    print("=" * 80)
                    print("GPU STATUS:")
                    print("-" * 80)

                    for gpu in gpu_stats:
                        gpu_load_bar = self.create_progress_bar(gpu["load"], 20)
                        gpu_mem_bar = self.create_progress_bar(gpu["memory_percent"], 20)

                        print(f"GPU {gpu['id']}: {gpu['name']}")
                        print(f"  Load   : {gpu_load_bar} {gpu['load']:5.1f}%")
                        print(f"  Memory : {gpu_mem_bar} {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB ({gpu['memory_percent']:.1f}%)")
                        print(f"  Temp   : {gpu['temperature']}°C")
                elif HAS_GPUTIL:
                    print()
                    print("=" * 80)
                    print("GPU STATUS: No GPUs detected")

                # Footer
                print()
                print("=" * 80)
                print("Refreshing in 5 seconds... (Press Ctrl+C to quit)")
                print("=" * 80)

                time.sleep(5)

            except KeyboardInterrupt:
                self.running = False
                print("\n\nMonitoring stopped.")
                break
            except Exception as e:
                print(f"\nError updating display: {e}")
                time.sleep(5)

    def run(self):
        """Run the monitoring dashboard."""
        try:
            self.run_text_mode()
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main monitoring function."""
    print("Starting Feature Extraction Monitor...")
    print("Windows-compatible text mode")
    print("-" * 80)

    if not HAS_GPUTIL:
        print("Warning: GPUtil not found. GPU monitoring will be disabled.")
        print("Install with: pip install gputil")
        print("-" * 80)

    monitor = ExtractionMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
