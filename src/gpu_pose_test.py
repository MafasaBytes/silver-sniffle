
# GPU-Accelerated Pose Estimation Test
# Using YOLOv8 Pose (Ultralytics) - The most practical solution

import torch
import cv2
import time
import numpy as np
from pathlib import Path
import json

def test_gpu_availability():
    print('=' * 50)
    print('GPU AVAILABILITY TEST')
    print('=' * 50)
    
    cuda_available = torch.cuda.is_available()
    print(f'CUDA Available: {cuda_available}')
    
    if cuda_available:
        print(f'CUDA Version: {torch.version.cuda}')
        print(f'GPU Count: {torch.cuda.device_count()}')
        print(f'GPU Name: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
        
        # Test GPU computation
        x = torch.randn(1000, 1000).cuda()
        start = time.time()
        _ = torch.matmul(x, x)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f'GPU Matrix Multiplication (1000x1000): {gpu_time*1000:.2f}ms')
        
        # CPU comparison
        x_cpu = x.cpu()
        start = time.time()
        _ = torch.matmul(x_cpu, x_cpu)
        cpu_time = time.time() - start
        print(f'CPU Matrix Multiplication (1000x1000): {cpu_time*1000:.2f}ms')
        print(f'GPU Speedup: {cpu_time/gpu_time:.2f}x')
    
    return cuda_available

if __name__ == '__main__':
    test_gpu_availability()
