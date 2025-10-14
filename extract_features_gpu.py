#!/usr/bin/env python
# GPU-Accelerated Feature Extractor for RWTH-PHOENIX Dataset

import torch
import cv2
import numpy as np
import json
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import time
import argparse
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class RWTHFeatureExtractor:
    def __init__(self, yolo_model='yolov8m-pose.pt', use_hands=True):
        print('Initializing RWTH Feature Extractor...')
        
        # GPU check
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f'[GPU] {torch.cuda.get_device_name(0)}')
        
        # YOLOv8 for body pose (GPU)
        print(f'Loading {yolo_model}...')
        self.pose_model = YOLO(yolo_model)
        if self.device == 'cuda':
            self.pose_model.to('cuda')
        
        # MediaPipe for hands (CPU)
        self.use_hands = use_hands
        if use_hands:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5
            )
        
        self.stats = {
            'sequences_processed': 0,
            'frames_processed': 0,
            'gpu_time': 0,
            'cpu_time': 0
        }

if __name__ == '__main__':
    print('RWTH Feature Extractor Ready')
    print('Usage: python extract_features_gpu.py --input_dir [path] --output_dir [path]')
