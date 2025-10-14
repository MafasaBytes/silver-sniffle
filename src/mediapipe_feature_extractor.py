"""
MediaPipe Feature Extractor for RWTH-PHOENIX-Weather 2014
Extracts pose, hand, and face landmarks using MediaPipe Holistic.
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import time
import psutil
import os
from tqdm import tqdm


class MediaPipeFeatureExtractor:
    """Extract MediaPipe Holistic features from video frames."""

    def __init__(self):
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Create holistic model
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,  # 0, 1, or 2 (higher = more accurate but slower)
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Feature dimensions
        self.pose_landmarks = 33  # 33 pose landmarks
        self.face_landmarks = 468  # 468 face landmarks
        self.hand_landmarks = 21  # 21 landmarks per hand

    def extract_landmarks(self, image):
        """
        Extract landmarks from a single image.

        Args:
            image: RGB image (numpy array)

        Returns:
            dict with pose, face, left_hand, right_hand landmarks
        """
        # Process image
        results = self.holistic.process(image)

        landmarks = {}

        # Extract pose landmarks (33 points x 4 values = 132 features)
        if results.pose_landmarks:
            landmarks['pose'] = np.array([
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ]).flatten()  # Shape: (132,)
        else:
            landmarks['pose'] = np.zeros(self.pose_landmarks * 4)

        # Extract face landmarks (468 points x 3 values = 1404 features)
        if results.face_landmarks:
            landmarks['face'] = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.face_landmarks.landmark
            ]).flatten()  # Shape: (1404,)
        else:
            landmarks['face'] = np.zeros(self.face_landmarks * 3)

        # Extract left hand landmarks (21 points x 3 values = 63 features)
        if results.left_hand_landmarks:
            landmarks['left_hand'] = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.left_hand_landmarks.landmark
            ]).flatten()  # Shape: (63,)
        else:
            landmarks['left_hand'] = np.zeros(self.hand_landmarks * 3)

        # Extract right hand landmarks (21 points x 3 values = 63 features)
        if results.right_hand_landmarks:
            landmarks['right_hand'] = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.right_hand_landmarks.landmark
            ]).flatten()  # Shape: (63,)
        else:
            landmarks['right_hand'] = np.zeros(self.hand_landmarks * 3)

        # Concatenate all features: 132 + 1404 + 63 + 63 = 1662 features
        features = np.concatenate([
            landmarks['pose'],
            landmarks['face'],
            landmarks['left_hand'],
            landmarks['right_hand']
        ])

        return features, results

    def visualize_landmarks(self, image, results):
        """Draw landmarks on image for visualization."""
        annotated_image = image.copy()

        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Draw face landmarks
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        # Draw hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )

        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )

        return annotated_image

    def process_video_frames(self, frame_folder):
        """
        Process all frames in a folder.

        Args:
            frame_folder: Path to folder containing frames

        Returns:
            features: numpy array of shape (num_frames, 1662)
            processing_time: time taken in seconds
            fps: frames per second processed
        """
        frame_folder = Path(frame_folder)
        frame_files = sorted(frame_folder.glob("*.png"))

        if len(frame_files) == 0:
            raise ValueError(f"No frames found in {frame_folder}")

        features_list = []
        start_time = time.time()

        # Get memory before processing
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        for frame_file in frame_files:
            # Read frame
            image = cv2.imread(str(frame_file))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract features
            features, _ = self.extract_landmarks(image_rgb)
            features_list.append(features)

        end_time = time.time()

        # Get memory after processing
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        processing_time = end_time - start_time
        fps = len(frame_files) / processing_time

        features_array = np.array(features_list)

        return {
            "features": features_array,
            "processing_time": processing_time,
            "fps": fps,
            "num_frames": len(frame_files),
            "memory_used_mb": mem_used,
            "feature_shape": features_array.shape
        }

    def demo_single_video(self, frame_folder, output_dir="outputs", visualize=True):
        """
        Demo: Process a single video and optionally save visualizations.

        Args:
            frame_folder: Path to folder with frames
            output_dir: Directory to save outputs
            visualize: Whether to create visualizations
        """
        frame_folder = Path(frame_folder)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"\nProcessing: {frame_folder.name}")
        print("=" * 60)

        # Process video
        result = self.process_video_frames(frame_folder)

        print(f"Frames processed: {result['num_frames']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"FPS: {result['fps']:.2f}")
        print(f"Memory used: {result['memory_used_mb']:.2f} MB")
        print(f"Feature shape: {result['feature_shape']}")
        print(f"Feature size: {result['features'].nbytes / 1024:.2f} KB")

        # Save features
        feature_file = output_dir / f"{frame_folder.parent.name}_features.npy"
        np.save(feature_file, result['features'])
        print(f"\nFeatures saved to: {feature_file}")

        # Visualize if requested
        if visualize:
            self._create_visualizations(frame_folder, output_dir, result)

        return result

    def _create_visualizations(self, frame_folder, output_dir, result):
        """Create and save visualizations."""
        frame_files = sorted(Path(frame_folder).glob("*.png"))

        # Visualize first frame, middle frame, and last frame
        indices = [0, len(frame_files) // 2, len(frame_files) - 1]

        for idx in indices:
            frame_file = frame_files[idx]
            image = cv2.imread(str(frame_file))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract landmarks
            _, results = self.extract_landmarks(image_rgb)

            # Draw landmarks
            annotated = self.visualize_landmarks(image_rgb, results)

            # Save
            output_file = output_dir / f"{frame_folder.parent.name}_frame_{idx:03d}_annotated.png"
            cv2.imwrite(str(output_file), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        print(f"Visualizations saved to: {output_dir}")

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'holistic'):
            self.holistic.close()


def main():
    """Demo extraction on a single video."""
    import pandas as pd

    # Load train data
    data_root = Path("data/raw_data/phoenix-2014-signerindependent-SI5")
    train_csv = data_root / "annotations" / "manual" / "train.SI5.corpus.csv"
    train_df = pd.read_csv(train_csv, delimiter="|")

    # Get first video
    first_sample = train_df.iloc[0]
    frame_folder = data_root / "features" / "fullFrame-210x260px" / "train" / first_sample["folder"].replace("/*.png", "")

    print(f"Sample ID: {first_sample['id']}")
    print(f"Signer: {first_sample['signer']}")
    print(f"Annotation: {first_sample['annotation']}")

    # Extract features
    extractor = MediaPipeFeatureExtractor()
    result = extractor.demo_single_video(frame_folder, visualize=True)

    print("\n" + "=" * 60)
    print("Extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
