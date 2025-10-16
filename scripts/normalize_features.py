"""
Normalize extracted features for training.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle


def compute_normalization_stats(features_dir, split='train'):
    """
    Compute mean and std for feature normalization.
    
    Args:
        features_dir: Directory containing .npy feature files
        split: Dataset split to use for computing stats
    
    Returns:
        (mean, std) as numpy arrays
    """
    features_dir = Path(features_dir) / split
    
    all_features = []
    
    print(f"Computing normalization statistics from {split} split...")
    
    for feature_file in tqdm(list(features_dir.glob("*.npy"))):
        features = np.load(feature_file)  # (T, 177)
        all_features.append(features)
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)  # (total_frames, 177)
    
    # Compute statistics
    mean = np.mean(all_features, axis=0)  # (177,)
    std = np.std(all_features, axis=0)  # (177,)
    
    # Prevent division by zero
    std = np.where(std < 1e-8, 1.0, std)
    
    print(f"\nNormalization statistics:")
    print(f"  Mean range: [{mean.min():.2f}, {mean.max():.2f}]")
    print(f"  Std range: [{std.min():.2f}, {std.max():.2f}]")
    print(f"  Total frames: {all_features.shape[0]:,}")
    
    return mean, std


def normalize_features(features_dir, mean, std, splits=['train', 'dev', 'test']):
    """
    Normalize features and save to separate directory (preserves originals).

    Args:
        features_dir: Root directory containing split folders
        mean: Mean for normalization
        std: Std for normalization
        splits: List of splits to normalize
    """
    features_dir = Path(features_dir)

    for split in splits:
        split_dir = features_dir / split
        output_dir = features_dir / f"{split}_normalized"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nNormalizing {split} split...")
        print(f"  Input:  {split_dir}")
        print(f"  Output: {output_dir}")

        feature_files = list(split_dir.glob("*.npy"))

        for feature_file in tqdm(feature_files):
            # Load features
            features = np.load(feature_file)  # (T, 177)

            # Normalize
            features_normalized = (features - mean) / std

            # Save to normalized directory
            output_file = output_dir / feature_file.name
            np.save(output_file, features_normalized)

        print(f"Normalized {len(feature_files)} files")


def main():
    features_root = "data/processed"
    
    # Compute normalization statistics from training set
    mean, std = compute_normalization_stats(features_root, split='train')
    
    # Save statistics
    stats_file = Path(features_root) / 'normalization_stats.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)
    print(f"\nSaved normalization stats to: {stats_file}")
    
    # Normalize all splits
    normalize_features(features_root, mean, std, splits=['train', 'dev', 'test'])
    
    print("\nFeature normalization complete!")


if __name__ == "__main__":
    main()