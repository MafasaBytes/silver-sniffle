"""
Diagnostic script to analyze input feature quality.
Checks for normalization issues, outliers, NaN/Inf values.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from phoenix_dataset import create_dataloaders


def analyze_features(dataloader, split_name='train', num_batches=10):
    """Analyze feature statistics from dataloader."""
    print(f"\n{'='*60}")
    print(f"Feature Analysis: {split_name.upper()} set")
    print(f"{'='*60}\n")

    all_features_list = []
    all_lengths = []
    batch_count = 0

    print("Collecting features...")
    for batch in dataloader:
        features = batch['features']  # (B, T, 177)
        lengths = batch['feature_lengths']  # (B,)

        # Extract only valid frames (remove padding)
        for i in range(features.size(0)):
            valid_length = lengths[i].item()
            valid_features = features[i, :valid_length, :]  # (T_valid, 177)
            all_features_list.append(valid_features)

        all_lengths.extend(lengths.tolist())

        batch_count += 1
        if batch_count >= num_batches:
            break

    # Concatenate all features (flattening sequences)
    all_features = torch.cat(all_features_list, dim=0)  # (Total_Frames, 177)
    print(f"Collected {len(all_features_list)} sequences")
    print(f"Total frames: {all_features.shape[0]:,}")
    print(f"Feature dimensions: {all_features.shape[1]}")

    # Global statistics
    print(f"\n{'-'*60}")
    print("Global Statistics")
    print(f"{'-'*60}")
    print(f"Mean: {all_features.mean().item():.6f}")
    print(f"Std:  {all_features.std().item():.6f}")
    print(f"Min:  {all_features.min().item():.6f}")
    print(f"Max:  {all_features.max().item():.6f}")

    # Check for problematic values
    print(f"\n{'-'*60}")
    print("Data Quality Checks")
    print(f"{'-'*60}")
    nan_count = torch.isnan(all_features).sum().item()
    inf_count = torch.isinf(all_features).sum().item()
    zero_count = (all_features == 0).sum().item()
    total_values = all_features.numel()

    print(f"NaN values: {nan_count:,} ({nan_count/total_values*100:.2f}%)")
    print(f"Inf values: {inf_count:,} ({inf_count/total_values*100:.2f}%)")
    print(f"Zero values: {zero_count:,} ({zero_count/total_values*100:.2f}%)")

    # Check for zero frames
    frame_sums = all_features.sum(dim=-1)  # (Total_Frames,)
    zero_frames = (frame_sums == 0).sum().item()
    total_frames = all_features.shape[0]
    print(f"Zero frames (all features=0): {zero_frames:,} ({zero_frames/total_frames*100:.2f}%)")

    # Per-dimension statistics
    print(f"\n{'-'*60}")
    print("Per-Dimension Analysis (first 20 dims)")
    print(f"{'-'*60}")
    print(f"{'Dim':<6} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print(f"{'-'*60}")

    for dim in range(min(20, all_features.shape[-1])):
        dim_values = all_features[:, dim]  # (Total_Frames,)
        print(f"{dim:<6} {dim_values.mean():<12.6f} {dim_values.std():<12.6f} "
              f"{dim_values.min():<12.6f} {dim_values.max():<12.6f}")

    # Check for constant dimensions
    print(f"\n{'-'*60}")
    print("Constant/Dead Dimensions")
    print(f"{'-'*60}")
    constant_dims = []
    for dim in range(all_features.shape[-1]):
        dim_values = all_features[:, dim]  # (Total_Frames,)
        if dim_values.std().item() < 1e-6:
            constant_dims.append(dim)

    if constant_dims:
        print(f"Found {len(constant_dims)} constant dimensions: {constant_dims[:10]}...")
    else:
        print("No constant dimensions found.")

    # Sequence length statistics
    print(f"\n{'-'*60}")
    print("Sequence Length Statistics")
    print(f"{'-'*60}")
    lengths_array = np.array(all_lengths)
    print(f"Mean length: {lengths_array.mean():.1f}")
    print(f"Std length:  {lengths_array.std():.1f}")
    print(f"Min length:  {lengths_array.min()}")
    print(f"Max length:  {lengths_array.max()}")

    # Distribution analysis
    print(f"\n{'-'*60}")
    print("Value Distribution")
    print(f"{'-'*60}")
    flat_features = all_features.flatten()
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("Percentiles:")
    for p in percentiles:
        val = np.percentile(flat_features.numpy(), p)
        print(f"  {p:>2}%: {val:>10.4f}")

    # Check for outliers (values beyond 3 std)
    mean = all_features.mean()
    std = all_features.std()
    outliers = ((all_features < mean - 3*std) | (all_features > mean + 3*std)).sum().item()
    print(f"\nOutliers (>3 std from mean): {outliers:,} ({outliers/total_values*100:.2f}%)")

    print(f"\n{'='*60}")
    print("Feature Analysis Complete")
    print(f"{'='*60}\n")

    return {
        'mean': all_features.mean().item(),
        'std': all_features.std().item(),
        'min': all_features.min().item(),
        'max': all_features.max().item(),
        'nan_count': nan_count,
        'inf_count': inf_count,
        'zero_frames': zero_frames,
        'constant_dims': constant_dims,
        'outliers': outliers,
    }


def main():
    print("\nLoading datasets...")

    # Load dataloaders
    train_loader, dev_loader, test_loader = create_dataloaders(
        data_root='data/raw_data/phoenix-2014-signerindependent-SI5',
        features_root='data/processed',
        batch_size=16,
        num_workers=0,
    )

    # Analyze training set
    train_stats = analyze_features(train_loader, 'train', num_batches=50)

    # Analyze dev set
    dev_stats = analyze_features(dev_loader, 'dev', num_batches=10)

    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    # Check for critical issues
    issues = []

    if train_stats['nan_count'] > 0:
        issues.append(f"[X] NaN values detected ({train_stats['nan_count']:,})")

    if train_stats['inf_count'] > 0:
        issues.append(f"[X] Inf values detected ({train_stats['inf_count']:,})")

    if abs(train_stats['mean']) > 1.0:
        issues.append(f"[X] Features not normalized (mean={train_stats['mean']:.2f}, expected ~0)")

    if train_stats['std'] < 0.5 or train_stats['std'] > 2.0:
        issues.append(f"[X] Unusual std deviation (std={train_stats['std']:.2f}, expected ~1)")

    if len(train_stats['constant_dims']) > 0:
        issues.append(f"[!] {len(train_stats['constant_dims'])} constant/dead dimensions")

    if train_stats['outliers'] > train_stats['mean'] * 0.05:  # >5% outliers
        issues.append(f"[!] High outlier rate ({train_stats['outliers']/(train_stats['mean']*100):.1f}%)")

    if issues:
        print("\nCRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\nRECOMMENDATION: Features require normalization/preprocessing")
    else:
        print("\n[OK] No critical issues detected")
        print("Features appear to be properly normalized")

    print("\nNote: Even if no critical issues are found, consider:")
    print("  1. Applying z-score normalization (mean=0, std=1)")
    print("  2. Checking for frame-level vs sequence-level normalization")
    print("  3. Removing or imputing zero frames")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
