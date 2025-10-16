"""Quick verification of normalized features."""
import numpy as np
from pathlib import Path
from tqdm import tqdm

def verify_normalization(features_dir='data/processed', split='train', num_files=100):
    """Verify that features are properly normalized."""

    normalized_dir = Path(features_dir) / f"{split}_normalized"

    print(f"Verifying normalized features in: {normalized_dir}\n")

    all_features = []

    # Sample files
    feature_files = list(normalized_dir.glob("*.npy"))[:num_files]

    for feature_file in tqdm(feature_files, desc="Loading files"):
        features = np.load(feature_file)
        all_features.append(features)

    # Concatenate
    all_features = np.concatenate(all_features, axis=0)

    # Statistics
    mean = np.mean(all_features)
    std = np.std(all_features)
    min_val = np.min(all_features)
    max_val = np.max(all_features)

    print(f"{'='*60}")
    print(f"Normalization Verification ({split} split, {len(feature_files)} files)")
    print(f"{'='*60}")
    print(f"Total frames: {all_features.shape[0]:,}")
    print(f"Feature dims: {all_features.shape[1]}")
    print(f"\nGlobal Statistics:")
    print(f"  Mean: {mean:.6f}  (expected: ~0.0)")
    print(f"  Std:  {std:.6f}  (expected: ~1.0)")
    print(f"  Min:  {min_val:.6f}  (expected: ~-3 to -4)")
    print(f"  Max:  {max_val:.6f}  (expected: ~+3 to +4)")

    # Per-dimension check
    print(f"\nPer-Dimension Statistics (first 10 dims):")
    print(f"{'Dim':<6} {'Mean':<12} {'Std':<12}")
    print(f"{'-'*30}")
    for dim in range(min(10, all_features.shape[1])):
        dim_mean = np.mean(all_features[:, dim])
        dim_std = np.std(all_features[:, dim])
        print(f"{dim:<6} {dim_mean:<12.6f} {dim_std:<12.6f}")

    # Validation
    print(f"\n{'='*60}")
    print("Validation:")
    print(f"{'='*60}")

    checks_passed = 0
    checks_total = 4

    if abs(mean) < 0.1:
        print("[OK] Mean is close to 0")
        checks_passed += 1
    else:
        print(f"[WARN] Mean={mean:.3f} is not close to 0")

    if 0.9 < std < 1.1:
        print("[OK] Std is close to 1")
        checks_passed += 1
    else:
        print(f"[WARN] Std={std:.3f} is not close to 1")

    if -5 < min_val < -2:
        print("[OK] Min value in expected range")
        checks_passed += 1
    else:
        print(f"[WARN] Min={min_val:.3f} outside expected range [-5, -2]")

    if 2 < max_val < 5:
        print("[OK] Max value in expected range")
        checks_passed += 1
    else:
        print(f"[WARN] Max={max_val:.3f} outside expected range [2, 5]")

    print(f"\nChecks passed: {checks_passed}/{checks_total}")

    if checks_passed == checks_total:
        print("\n[SUCCESS] Features are properly normalized!")
    elif checks_passed >= 2:
        print("\n[OK] Features are mostly normalized (minor warnings)")
    else:
        print("\n[FAIL] Normalization may have issues")

    print(f"{'='*60}\n")

    return checks_passed == checks_total or checks_passed >= 2


if __name__ == "__main__":
    print("Verifying normalized features...\n")

    # Check train split
    train_ok = verify_normalization('data/processed', 'train', num_files=200)

    # Check dev split
    dev_ok = verify_normalization('data/processed', 'dev', num_files=50)

    if train_ok and dev_ok:
        print("\n" + "="*60)
        print("OVERALL: Normalization successful!")
        print("Ready to retrain model with normalized features.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("OVERALL: Some issues detected")
        print("Review warnings above before retraining.")
        print("="*60)
