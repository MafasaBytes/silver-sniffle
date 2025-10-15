"""
Comprehensive Feature Validation Script for RWTH-PHOENIX-Weather 2014 SI5

Validates extracted features across train/dev/test splits:
- Dataset completeness
- Feature quality (shape, NaN, inf, value ranges)
- Sequence length statistics
- Dataset loader verification
- Performance analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from phoenix_dataset import PhoenixFeatureDataset, create_dataloaders


class FeatureValidator:
    """Comprehensive validator for extracted features."""

    def __init__(self, data_root: str, features_root: str):
        self.data_root = Path(data_root)
        self.features_root = Path(features_root)
        self.splits = ['train', 'dev', 'test']
        self.expected_feature_dim = 177  # 51 body + 126 hands

        # Expected sample counts from SI5 dataset
        self.expected_counts = {
            'train': 4259,
            'dev': 111,
            'test': 180
        }

        self.results = {
            'summary': {},
            'splits': {},
            'quality': {},
            'sequence_stats': {},
            'loader_test': {},
            'performance': {},
            'recommendations': []
        }

    def validate_all(self) -> Dict:
        """Run all validation checks."""
        print("="*80)
        print("RWTH-PHOENIX-WEATHER 2014 SI5 - FEATURE EXTRACTION VALIDATION")
        print("="*80)
        print()

        # 1. Dataset completeness
        print("1. DATASET COMPLETENESS CHECK")
        print("-"*80)
        self._validate_completeness()
        print()

        # 2. Feature quality
        print("2. FEATURE QUALITY VALIDATION")
        print("-"*80)
        self._validate_quality()
        print()

        # 3. Sequence statistics
        print("3. SEQUENCE LENGTH ANALYSIS")
        print("-"*80)
        self._analyze_sequences()
        print()

        # 4. Dataset loader verification
        print("4. DATASET LOADER VERIFICATION")
        print("-"*80)
        self._verify_loader()
        print()

        # 5. Generate recommendations
        print("5. TRAINING RECOMMENDATIONS")
        print("-"*80)
        self._generate_recommendations()
        print()

        # Save results
        self._save_results()

        return self.results

    def _validate_completeness(self):
        """Validate dataset completeness across all splits."""
        total_samples = 0
        total_frames = 0
        total_size_mb = 0

        for split in self.splits:
            split_path = self.features_root / split

            # Load annotations
            annotations_file = (self.data_root / "annotations" / "manual" /
                              f"{split}.SI5.corpus.csv")
            if not annotations_file.exists():
                print(f"ERROR: Annotations file not found for {split}")
                continue

            annotations_df = pd.read_csv(annotations_file, delimiter="|")
            expected_count = len(annotations_df)

            # Count .npy files
            npy_files = list(split_path.glob("*.npy"))
            actual_count = len(npy_files)

            # Calculate storage
            split_size_mb = sum(f.stat().st_size for f in npy_files) / (1024**2)

            # Count frames
            split_frames = 0
            for npy_file in npy_files:
                try:
                    features = np.load(npy_file)
                    split_frames += features.shape[0]
                except Exception as e:
                    print(f"  WARNING: Failed to load {npy_file.name}: {e}")

            # Calculate completeness
            completeness_pct = (actual_count / expected_count) * 100

            # Store results
            self.results['splits'][split] = {
                'expected_samples': expected_count,
                'actual_samples': actual_count,
                'missing_samples': expected_count - actual_count,
                'completeness_pct': completeness_pct,
                'total_frames': split_frames,
                'storage_mb': split_size_mb
            }

            total_samples += actual_count
            total_frames += split_frames
            total_size_mb += split_size_mb

            # Print results
            status = "PASS" if completeness_pct == 100 else "FAIL"
            print(f"  {split.upper()}: {actual_count}/{expected_count} samples ({completeness_pct:.1f}%) [{status}]")
            print(f"    Frames: {split_frames:,} | Storage: {split_size_mb:.1f} MB")

            if completeness_pct < 100:
                print(f"    WARNING: {expected_count - actual_count} samples missing!")

        # Calculate overall compression
        raw_size_gb = 53  # From CLAUDE.md
        compression_ratio = (1 - (total_size_mb / 1024) / raw_size_gb) * 100

        self.results['summary'] = {
            'total_samples': total_samples,
            'expected_total': sum(self.expected_counts.values()),
            'total_frames': total_frames,
            'total_storage_mb': total_size_mb,
            'compression_ratio_pct': compression_ratio
        }

        print()
        print(f"  TOTAL: {total_samples}/{sum(self.expected_counts.values())} samples")
        print(f"  FRAMES: {total_frames:,} frames")
        print(f"  STORAGE: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
        print(f"  COMPRESSION: {compression_ratio:.1f}% reduction from {raw_size_gb}GB raw")

    def _validate_quality(self):
        """Validate feature quality: shape, NaN, inf, value ranges."""
        for split in self.splits:
            split_path = self.features_root / split
            npy_files = list(split_path.glob("*.npy"))

            if len(npy_files) == 0:
                print(f"  {split.upper()}: No files found")
                continue

            # Sample 10% of files for quality check (minimum 10, maximum 500)
            sample_size = min(500, max(10, len(npy_files) // 10))
            sample_files = np.random.choice(npy_files, size=sample_size, replace=False)

            quality_issues = {
                'shape_errors': [],
                'nan_samples': [],
                'inf_samples': [],
                'value_range_issues': []
            }

            feature_values = []

            for npy_file in sample_files:
                try:
                    features = np.load(npy_file)

                    # Check shape
                    if len(features.shape) != 2:
                        quality_issues['shape_errors'].append(
                            f"{npy_file.name}: wrong dims {features.shape}"
                        )
                        continue

                    if features.shape[1] != self.expected_feature_dim:
                        quality_issues['shape_errors'].append(
                            f"{npy_file.name}: expected {self.expected_feature_dim}, got {features.shape[1]}"
                        )
                        continue

                    # Check for NaN
                    if np.isnan(features).any():
                        nan_count = np.isnan(features).sum()
                        quality_issues['nan_samples'].append(
                            f"{npy_file.name}: {nan_count} NaN values"
                        )

                    # Check for inf
                    if np.isinf(features).any():
                        inf_count = np.isinf(features).sum()
                        quality_issues['inf_samples'].append(
                            f"{npy_file.name}: {inf_count} inf values"
                        )

                    # Collect valid feature values for statistics
                    valid_features = features[~np.isnan(features) & ~np.isinf(features)]
                    if len(valid_features) > 0:
                        feature_values.extend(valid_features.flatten().tolist())

                except Exception as e:
                    quality_issues['shape_errors'].append(f"{npy_file.name}: load error - {e}")

            # Calculate feature statistics
            if len(feature_values) > 0:
                feature_values = np.array(feature_values)
                feature_stats = {
                    'min': float(np.min(feature_values)),
                    'max': float(np.max(feature_values)),
                    'mean': float(np.mean(feature_values)),
                    'std': float(np.std(feature_values)),
                    'median': float(np.median(feature_values)),
                    'q25': float(np.percentile(feature_values, 25)),
                    'q75': float(np.percentile(feature_values, 75))
                }
            else:
                feature_stats = None

            # Count total issues
            total_issues = sum(len(v) for v in quality_issues.values())

            self.results['quality'][split] = {
                'samples_checked': sample_size,
                'total_issues': total_issues,
                'shape_errors': len(quality_issues['shape_errors']),
                'nan_samples': len(quality_issues['nan_samples']),
                'inf_samples': len(quality_issues['inf_samples']),
                'feature_stats': feature_stats
            }

            # Print results
            status = "PASS" if total_issues == 0 else "WARNING"
            print(f"  {split.upper()}: {sample_size} samples checked [{status}]")

            if total_issues > 0:
                for issue_type, issues in quality_issues.items():
                    if len(issues) > 0:
                        print(f"    {issue_type}: {len(issues)} samples")
                        for issue in issues[:3]:  # Show first 3
                            print(f"      - {issue}")
                        if len(issues) > 3:
                            print(f"      ... and {len(issues)-3} more")
            else:
                print(f"    No quality issues found")

            if feature_stats:
                print(f"    Feature value range: [{feature_stats['min']:.4f}, {feature_stats['max']:.4f}]")
                print(f"    Mean: {feature_stats['mean']:.4f} | Std: {feature_stats['std']:.4f}")

    def _analyze_sequences(self):
        """Analyze sequence length statistics across all splits."""
        for split in self.splits:
            split_path = self.features_root / split
            npy_files = list(split_path.glob("*.npy"))

            if len(npy_files) == 0:
                print(f"  {split.upper()}: No files found")
                continue

            sequence_lengths = []

            for npy_file in npy_files:
                try:
                    features = np.load(npy_file)
                    sequence_lengths.append(features.shape[0])
                except Exception as e:
                    print(f"  WARNING: Failed to load {npy_file.name}: {e}")

            if len(sequence_lengths) == 0:
                print(f"  {split.upper()}: No valid sequences")
                continue

            sequence_lengths = np.array(sequence_lengths)

            # Calculate statistics
            stats = {
                'min': int(np.min(sequence_lengths)),
                'max': int(np.max(sequence_lengths)),
                'mean': float(np.mean(sequence_lengths)),
                'median': float(np.median(sequence_lengths)),
                'std': float(np.std(sequence_lengths)),
                'p50': int(np.percentile(sequence_lengths, 50)),
                'p75': int(np.percentile(sequence_lengths, 75)),
                'p90': int(np.percentile(sequence_lengths, 90)),
                'p95': int(np.percentile(sequence_lengths, 95)),
                'p99': int(np.percentile(sequence_lengths, 99))
            }

            # Identify anomalies
            q1 = np.percentile(sequence_lengths, 25)
            q3 = np.percentile(sequence_lengths, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            anomalies = {
                'very_short': int(np.sum(sequence_lengths < lower_bound)),
                'very_long': int(np.sum(sequence_lengths > upper_bound))
            }

            self.results['sequence_stats'][split] = {
                'statistics': stats,
                'anomalies': anomalies
            }

            # Print results
            print(f"  {split.upper()}:")
            print(f"    Range: {stats['min']} - {stats['max']} frames")
            print(f"    Mean: {stats['mean']:.1f} | Median: {stats['median']:.1f} | Std: {stats['std']:.1f}")
            print(f"    Percentiles: P50={stats['p50']}, P75={stats['p75']}, P90={stats['p90']}, P95={stats['p95']}, P99={stats['p99']}")

            if anomalies['very_short'] > 0 or anomalies['very_long'] > 0:
                print(f"    Anomalies: {anomalies['very_short']} very short, {anomalies['very_long']} very long")

    def _verify_loader(self):
        """Verify dataset loader functionality."""
        try:
            # Create datasets
            train_dataset = PhoenixFeatureDataset(
                data_root=str(self.data_root),
                features_root=str(self.features_root),
                split="train",
            )

            dev_dataset = PhoenixFeatureDataset(
                data_root=str(self.data_root),
                features_root=str(self.features_root),
                split="dev",
            )

            test_dataset = PhoenixFeatureDataset(
                data_root=str(self.data_root),
                features_root=str(self.features_root),
                split="test",
            )

            # Test single samples
            print(f"  Train dataset: {len(train_dataset)} samples")
            print(f"  Dev dataset: {len(dev_dataset)} samples")
            print(f"  Test dataset: {len(test_dataset)} samples")
            print(f"  Vocabulary size: {len(train_dataset.vocab)} signs")

            # Test dataloader
            train_loader, dev_loader, test_loader = create_dataloaders(
                data_root=str(self.data_root),
                features_root=str(self.features_root),
                batch_size=4,
                num_workers=0,
            )

            print(f"\n  DataLoader batches: train={len(train_loader)}, dev={len(dev_loader)}, test={len(test_loader)}")

            # Test batch loading
            print("\n  Testing batch loading...")
            for split_name, loader in [('train', train_loader), ('dev', dev_loader), ('test', test_loader)]:
                try:
                    batch = next(iter(loader))
                    print(f"    {split_name}: features {batch['features'].shape}, targets {batch['targets'].shape}")

                    # Verify shapes
                    assert batch['features'].shape[2] == self.expected_feature_dim, \
                        f"Wrong feature dimension: {batch['features'].shape[2]}"
                    assert len(batch['feature_lengths']) == batch['features'].shape[0], \
                        "Batch size mismatch"
                except Exception as e:
                    print(f"    {split_name}: ERROR - {e}")

            self.results['loader_test'] = {
                'status': 'PASS',
                'train_samples': len(train_dataset),
                'dev_samples': len(dev_dataset),
                'test_samples': len(test_dataset),
                'vocab_size': len(train_dataset.vocab),
                'train_batches': len(train_loader),
                'dev_batches': len(dev_loader),
                'test_batches': len(test_loader)
            }

            print("\n  Dataset loader verification: PASS")

        except Exception as e:
            print(f"\n  Dataset loader verification: FAIL")
            print(f"    Error: {e}")
            self.results['loader_test'] = {
                'status': 'FAIL',
                'error': str(e)
            }

    def _generate_recommendations(self):
        """Generate training recommendations based on analysis."""
        recommendations = []

        # 1. Max sequence length recommendation
        if 'sequence_stats' in self.results and 'train' in self.results['sequence_stats']:
            train_stats = self.results['sequence_stats']['train']['statistics']
            p95 = train_stats['p95']
            p99 = train_stats['p99']

            # Recommend P95 or P99 based on VRAM constraints
            if p99 < 512:
                rec_seq_len = p99
                coverage = 99
            else:
                rec_seq_len = p95
                coverage = 95

            recommendations.append({
                'parameter': 'max_sequence_length',
                'value': rec_seq_len,
                'reason': f'Covers {coverage}% of training sequences'
            })

        # 2. Batch size recommendation (considering 8GB VRAM)
        # Rough estimate: seq_len * feature_dim * batch_size * 4 bytes (FP32)
        if 'sequence_stats' in self.results and 'train' in self.results['sequence_stats']:
            avg_seq_len = self.results['sequence_stats']['train']['statistics']['mean']

            # Conservative estimate: allocate 4GB for features + activations
            available_memory_gb = 4
            memory_per_sample_gb = (avg_seq_len * self.expected_feature_dim * 4) / (1024**3)

            # Account for gradient storage (2x) and LSTM hidden states (4x)
            total_multiplier = 6
            memory_per_sample_gb *= total_multiplier

            batch_size = int(available_memory_gb / memory_per_sample_gb)
            batch_size = max(1, min(batch_size, 32))  # Clamp between 1 and 32

            recommendations.append({
                'parameter': 'batch_size',
                'value': batch_size,
                'reason': f'Estimated for 8GB VRAM with avg sequence length {avg_seq_len:.0f}'
            })

        # 3. Learning rate recommendation
        recommendations.append({
            'parameter': 'learning_rate',
            'value': 0.0001,
            'reason': 'Standard for BiLSTM-CTC training'
        })

        # 4. LSTM configuration
        vocab_size = self.results['loader_test'].get('vocab_size', 1000)
        recommendations.append({
            'parameter': 'lstm_hidden_dim',
            'value': 256,
            'reason': f'Balanced capacity for {vocab_size}-sign vocabulary'
        })

        recommendations.append({
            'parameter': 'lstm_num_layers',
            'value': 2,
            'reason': 'Standard for SLR with memory constraints'
        })

        # 5. Training epochs
        recommendations.append({
            'parameter': 'num_epochs',
            'value': 50,
            'reason': 'Typical for convergence with early stopping'
        })

        # 6. Gradient clipping
        recommendations.append({
            'parameter': 'gradient_clip_norm',
            'value': 5.0,
            'reason': 'Prevents exploding gradients in LSTM'
        })

        self.results['recommendations'] = recommendations

        # Print recommendations
        print("\n  Recommended Training Hyperparameters:")
        print()
        for rec in recommendations:
            print(f"    {rec['parameter']}: {rec['value']}")
            print(f"      Reason: {rec['reason']}")

    def _save_results(self):
        """Save validation results to JSON."""
        output_file = self.features_root / "validation_report.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print()
        print("="*80)
        print(f"Validation report saved to: {output_file}")


def generate_summary_report(results: Dict):
    """Generate executive summary."""
    print()
    print("="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    print()

    # Overall status
    completeness_ok = results['summary']['total_samples'] == results['summary']['expected_total']
    quality_ok = all(
        results['quality'][split]['total_issues'] == 0
        for split in ['train', 'dev', 'test']
        if split in results['quality']
    )
    loader_ok = results['loader_test']['status'] == 'PASS'

    overall_status = "PASS" if (completeness_ok and quality_ok and loader_ok) else "WARNING"

    print(f"Overall Status: {overall_status}")
    print()
    print(f"Dataset Completeness: {'PASS' if completeness_ok else 'FAIL'}")
    print(f"  - Total samples: {results['summary']['total_samples']}/{results['summary']['expected_total']}")
    print(f"  - Total frames: {results['summary']['total_frames']:,}")
    print(f"  - Storage: {results['summary']['total_storage_mb']:.1f} MB")
    print(f"  - Compression: {results['summary']['compression_ratio_pct']:.1f}% reduction")
    print()

    print(f"Feature Quality: {'PASS' if quality_ok else 'WARNING'}")
    for split in ['train', 'dev', 'test']:
        if split in results['quality']:
            issues = results['quality'][split]['total_issues']
            print(f"  - {split}: {issues} issues found")
    print()

    print(f"Dataset Loader: {'PASS' if loader_ok else 'FAIL'}")
    print(f"  - Vocabulary size: {results['loader_test'].get('vocab_size', 'N/A')}")
    print()

    print("Readiness for Training:")
    if overall_status == "PASS":
        print("  - Dataset is READY for BiLSTM training")
        print("  - All prerequisites met")
        print("  - See recommendations below for optimal hyperparameters")
    else:
        print("  - WARNING: Issues detected that should be addressed")
        if not completeness_ok:
            print("    - Some samples are missing features")
        if not quality_ok:
            print("    - Some feature quality issues detected")
        if not loader_ok:
            print("    - Dataset loader has errors")
    print()

    print("Recommended Next Steps:")
    if overall_status == "PASS":
        print("  1. Implement BiLSTM-CTC model architecture")
        print("  2. Set up training pipeline with recommended hyperparameters")
        print("  3. Implement WER/SER evaluation metrics")
        print("  4. Run baseline training experiment")
    else:
        print("  1. Investigate and resolve validation issues")
        print("  2. Re-run feature extraction for missing samples if needed")
        print("  3. Re-validate before proceeding to training")


if __name__ == "__main__":
    # Configuration
    DATA_ROOT = "data/raw_data/phoenix-2014-signerindependent-SI5"
    FEATURES_ROOT = "data/processed"

    # Run validation
    validator = FeatureValidator(DATA_ROOT, FEATURES_ROOT)
    results = validator.validate_all()

    # Generate summary
    generate_summary_report(results)
