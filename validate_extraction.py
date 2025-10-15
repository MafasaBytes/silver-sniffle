"""
Comprehensive Validation Script for Feature Extraction Results
Validates data quality, calculates statistics, and prepares for training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

class ExtractionValidator:
    """Validates extracted features and prepares dataset statistics."""

    def __init__(self,
                 data_root: str = "data/raw_data/phoenix-2014-signerindependent-SI5",
                 features_root: str = "data/processed"):
        self.data_root = Path(data_root)
        self.features_root = Path(features_root)
        self.stats = {}

    def validate_all(self):
        """Run all validation checks."""
        print("=" * 80)
        print("FEATURE EXTRACTION VALIDATION REPORT")
        print("=" * 80)

        # 1. File count validation
        print("\n[1/7] Validating file counts...")
        self.validate_file_counts()

        # 2. Data quality checks
        print("\n[2/7] Performing data quality checks...")
        self.validate_data_quality()

        # 3. Sequence length analysis
        print("\n[3/7] Analyzing sequence length distributions...")
        self.analyze_sequence_lengths()

        # 4. Vocabulary validation
        print("\n[4/7] Validating vocabulary...")
        self.validate_vocabulary()

        # 5. Data leakage check
        print("\n[5/7] Checking for data leakage...")
        self.check_data_leakage()

        # 6. Storage calculation
        print("\n[6/7] Calculating storage requirements...")
        self.calculate_storage()

        # 7. Generate summary report
        print("\n[7/7] Generating summary report...")
        self.generate_report()

    def validate_file_counts(self):
        """Validate extracted file counts against corpus annotations."""
        print("\n" + "-" * 80)
        print("FILE COUNT VALIDATION")
        print("-" * 80)

        results = {}
        for split in ['train', 'dev', 'test']:
            # Count extracted features
            split_dir = self.features_root / split
            feature_files = list(split_dir.glob('*.npy'))
            feature_count = len(feature_files)

            # Count expected from corpus
            corpus_file = self.data_root / "annotations" / "manual" / f"{split}.SI5.corpus.csv"
            if corpus_file.exists():
                corpus_df = pd.read_csv(corpus_file, delimiter="|")
                expected_count = len(corpus_df)
            else:
                expected_count = 0

            # Calculate frames
            frame_count = 0
            for f in feature_files:
                features = np.load(f)
                frame_count += len(features)

            results[split] = {
                'extracted': feature_count,
                'expected': expected_count,
                'frames': frame_count,
                'match': feature_count == expected_count
            }

            print(f"\n{split.upper()}:")
            print(f"  Extracted: {feature_count:,} sequences")
            print(f"  Expected:  {expected_count:,} sequences")
            print(f"  Frames:    {frame_count:,} frames")
            print(f"  Status:    {'OK' if results[split]['match'] else 'MISMATCH'}")

            if not results[split]['match']:
                discrepancy = feature_count - expected_count
                print(f"  WARNING: {abs(discrepancy)} sequence discrepancy!")

        # Summary
        total_extracted = sum(r['extracted'] for r in results.values())
        total_expected = sum(r['expected'] for r in results.values())
        total_frames = sum(r['frames'] for r in results.values())

        print(f"\nTOTAL:")
        print(f"  Extracted: {total_extracted:,} sequences")
        print(f"  Expected:  {total_expected:,} sequences")
        print(f"  Frames:    {total_frames:,} frames")
        print(f"  Success Rate: {100 * total_extracted / total_expected:.2f}%")

        self.stats['file_counts'] = results
        self.stats['totals'] = {
            'extracted': total_extracted,
            'expected': total_expected,
            'frames': total_frames
        }

    def validate_data_quality(self):
        """Check for NaN values, all-zeros, and validate dimensions."""
        print("\n" + "-" * 80)
        print("DATA QUALITY VALIDATION")
        print("-" * 80)

        quality_results = {}

        for split in ['train', 'dev', 'test']:
            split_dir = self.features_root / split
            files = list(split_dir.glob('*.npy'))

            # Sample files for quality check (all if < 100, else random 100)
            if len(files) > 100:
                sample_files = np.random.choice(files, 100, replace=False)
                print(f"\n{split.upper()} (sampling 100/{len(files)} files):")
            else:
                sample_files = files
                print(f"\n{split.upper()} (checking all {len(files)} files):")

            shapes = []
            has_nan = 0
            all_zero = 0
            value_mins = []
            value_maxs = []

            for f in sample_files:
                features = np.load(f)
                shapes.append(features.shape)

                # Check for NaN
                if np.isnan(features).any():
                    has_nan += 1
                    print(f"  WARNING: NaN found in {f.name}")

                # Check for all-zero sequences
                if np.all(features == 0):
                    all_zero += 1
                    print(f"  WARNING: All-zero sequence in {f.name}")

                # Value ranges
                value_mins.append(features.min())
                value_maxs.append(features.max())

            # Validate dimensions
            feature_dims = set([s[1] for s in shapes])
            expected_dim = 177  # YOLOv8 (51) + MediaPipe Hands (126)

            print(f"  Feature dimensions: {feature_dims}")
            print(f"  Expected: (variable, {expected_dim})")

            if len(feature_dims) == 1 and expected_dim in feature_dims:
                print(f"  Dimension check: PASS")
            else:
                print(f"  Dimension check: FAIL - Unexpected dimensions!")

            print(f"  Sequences with NaN: {has_nan}/{len(sample_files)}")
            print(f"  All-zero sequences: {all_zero}/{len(sample_files)}")
            print(f"  Value range: [{min(value_mins):.3f}, {max(value_maxs):.3f}]")

            quality_results[split] = {
                'dimensions': list(feature_dims),
                'nan_count': has_nan,
                'zero_count': all_zero,
                'value_range': (float(min(value_mins)), float(max(value_maxs))),
                'samples_checked': len(sample_files)
            }

        self.stats['quality'] = quality_results

    def analyze_sequence_lengths(self):
        """Analyze sequence length distributions across splits."""
        print("\n" + "-" * 80)
        print("SEQUENCE LENGTH ANALYSIS")
        print("-" * 80)

        length_stats = {}
        all_lengths = []

        for split in ['train', 'dev', 'test']:
            split_dir = self.features_root / split
            files = list(split_dir.glob('*.npy'))

            lengths = []
            for f in files:
                features = np.load(f)
                lengths.append(len(features))

            all_lengths.extend(lengths)

            # Calculate statistics
            lengths_array = np.array(lengths)
            stats = {
                'mean': float(np.mean(lengths_array)),
                'std': float(np.std(lengths_array)),
                'median': float(np.median(lengths_array)),
                'min': int(np.min(lengths_array)),
                'max': int(np.max(lengths_array)),
                'q25': float(np.percentile(lengths_array, 25)),
                'q75': float(np.percentile(lengths_array, 75)),
                'q95': float(np.percentile(lengths_array, 95)),
                'q99': float(np.percentile(lengths_array, 99)),
            }

            print(f"\n{split.upper()}:")
            print(f"  Mean:   {stats['mean']:.1f} frames")
            print(f"  Std:    {stats['std']:.1f} frames")
            print(f"  Median: {stats['median']:.0f} frames")
            print(f"  Min:    {stats['min']} frames")
            print(f"  Max:    {stats['max']} frames")
            print(f"  95th percentile: {stats['q95']:.0f} frames")
            print(f"  99th percentile: {stats['q99']:.0f} frames")

            length_stats[split] = stats

        # Overall statistics
        all_lengths_array = np.array(all_lengths)
        overall_stats = {
            'mean': float(np.mean(all_lengths_array)),
            'std': float(np.std(all_lengths_array)),
            'median': float(np.median(all_lengths_array)),
            'min': int(np.min(all_lengths_array)),
            'max': int(np.max(all_lengths_array)),
            'q95': float(np.percentile(all_lengths_array, 95)),
            'q99': float(np.percentile(all_lengths_array, 99)),
        }

        print(f"\nOVERALL:")
        print(f"  Mean:   {overall_stats['mean']:.1f} frames")
        print(f"  Std:    {overall_stats['std']:.1f} frames")
        print(f"  Median: {overall_stats['median']:.0f} frames")
        print(f"  Range:  [{overall_stats['min']}, {overall_stats['max']}] frames")

        # Truncation recommendations
        print(f"\nTRUNCATION RECOMMENDATIONS:")
        for max_len in [150, 200, 250, 300]:
            pct_kept = 100 * np.mean(all_lengths_array <= max_len)
            num_kept = int(np.sum(all_lengths_array <= max_len))
            print(f"  {max_len} frames: keeps {pct_kept:.1f}% ({num_kept:,}/{len(all_lengths_array):,} sequences)")

        self.stats['sequence_lengths'] = {
            'by_split': length_stats,
            'overall': overall_stats,
            'all_lengths': all_lengths  # For plotting
        }

    def validate_vocabulary(self):
        """Validate vocabulary file and sign distribution."""
        print("\n" + "-" * 80)
        print("VOCABULARY VALIDATION")
        print("-" * 80)

        vocab_file = self.features_root / "train" / "vocabulary.txt"

        if not vocab_file.exists():
            print("  ERROR: Vocabulary file not found!")
            self.stats['vocabulary'] = {'status': 'missing'}
            return

        # Load vocabulary
        vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    sign, idx = parts
                    vocab[sign] = int(idx)

        # Count special tokens
        special_tokens = ['<PAD>', '<BLANK>', '<UNK>']
        special_count = sum(1 for s in special_tokens if s in vocab)
        sign_count = len(vocab) - special_count

        print(f"  Total vocabulary size: {len(vocab)}")
        print(f"  Special tokens: {special_count} {special_tokens}")
        print(f"  Sign vocabulary: {sign_count}")
        print(f"  Expected sign vocabulary: ~1,066")

        # Analyze sign frequency in corpus
        corpus_file = self.data_root / "annotations" / "manual" / "train.SI5.corpus.csv"
        if corpus_file.exists():
            corpus_df = pd.read_csv(corpus_file, delimiter="|")

            sign_counts = defaultdict(int)
            total_signs = 0

            for annotation in corpus_df['annotation']:
                signs = annotation.split()
                for sign in signs:
                    sign_counts[sign] += 1
                    total_signs += 1

            print(f"\n  Corpus statistics:")
            print(f"    Total sign tokens: {total_signs:,}")
            print(f"    Unique signs: {len(sign_counts)}")
            print(f"    Average signs per sequence: {total_signs / len(corpus_df):.1f}")

            # Most common signs
            sorted_signs = sorted(sign_counts.items(), key=lambda x: x[1], reverse=True)
            print(f"\n  Top 10 most frequent signs:")
            for i, (sign, count) in enumerate(sorted_signs[:10], 1):
                pct = 100 * count / total_signs
                print(f"    {i:2d}. {sign:20s} {count:5d} ({pct:5.2f}%)")

            self.stats['vocabulary'] = {
                'status': 'ok',
                'total_size': len(vocab),
                'sign_count': sign_count,
                'special_tokens': special_count,
                'corpus_sign_tokens': total_signs,
                'corpus_unique_signs': len(sign_counts),
                'top_signs': sorted_signs[:20]
            }
        else:
            self.stats['vocabulary'] = {
                'status': 'ok',
                'total_size': len(vocab),
                'sign_count': sign_count,
                'special_tokens': special_count
            }

    def check_data_leakage(self):
        """Check for data leakage across train/dev/test splits."""
        print("\n" + "-" * 80)
        print("DATA LEAKAGE CHECK")
        print("-" * 80)

        # Get sequence IDs from each split
        train_ids = set([f.stem for f in (self.features_root / 'train').glob('*.npy')])
        dev_ids = set([f.stem for f in (self.features_root / 'dev').glob('*.npy')])
        test_ids = set([f.stem for f in (self.features_root / 'test').glob('*.npy')])

        # Check for overlaps
        train_dev_overlap = train_ids & dev_ids
        train_test_overlap = train_ids & test_ids
        dev_test_overlap = dev_ids & test_ids

        print(f"\n  Train sequences: {len(train_ids):,}")
        print(f"  Dev sequences:   {len(dev_ids):,}")
        print(f"  Test sequences:  {len(test_ids):,}")

        print(f"\n  Train/Dev overlap:  {len(train_dev_overlap)} sequences")
        print(f"  Train/Test overlap: {len(train_test_overlap)} sequences")
        print(f"  Dev/Test overlap:   {len(dev_test_overlap)} sequences")

        if len(train_dev_overlap) == 0 and len(train_test_overlap) == 0 and len(dev_test_overlap) == 0:
            print(f"\n  STATUS: PASS - No data leakage detected")
            leakage_status = 'pass'
        else:
            print(f"\n  STATUS: FAIL - Data leakage detected!")
            leakage_status = 'fail'
            if train_dev_overlap:
                print(f"    Train/Dev overlap IDs: {list(train_dev_overlap)[:10]}")
            if train_test_overlap:
                print(f"    Train/Test overlap IDs: {list(train_test_overlap)[:10]}")
            if dev_test_overlap:
                print(f"    Dev/Test overlap IDs: {list(dev_test_overlap)[:10]}")

        self.stats['data_leakage'] = {
            'status': leakage_status,
            'train_dev_overlap': len(train_dev_overlap),
            'train_test_overlap': len(train_test_overlap),
            'dev_test_overlap': len(dev_test_overlap)
        }

    def calculate_storage(self):
        """Calculate storage requirements."""
        print("\n" + "-" * 80)
        print("STORAGE REQUIREMENTS")
        print("-" * 80)

        storage_stats = {}

        for split in ['train', 'dev', 'test']:
            split_dir = self.features_root / split
            files = list(split_dir.glob('*.npy'))

            total_bytes = sum(f.stat().st_size for f in files)
            total_mb = total_bytes / (1024 ** 2)

            # Calculate from dimensions
            num_frames = self.stats['file_counts'][split]['frames']
            bytes_per_frame = 177 * 4  # float32
            calculated_mb = (num_frames * bytes_per_frame) / (1024 ** 2)

            print(f"\n{split.upper()}:")
            print(f"  Files: {len(files):,}")
            print(f"  Frames: {num_frames:,}")
            print(f"  Actual size: {total_mb:.1f} MB")
            print(f"  Calculated size: {calculated_mb:.1f} MB")
            print(f"  Bytes per frame: {bytes_per_frame} bytes")

            storage_stats[split] = {
                'files': len(files),
                'frames': num_frames,
                'size_mb': total_mb,
                'calculated_mb': calculated_mb
            }

        total_mb = sum(s['size_mb'] for s in storage_stats.values())
        total_calculated_mb = sum(s['calculated_mb'] for s in storage_stats.values())

        print(f"\nTOTAL:")
        print(f"  Actual size: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")
        print(f"  Calculated size: {total_calculated_mb:.1f} MB ({total_calculated_mb/1024:.2f} GB)")

        self.stats['storage'] = storage_stats

    def generate_report(self):
        """Generate summary JSON report."""
        print("\n" + "-" * 80)
        print("GENERATING SUMMARY REPORT")
        print("-" * 80)

        # Save detailed stats (excluding large arrays)
        report_stats = self.stats.copy()
        if 'sequence_lengths' in report_stats:
            # Remove the large all_lengths array from JSON output
            if 'all_lengths' in report_stats['sequence_lengths']:
                del report_stats['sequence_lengths']['all_lengths']

        report_file = Path('validation_report.json')
        with open(report_file, 'w') as f:
            json.dump(report_stats, f, indent=2)

        print(f"\n  Detailed report saved to: {report_file.absolute()}")

        # Generate plots
        self.generate_plots()

    def generate_plots(self):
        """Generate visualization plots."""
        print("\n  Generating visualization plots...")

        if 'sequence_lengths' not in self.stats:
            print("    No sequence length data available for plotting")
            return

        all_lengths = self.stats['sequence_lengths'].get('all_lengths', [])
        if not all_lengths:
            print("    No sequence length data available for plotting")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Feature Extraction Validation Report', fontsize=16, fontweight='bold')

        # 1. Sequence length histogram
        ax = axes[0, 0]
        ax.hist(all_lengths, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(all_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(all_lengths):.1f}')
        ax.axvline(np.median(all_lengths), color='green', linestyle='--', label=f'Median: {np.median(all_lengths):.1f}')
        ax.set_xlabel('Sequence Length (frames)')
        ax.set_ylabel('Frequency')
        ax.set_title('Sequence Length Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Cumulative distribution
        ax = axes[0, 1]
        sorted_lengths = np.sort(all_lengths)
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
        ax.plot(sorted_lengths, cumulative, linewidth=2)
        for threshold in [150, 200, 250, 300]:
            pct = 100 * np.mean(np.array(all_lengths) <= threshold)
            ax.axhline(pct, color='red', linestyle=':', alpha=0.5)
            ax.axvline(threshold, color='red', linestyle=':', alpha=0.5)
            ax.text(threshold + 5, pct - 3, f'{threshold}f: {pct:.1f}%', fontsize=8)
        ax.set_xlabel('Sequence Length (frames)')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title('Cumulative Distribution (Truncation Analysis)')
        ax.grid(True, alpha=0.3)

        # 3. Split statistics comparison
        ax = axes[1, 0]
        splits = ['train', 'dev', 'test']
        if 'by_split' in self.stats['sequence_lengths']:
            means = [self.stats['sequence_lengths']['by_split'][s]['mean'] for s in splits]
            stds = [self.stats['sequence_lengths']['by_split'][s]['std'] for s in splits]
            x = np.arange(len(splits))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels(splits)
            ax.set_ylabel('Mean Sequence Length (frames)')
            ax.set_title('Sequence Length by Split (Mean ± Std)')
            ax.grid(True, alpha=0.3, axis='y')

        # 4. File count summary
        ax = axes[1, 1]
        if 'file_counts' in self.stats:
            splits_data = []
            for s in splits:
                if s in self.stats['file_counts']:
                    splits_data.append({
                        'split': s,
                        'extracted': self.stats['file_counts'][s]['extracted'],
                        'expected': self.stats['file_counts'][s]['expected']
                    })

            x = np.arange(len(splits_data))
            width = 0.35
            extracted = [d['extracted'] for d in splits_data]
            expected = [d['expected'] for d in splits_data]

            ax.bar(x - width/2, extracted, width, label='Extracted', alpha=0.7, edgecolor='black')
            ax.bar(x + width/2, expected, width, label='Expected', alpha=0.7, edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels([d['split'] for d in splits_data])
            ax.set_ylabel('Number of Sequences')
            ax.set_title('Extracted vs Expected Sequences by Split')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_file = Path('validation_plots.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"    Plots saved to: {plot_file.absolute()}")
        plt.close()


def main():
    """Run comprehensive validation."""
    validator = ExtractionValidator()
    validator.validate_all()

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review validation_report.json for detailed statistics")
    print("  2. Examine validation_plots.png for visualizations")
    print("  3. Fix dimension mismatch in src/phoenix_dataset.py (1662 -> 177)")
    print("  4. Test dataset loader with corrected dimensions")
    print("  5. Proceed to BiLSTM model training")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
