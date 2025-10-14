"""
Dataset Explorer for RWTH-PHOENIX-Weather 2014 SI5
Analyzes dataset structure, statistics, and validates integrity.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class PhoenixDatasetExplorer:
    """Explore and validate RWTH-PHOENIX-Weather 2014 SI5 dataset."""

    def __init__(self, data_root="data/raw_data/phoenix-2014-signerindependent-SI5"):
        self.data_root = Path(data_root)
        self.features_root = self.data_root / "features" / "fullFrame-210x260px"
        self.annotations_root = self.data_root / "annotations" / "manual"

        # Load splits
        self.train_df = self._load_corpus("train.SI5.corpus.csv")
        self.dev_df = self._load_corpus("dev.SI5.corpus.csv")
        self.test_df = self._load_corpus("test.SI5.corpus.csv")

    def _load_corpus(self, filename):
        """Load corpus CSV file."""
        filepath = self.annotations_root / filename
        return pd.read_csv(filepath, delimiter="|")

    def get_basic_statistics(self):
        """Get basic dataset statistics."""
        stats = {
            "train_samples": len(self.train_df),
            "dev_samples": len(self.dev_df),
            "test_samples": len(self.test_df),
            "total_samples": len(self.train_df) + len(self.dev_df) + len(self.test_df),
        }

        # Get unique signers
        stats["train_signers"] = self.train_df["signer"].unique().tolist()
        stats["dev_signers"] = self.dev_df["signer"].unique().tolist()
        stats["test_signers"] = self.test_df["signer"].unique().tolist()

        return stats

    def analyze_vocabulary(self):
        """Analyze vocabulary and sign frequency."""
        def extract_signs(df):
            """Extract all signs from annotations."""
            signs = []
            for annotation in df["annotation"]:
                # Split by space and filter out special tokens
                tokens = annotation.split()
                signs.extend(tokens)
            return signs

        train_signs = extract_signs(self.train_df)
        dev_signs = extract_signs(self.dev_df)
        test_signs = extract_signs(self.test_df)

        # Count frequencies
        train_counter = Counter(train_signs)
        dev_counter = Counter(dev_signs)
        test_counter = Counter(test_signs)

        vocab = {
            "total_vocabulary": len(set(train_signs + dev_signs + test_signs)),
            "train_vocabulary": len(set(train_signs)),
            "dev_vocabulary": len(set(dev_signs)),
            "test_vocabulary": len(set(test_signs)),
            "train_total_signs": len(train_signs),
            "dev_total_signs": len(dev_signs),
            "test_total_signs": len(test_signs),
            "train_top_20": train_counter.most_common(20),
            "train_counter": train_counter,
            "dev_counter": dev_counter,
            "test_counter": test_counter,
        }

        return vocab

    def analyze_sequence_lengths(self, split="train", sample_size=None):
        """Analyze sequence lengths (number of frames per sample)."""
        df = getattr(self, f"{split}_df")

        if sample_size:
            df = df.sample(min(sample_size, len(df)), random_state=42)

        sequence_lengths = []
        annotation_lengths = []

        print(f"\nAnalyzing {len(df)} samples from {split} split...")

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            folder_path = self.features_root / split / row["folder"].replace("/*.png", "")

            if folder_path.exists():
                num_frames = len(list(folder_path.glob("*.png")))
                sequence_lengths.append(num_frames)
            else:
                sequence_lengths.append(0)  # Missing data

            # Count annotation tokens (signs)
            annotation_tokens = row["annotation"].split()
            annotation_lengths.append(len(annotation_tokens))

        return {
            "sequence_lengths": sequence_lengths,
            "annotation_lengths": annotation_lengths,
            "mean_frames": np.mean([s for s in sequence_lengths if s > 0]),
            "median_frames": np.median([s for s in sequence_lengths if s > 0]),
            "max_frames": max(sequence_lengths),
            "min_frames": min([s for s in sequence_lengths if s > 0]),
            "mean_signs": np.mean(annotation_lengths),
            "median_signs": np.median(annotation_lengths),
            "max_signs": max(annotation_lengths),
            "min_signs": min(annotation_lengths),
            "missing_samples": sum(1 for s in sequence_lengths if s == 0),
        }

    def verify_dataset_integrity(self, split="train", sample_size=100):
        """Verify that video frames exist and are readable."""
        df = getattr(self, f"{split}_df")

        if sample_size:
            df = df.sample(min(sample_size, len(df)), random_state=42)

        issues = []

        print(f"\nVerifying integrity of {len(df)} samples from {split} split...")

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            sample_id = row["id"]
            folder_path = self.features_root / split / row["folder"].replace("/*.png", "")

            if not folder_path.exists():
                issues.append({
                    "sample_id": sample_id,
                    "issue": "folder_not_found",
                    "path": str(folder_path)
                })
                continue

            # Check if frames exist
            frames = list(folder_path.glob("*.png"))
            if len(frames) == 0:
                issues.append({
                    "sample_id": sample_id,
                    "issue": "no_frames",
                    "path": str(folder_path)
                })

        return {
            "total_checked": len(df),
            "issues_found": len(issues),
            "issues": issues,
            "integrity_rate": (len(df) - len(issues)) / len(df) * 100
        }

    def print_summary(self):
        """Print comprehensive dataset summary."""
        print("=" * 80)
        print("RWTH-PHOENIX-Weather 2014 SI5 Dataset Summary")
        print("=" * 80)

        # Basic statistics
        stats = self.get_basic_statistics()
        print("\n--- Split Statistics ---")
        print(f"Train samples: {stats['train_samples']}")
        print(f"Dev samples: {stats['dev_samples']}")
        print(f"Test samples: {stats['test_samples']}")
        print(f"Total samples: {stats['total_samples']}")

        print("\n--- Signer Information ---")
        print(f"Train signers: {stats['train_signers']}")
        print(f"Dev signers: {stats['dev_signers']}")
        print(f"Test signers: {stats['test_signers']}")

        # Vocabulary analysis
        print("\n--- Vocabulary Analysis ---")
        vocab = self.analyze_vocabulary()
        print(f"Total vocabulary size: {vocab['total_vocabulary']} unique signs")
        print(f"Train vocabulary: {vocab['train_vocabulary']}")
        print(f"Dev vocabulary: {vocab['dev_vocabulary']}")
        print(f"Test vocabulary: {vocab['test_vocabulary']}")
        print(f"\nTotal sign instances:")
        print(f"  Train: {vocab['train_total_signs']}")
        print(f"  Dev: {vocab['dev_total_signs']}")
        print(f"  Test: {vocab['test_total_signs']}")

        print(f"\nTop 20 most frequent signs in training set:")
        for sign, count in vocab['train_top_20']:
            print(f"  {sign}: {count}")

        return stats, vocab


def main():
    """Main exploration function."""
    explorer = PhoenixDatasetExplorer()

    # Print summary
    stats, vocab = explorer.print_summary()

    # Verify integrity on small sample
    print("\n--- Dataset Integrity Check (100 samples) ---")
    integrity = explorer.verify_dataset_integrity(split="train", sample_size=100)
    print(f"Checked: {integrity['total_checked']} samples")
    print(f"Issues found: {integrity['issues_found']}")
    print(f"Integrity rate: {integrity['integrity_rate']:.2f}%")

    if integrity['issues_found'] > 0:
        print("\nSample issues:")
        for issue in integrity['issues'][:5]:
            print(f"  - {issue['sample_id']}: {issue['issue']}")

    # Analyze sequence lengths on sample
    print("\n--- Sequence Length Analysis (100 samples) ---")
    seq_stats = explorer.analyze_sequence_lengths(split="train", sample_size=100)
    print(f"Mean frames per sequence: {seq_stats['mean_frames']:.1f}")
    print(f"Median frames: {seq_stats['median_frames']:.1f}")
    print(f"Min-Max frames: {seq_stats['min_frames']}-{seq_stats['max_frames']}")
    print(f"\nMean signs per annotation: {seq_stats['mean_signs']:.1f}")
    print(f"Median signs: {seq_stats['median_signs']:.1f}")
    print(f"Min-Max signs: {seq_stats['min_signs']}-{seq_stats['max_signs']}")
    print(f"Missing samples: {seq_stats['missing_samples']}")

    print("\n" + "=" * 80)
    print("Exploration complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
