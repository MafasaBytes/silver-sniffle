"""
PyTorch Dataset for Pre-extracted RWTH-PHOENIX-Weather 2014 Features
Loads MediaPipe Holistic features from .npy files.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class PhoenixFeatureDataset(Dataset):
    """
    Dataset for pre-extracted MediaPipe features from RWTH-PHOENIX-Weather 2014.

    Features are loaded from .npy files with shape (num_frames, 1662).
    """

    def __init__(
        self,
        data_root: str = "data/raw_data/phoenix-2014-signerindependent-SI5",
        features_root: str = "data/processed",
        split: str = "train",
        max_sequence_length: Optional[int] = None,
    ):
        """
        Initialize Phoenix dataset.

        Args:
            data_root: Root directory of raw dataset (for annotations)
            features_root: Root directory of pre-extracted features
            split: 'train', 'dev', or 'test'
            max_sequence_length: Maximum sequence length (truncate if longer)
        """
        self.data_root = Path(data_root)
        self.features_root = Path(features_root) / split
        self.split = split
        self.max_sequence_length = max_sequence_length

        # Load annotations
        annotations_file = self.data_root / "annotations" / "manual" / f"{split}.SI5.corpus.csv"
        self.annotations_df = pd.read_csv(annotations_file, delimiter="|")

        # Build vocabulary from training data
        if split == "train":
            self.vocab = self._build_vocabulary()
        else:
            # Load vocabulary from training set
            vocab_file = Path(features_root) / "train" / "vocabulary.txt"
            if vocab_file.exists():
                self.vocab = self._load_vocabulary(vocab_file)
            else:
                # If vocabulary doesn't exist yet, build it
                print(f"Warning: Vocabulary file not found. Building from {split} split.")
                self.vocab = self._build_vocabulary()

        # Filter samples where features exist
        self.samples = []
        for idx, row in self.annotations_df.iterrows():
            feature_file = self.features_root / f"{row['id']}.npy"
            if feature_file.exists():
                self.samples.append({
                    'id': row['id'],
                    'signer': row['signer'],
                    'annotation': row['annotation'],
                    'feature_file': feature_file,
                })

        print(f"Loaded {len(self.samples)}/{len(self.annotations_df)} samples for {split} split")

        if len(self.samples) < len(self.annotations_df):
            print(f"Warning: {len(self.annotations_df) - len(self.samples)} samples have missing features")

    def _build_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary from annotations."""
        all_signs = set()
        for annotation in self.annotations_df['annotation']:
            signs = annotation.split()
            all_signs.update(signs)

        # Create vocabulary with special tokens
        vocab = {
            '<PAD>': 0,   # Padding token
            '<BLANK>': 1, # CTC blank token
            '<UNK>': 2,   # Unknown token
        }

        # Add all signs from dataset
        for sign in sorted(all_signs):
            if sign not in vocab:
                vocab[sign] = len(vocab)

        # Save vocabulary if training split
        if self.split == "train":
            vocab_file = self.features_root / "vocabulary.txt"
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for sign, idx in sorted(vocab.items(), key=lambda x: x[1]):
                    f.write(f"{sign}\t{idx}\n")

        return vocab

    def _load_vocabulary(self, vocab_file: Path) -> Dict[str, int]:
        """Load vocabulary from file."""
        vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                sign, idx = line.strip().split('\t')
                vocab[sign] = int(idx)
        return vocab

    def encode_annotation(self, annotation: str) -> List[int]:
        """Encode annotation string to list of indices."""
        signs = annotation.split()
        encoded = []
        for sign in signs:
            if sign in self.vocab:
                encoded.append(self.vocab[sign])
            else:
                encoded.append(self.vocab['<UNK>'])
        return encoded

    def decode_annotation(self, indices: List[int]) -> str:
        """Decode list of indices back to annotation string."""
        # Reverse vocabulary
        idx_to_sign = {idx: sign for sign, idx in self.vocab.items()}
        signs = []
        for idx in indices:
            if idx in idx_to_sign:
                sign = idx_to_sign[idx]
                # Skip special tokens
                if sign not in ['<PAD>', '<BLANK>', '<UNK>']:
                    signs.append(sign)
        return ' '.join(signs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - features: Tensor of shape (seq_len, 1662)
                - target: Tensor of shape (target_len,) with encoded annotation
                - feature_length: Scalar tensor with sequence length
                - target_length: Scalar tensor with target length
                - sample_id: String identifier
        """
        sample = self.samples[idx]

        # Load features
        features = np.load(sample['feature_file'])  # Shape: (seq_len, 1662)

        # Truncate if needed
        if self.max_sequence_length and features.shape[0] > self.max_sequence_length:
            features = features[:self.max_sequence_length]

        # Encode annotation
        target = self.encode_annotation(sample['annotation'])

        # Convert to tensors
        features_tensor = torch.from_numpy(features).float()
        target_tensor = torch.tensor(target, dtype=torch.long)
        feature_length = torch.tensor(features.shape[0], dtype=torch.long)
        target_length = torch.tensor(len(target), dtype=torch.long)

        return {
            'features': features_tensor,
            'target': target_tensor,
            'feature_length': feature_length,
            'target_length': target_length,
            'sample_id': sample['id'],
            'signer': sample['signer'],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader to handle variable-length sequences.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Dictionary with batched and padded tensors
    """
    # Sort batch by feature length (descending) for efficient packing
    batch = sorted(batch, key=lambda x: x['feature_length'], reverse=True)

    # Pad features
    features = [sample['features'] for sample in batch]
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)

    # Pad targets
    targets = [sample['target'] for sample in batch]
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)

    # Stack lengths
    feature_lengths = torch.stack([sample['feature_length'] for sample in batch])
    target_lengths = torch.stack([sample['target_length'] for sample in batch])

    # Collect metadata
    sample_ids = [sample['sample_id'] for sample in batch]
    signers = [sample['signer'] for sample in batch]

    return {
        'features': features_padded,          # (batch, max_seq_len, 1662)
        'targets': targets_padded,            # (batch, max_target_len)
        'feature_lengths': feature_lengths,   # (batch,)
        'target_lengths': target_lengths,     # (batch,)
        'sample_ids': sample_ids,
        'signers': signers,
    }


def create_dataloaders(
    data_root: str = "data/raw_data/phoenix-2014-signerindependent-SI5",
    features_root: str = "data/processed",
    batch_size: int = 4,
    num_workers: int = 0,
    max_sequence_length: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, dev, and test splits.

    Args:
        data_root: Root directory of raw dataset
        features_root: Root directory of pre-extracted features
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        max_sequence_length: Maximum sequence length

    Returns:
        Tuple of (train_loader, dev_loader, test_loader)
    """
    # Create datasets
    train_dataset = PhoenixFeatureDataset(
        data_root=data_root,
        features_root=features_root,
        split="train",
        max_sequence_length=max_sequence_length,
    )

    dev_dataset = PhoenixFeatureDataset(
        data_root=data_root,
        features_root=features_root,
        split="dev",
        max_sequence_length=max_sequence_length,
    )

    test_dataset = PhoenixFeatureDataset(
        data_root=data_root,
        features_root=features_root,
        split="test",
        max_sequence_length=max_sequence_length,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, dev_loader, test_loader


def test_dataset():
    """Test dataset loading functionality."""
    print("Testing Phoenix Feature Dataset...")
    print("="*60)

    # Create dataset
    dataset = PhoenixFeatureDataset(
        data_root="data/raw_data/phoenix-2014-signerindependent-SI5",
        features_root="data/processed",
        split="train",
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Vocabulary size: {len(dataset.vocab)}")

    # Test single sample
    print("\n--- Testing Single Sample ---")
    sample = dataset[0]
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Signer: {sample['signer']}")
    print(f"Features shape: {sample['features'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    print(f"Feature length: {sample['feature_length']}")
    print(f"Target length: {sample['target_length']}")

    # Decode annotation
    decoded = dataset.decode_annotation(sample['target'].tolist())
    print(f"Decoded annotation: {decoded}")

    # Test dataloader
    print("\n--- Testing DataLoader ---")
    train_loader, dev_loader, test_loader = create_dataloaders(
        batch_size=4,
        num_workers=0,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test single batch
    batch = next(iter(train_loader))
    print(f"\nBatch features shape: {batch['features'].shape}")
    print(f"Batch targets shape: {batch['targets'].shape}")
    print(f"Batch feature lengths: {batch['feature_lengths']}")
    print(f"Batch target lengths: {batch['target_lengths']}")

    print("\n" + "="*60)
    print("Dataset test complete!")


if __name__ == "__main__":
    test_dataset()
