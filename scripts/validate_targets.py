"""
Diagnostic script to validate target sequences.
Checks for token ID validity, padding contamination, and vocabulary consistency.
"""

import torch
import sys
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from phoenix_dataset import create_dataloaders


def validate_targets(dataloader, split_name='train', vocab=None, num_batches=50):
    """Validate target sequences for correctness."""
    print(f"\n{'='*60}")
    print(f"Target Validation: {split_name.upper()} set")
    print(f"{'='*60}\n")

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Expected token range: [3, {vocab_size-1}] (excluding PAD=0, BLANK=1, UNK=2)")

    # Statistics
    total_sequences = 0
    total_tokens = 0
    token_counter = Counter()
    invalid_sequences = []

    pad_contamination = 0
    blank_contamination = 0
    unk_contamination = 0
    out_of_range = 0

    print("\nProcessing batches...")
    batch_count = 0

    for batch_idx, batch in enumerate(dataloader):
        targets = batch['targets']  # (B, S)
        target_lengths = batch['target_lengths']  # (B,)
        sample_ids = batch['sample_ids']

        batch_size = targets.size(0)
        total_sequences += batch_size

        for i in range(batch_size):
            seq_len = target_lengths[i].item()
            target_seq = targets[i, :seq_len]  # Extract actual target (no padding)

            # Count tokens
            total_tokens += seq_len
            token_counter.update(target_seq.tolist())

            # Check for invalid tokens
            issues = []

            # Check for PAD (0)
            if (target_seq == 0).any():
                pad_contamination += 1
                issues.append("PAD=0")

            # Check for BLANK (1)
            if (target_seq == 1).any():
                blank_contamination += 1
                issues.append("BLANK=1")

            # Check for UNK (2)
            if (target_seq == 2).any():
                unk_contamination += 1
                issues.append("UNK=2")

            # Check for out-of-range tokens
            if (target_seq >= vocab_size).any():
                out_of_range += 1
                issues.append("OUT_OF_RANGE")

            # Check for negative tokens
            if (target_seq < 0).any():
                issues.append("NEGATIVE")

            if issues:
                invalid_sequences.append({
                    'sample_id': sample_ids[i],
                    'target_seq': target_seq.tolist(),
                    'issues': issues
                })

        batch_count += 1
        if batch_count >= num_batches:
            break

    # Report statistics
    print(f"\n{'-'*60}")
    print("Validation Statistics")
    print(f"{'-'*60}")
    print(f"Total sequences: {total_sequences:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average sequence length: {total_tokens/total_sequences:.1f}")
    print(f"Unique tokens found: {len(token_counter)}")

    # Check for contamination
    print(f"\n{'-'*60}")
    print("Token Contamination Check")
    print(f"{'-'*60}")
    print(f"Sequences with PAD (0): {pad_contamination} ({pad_contamination/total_sequences*100:.2f}%)")
    print(f"Sequences with BLANK (1): {blank_contamination} ({blank_contamination/total_sequences*100:.2f}%)")
    print(f"Sequences with UNK (2): {unk_contamination} ({unk_contamination/total_sequences*100:.2f}%)")
    print(f"Sequences out of range: {out_of_range} ({out_of_range/total_sequences*100:.2f}%)")

    # Token ID distribution
    print(f"\n{'-'*60}")
    print("Token ID Distribution")
    print(f"{'-'*60}")
    print("Most common tokens:")
    for token_id, count in token_counter.most_common(10):
        percentage = count / total_tokens * 100
        print(f"  Token {token_id}: {count:>6,} ({percentage:>5.2f}%)")

    print("\nLeast common tokens:")
    for token_id, count in token_counter.most_common()[-10:]:
        percentage = count / total_tokens * 100
        print(f"  Token {token_id}: {count:>6,} ({percentage:>5.2f}%)")

    # Token ID range check
    min_token = min(token_counter.keys())
    max_token = max(token_counter.keys())
    print(f"\nToken ID range: [{min_token}, {max_token}]")

    if min_token < 3:
        print(f"  ✗ WARNING: Min token {min_token} < 3 (should be >= 3)")
    else:
        print(f"  ✓ Min token OK (>= 3)")

    if max_token >= vocab_size:
        print(f"  ✗ WARNING: Max token {max_token} >= {vocab_size} (out of range)")
    else:
        print(f"  ✓ Max token OK (< {vocab_size})")

    # Show examples of invalid sequences
    if invalid_sequences:
        print(f"\n{'-'*60}")
        print("Examples of Invalid Sequences")
        print(f"{'-'*60}")
        for seq_info in invalid_sequences[:5]:
            print(f"\nSample: {seq_info['sample_id']}")
            print(f"  Issues: {', '.join(seq_info['issues'])}")
            print(f"  Target: {seq_info['target_seq'][:20]}{'...' if len(seq_info['target_seq']) > 20 else ''}")

    # Vocabulary consistency check
    print(f"\n{'-'*60}")
    print("Vocabulary Consistency")
    print(f"{'-'*60}")

    # Get inverse vocabulary
    idx_to_sign = {idx: sign for sign, idx in vocab.items()}

    # Check if all target tokens can be decoded
    unmapped_tokens = set()
    for token_id in token_counter.keys():
        if token_id not in idx_to_sign:
            unmapped_tokens.add(token_id)

    if unmapped_tokens:
        print(f"✗ Found {len(unmapped_tokens)} tokens not in vocabulary:")
        print(f"  {list(unmapped_tokens)[:20]}")
    else:
        print("✓ All target tokens can be mapped to vocabulary")

    # Show example decoded sequences
    print(f"\n{'-'*60}")
    print("Example Decoded Sequences")
    print(f"{'-'*60}")

    for batch_idx, batch in enumerate(dataloader):
        targets = batch['targets']
        target_lengths = batch['target_lengths']
        sample_ids = batch['sample_ids']

        for i in range(min(3, targets.size(0))):
            seq_len = target_lengths[i].item()
            target_seq = targets[i, :seq_len].tolist()

            # Decode to text
            decoded = []
            for token_id in target_seq:
                if token_id in idx_to_sign:
                    sign = idx_to_sign[token_id]
                    if sign not in ['<PAD>', '<BLANK>', '<UNK>']:
                        decoded.append(sign)
                else:
                    decoded.append(f"<UNK_{token_id}>")

            print(f"\n{sample_ids[i]}:")
            print(f"  IDs: {target_seq[:10]}{'...' if len(target_seq) > 10 else ''}")
            print(f"  Text: {' '.join(decoded[:10])}{'...' if len(decoded) > 10 else ''}")

        if batch_idx >= 0:  # Only show first batch
            break

    print(f"\n{'='*60}")
    print("Target Validation Complete")
    print(f"{'='*60}\n")

    return {
        'total_sequences': total_sequences,
        'total_tokens': total_tokens,
        'pad_contamination': pad_contamination,
        'blank_contamination': blank_contamination,
        'unk_contamination': unk_contamination,
        'out_of_range': out_of_range,
        'invalid_sequences': len(invalid_sequences),
        'unmapped_tokens': len(unmapped_tokens),
    }


def main():
    print("\nLoading datasets...")

    # Load dataloaders
    train_loader, dev_loader, test_loader = create_dataloaders(
        data_root='data/raw_data/phoenix-2014-signerindependent-SI5',
        features_root='data/processed',
        batch_size=32,
        num_workers=0,
    )

    # Get vocabulary
    vocab = train_loader.dataset.vocab

    # Validate training set
    train_stats = validate_targets(train_loader, 'train', vocab, num_batches=50)

    # Validate dev set
    dev_stats = validate_targets(dev_loader, 'dev', vocab, num_batches=10)

    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    # Check for critical issues
    issues = []

    if train_stats['pad_contamination'] > 0:
        issues.append(f"✗ PAD tokens (0) found in {train_stats['pad_contamination']} sequences")

    if train_stats['blank_contamination'] > 0:
        issues.append(f"✗ BLANK tokens (1) found in {train_stats['blank_contamination']} sequences")

    if train_stats['unk_contamination'] > 0:
        issues.append(f"⚠ UNK tokens (2) found in {train_stats['unk_contamination']} sequences")

    if train_stats['out_of_range'] > 0:
        issues.append(f"✗ Out-of-range tokens in {train_stats['out_of_range']} sequences")

    if train_stats['unmapped_tokens'] > 0:
        issues.append(f"✗ {train_stats['unmapped_tokens']} tokens cannot be mapped to vocabulary")

    if issues:
        print("\nCRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\nRECOMMENDATION: Fix dataset preprocessing or vocabulary mapping")
    else:
        print("\n✓ No critical issues detected")
        print("Target sequences appear to be correctly formatted")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
