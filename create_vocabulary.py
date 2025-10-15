"""
Create vocabulary file from training corpus annotations.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

def create_vocabulary():
    """Build and save vocabulary from training corpus."""

    # Load training corpus
    corpus_file = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")

    if not corpus_file.exists():
        print(f"ERROR: Corpus file not found: {corpus_file}")
        return

    print("Loading training corpus...")
    df = pd.read_csv(corpus_file, delimiter="|")
    print(f"Loaded {len(df)} sequences")

    # Extract all unique signs
    all_signs = set()
    sign_counts = defaultdict(int)
    total_tokens = 0

    for annotation in df['annotation']:
        signs = annotation.split()
        for sign in signs:
            all_signs.add(sign)
            sign_counts[sign] += 1
            total_tokens += 1

    print(f"\nVocabulary statistics:")
    print(f"  Unique signs: {len(all_signs)}")
    print(f"  Total sign tokens: {total_tokens:,}")
    print(f"  Average signs per sequence: {total_tokens / len(df):.1f}")

    # Create vocabulary with special tokens
    vocab = {
        '<PAD>': 0,    # Padding token
        '<BLANK>': 1,  # CTC blank token
        '<UNK>': 2,    # Unknown token
    }

    # Add all signs (sorted alphabetically for consistency)
    for sign in sorted(all_signs):
        if sign not in vocab:
            vocab[sign] = len(vocab)

    print(f"\nTotal vocabulary size: {len(vocab)}")
    print(f"  Special tokens: 3 (<PAD>, <BLANK>, <UNK>)")
    print(f"  Sign vocabulary: {len(vocab) - 3}")

    # Save vocabulary
    output_dir = Path("data/processed/train")
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_file = output_dir / "vocabulary.txt"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for sign, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{sign}\t{idx}\n")

    print(f"\nVocabulary saved to: {vocab_file.absolute()}")

    # Show most common signs
    sorted_signs = sorted(sign_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 20 most frequent signs:")
    for i, (sign, count) in enumerate(sorted_signs[:20], 1):
        pct = 100 * count / total_tokens
        print(f"  {i:2d}. {sign:20s} {count:5d} ({pct:5.2f}%)")

    print("\nVocabulary creation complete!")

if __name__ == "__main__":
    create_vocabulary()
