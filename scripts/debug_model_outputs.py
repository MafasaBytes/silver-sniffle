"""
Debug script to inspect model outputs and CTC predictions.
Check if model is actually producing varied predictions or just blanks.
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.bilstm import BiLSTMModel
from phoenix_dataset import create_dataloaders


def analyze_model_predictions(checkpoint_path, device='cuda'):
    """Analyze what the model is actually predicting."""
    print(f"\n{'='*60}")
    print("Model Output Analysis")
    print(f"{'='*60}\n")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = BiLSTMModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        vocab_size=config['vocab_size'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load data
    _, dev_loader, _ = create_dataloaders(
        data_root='data/raw_data/phoenix-2014-signerindependent-SI5',
        features_root='data/processed',
        batch_size=4,
        num_workers=0,
    )

    # Analyze first batch
    batch = next(iter(dev_loader))
    features = batch['features'].to(device)
    targets = batch['targets'].to(device)
    feature_lengths = batch['feature_lengths'].to(device)
    target_lengths = batch['target_lengths'].to(device)

    print(f"Batch info:")
    print(f"  Features shape: {features.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Feature lengths: {feature_lengths.tolist()}")
    print(f"  Target lengths: {target_lengths.tolist()}")

    # Forward pass
    with torch.no_grad():
        log_probs, output_lengths = model(features, feature_lengths)

    print(f"\nModel output:")
    print(f"  Log probs shape: {log_probs.shape}  # (T, N, vocab_size)")
    print(f"  Output lengths: {output_lengths.tolist()}")

    # Analyze predictions
    print(f"\n{'-'*60}")
    print("Prediction Analysis")
    print(f"{'-'*60}\n")

    # Get most likely token at each timestep
    predictions = log_probs.argmax(dim=-1)  # (T, N)

    for sample_idx in range(min(2, predictions.size(1))):
        print(f"Sample {sample_idx + 1}:")
        print(f"  Reference ({target_lengths[sample_idx]} signs): {targets[sample_idx, :target_lengths[sample_idx]].tolist()}")

        sample_preds = predictions[:output_lengths[sample_idx], sample_idx].cpu().numpy()
        print(f"  Predictions ({output_lengths[sample_idx]} frames): {sample_preds.tolist()[:30]}...")

        # Count token occurrences
        unique, counts = torch.unique(predictions[:output_lengths[sample_idx], sample_idx], return_counts=True)
        print(f"  Unique tokens predicted: {len(unique)}")
        print(f"  Token distribution:")
        for token_id, count in zip(unique.tolist(), counts.tolist()):
            percentage = count / output_lengths[sample_idx].item() * 100
            if percentage > 5.0:  # Only show tokens that appear >5%
                print(f"    Token {token_id}: {count:>4} ({percentage:>5.1f}%)")

        # Check if BLANK (token 1) dominates
        blank_count = (predictions[:output_lengths[sample_idx], sample_idx] == 1).sum().item()
        blank_percentage = blank_count / output_lengths[sample_idx].item() * 100
        print(f"  BLANK token (1) percentage: {blank_percentage:.1f}%")

        print()

    # Global statistics across all samples
    print(f"{'-'*60}")
    print("Global Prediction Statistics")
    print(f"{'-'*60}\n")

    all_predictions = []
    blank_percentages = []

    with torch.no_grad():
        for batch in dev_loader:
            features = batch['features'].to(device)
            feature_lengths = batch['feature_lengths'].to(device)

            log_probs, output_lengths = model(features, feature_lengths)
            predictions = log_probs.argmax(dim=-1)  # (T, N)

            for sample_idx in range(predictions.size(1)):
                sample_preds = predictions[:output_lengths[sample_idx], sample_idx].cpu()
                all_predictions.extend(sample_preds.tolist())

                blank_count = (sample_preds == 1).sum().item()
                blank_percentage = blank_count / output_lengths[sample_idx].item() * 100
                blank_percentages.append(blank_percentage)

    # Overall token distribution
    pred_tensor = torch.tensor(all_predictions)
    unique, counts = torch.unique(pred_tensor, return_counts=True)

    print(f"Overall prediction statistics (dev set):")
    print(f"  Total frames: {len(all_predictions):,}")
    print(f"  Unique tokens predicted: {len(unique)}")
    print(f"  Average BLANK percentage: {sum(blank_percentages)/len(blank_percentages):.1f}%")

    print(f"\nTop 10 most predicted tokens:")
    sorted_indices = counts.argsort(descending=True)
    for i in range(min(10, len(unique))):
        token_id = unique[sorted_indices[i]].item()
        count = counts[sorted_indices[i]].item()
        percentage = count / len(all_predictions) * 100
        print(f"  Token {token_id:>4}: {count:>6,} ({percentage:>5.1f}%)")

    # Check confidence
    print(f"\n{'-'*60}")
    print("Prediction Confidence Analysis")
    print(f"{'-'*60}\n")

    with torch.no_grad():
        batch = next(iter(dev_loader))
        features = batch['features'].to(device)
        feature_lengths = batch['feature_lengths'].to(device)

        log_probs, output_lengths = model(features, feature_lengths)
        probs = torch.exp(log_probs)  # Convert log probs to probs

        max_probs = probs.max(dim=-1)[0]  # Max probability at each timestep

        print(f"Confidence statistics:")
        print(f"  Mean max probability: {max_probs.mean():.4f}")
        print(f"  Min max probability: {max_probs.min():.4f}")
        print(f"  Max max probability: {max_probs.max():.4f}")

        # Check entropy (low entropy = high confidence)
        entropy = -(probs * log_probs).sum(dim=-1)
        print(f"\nPrediction entropy:")
        print(f"  Mean entropy: {entropy.mean():.4f}")
        print(f"  Min entropy: {entropy.min():.4f}")
        print(f"  Max entropy: {entropy.max():.4f}")
        print(f"  (Low entropy = confident, High entropy = uncertain)")

    print(f"\n{'='*60}")
    print("Analysis Complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    checkpoint_path = "models/bilstm_baseline/best_model.pth"
    analyze_model_predictions(checkpoint_path)
