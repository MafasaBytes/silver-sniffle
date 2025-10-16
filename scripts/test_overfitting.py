"""
Minimal overfitting test - try to memorize 10 samples.
If the model can't even overfit 10 samples, features are insufficient.
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import torch
import torch.nn as nn
from tqdm import tqdm
from models.bilstm import create_bilstm_model
from phoenix_dataset import create_dataloaders


def overfitting_test():
    """Test if model can overfit tiny dataset."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, _, _ = create_dataloaders(
        data_root='data/raw_data/phoenix-2014-signerindependent-SI5',
        features_root='data/processed',
        batch_size=20,  # Small batch
        num_workers=0,
        max_sequence_length=241
    )
    
    # Get 10 samples only
    print("Loading mini-batch...")
    mini_batch = next(iter(train_loader))
    
    print(f"Mini-batch size: {mini_batch['features'].shape[0]} samples")
    print(f"Feature shape: {mini_batch['features'].shape}")
    print(f"Target shape: {mini_batch['targets'].shape}")
    
    # Create model
    model = create_bilstm_model(
        input_dim=177,
        hidden_dim=256,
        num_layers=2,
        vocab_size=1120,
        dropout=0.0,  # No dropout for overfitting test
        device=str(device)
    )
    
    criterion = nn.CTCLoss(blank=1, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n" + "="*60)
    print("Overfitting Test: Trying to memorize 10 samples")
    print("="*60)
    print(f"Sample target sequence (first 10 tokens): {mini_batch['targets'][0, :10].tolist()}")
    print(f"Target length: {mini_batch['target_lengths'][0].item()}")
    print()
    
    # Train for many iterations on same batch
    model.train()
    for iteration in range(500):
        features = mini_batch['features'].to(device)
        targets = mini_batch['targets'].to(device)
        feature_lengths = mini_batch['feature_lengths'].to(device)
        target_lengths = mini_batch['target_lengths'].to(device)
        
        # Forward
        log_probs, output_lengths = model(features, feature_lengths)
        
        # Loss
        loss = criterion(log_probs, targets, output_lengths, target_lengths)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        # Check predictions every 50 iterations
        if iteration % 50 == 0:
            with torch.no_grad():
                predictions = log_probs.argmax(dim=-1).transpose(0, 1)
                pred_sample = predictions[0].cpu().tolist()[:20]
                
                # Count unique predictions
                unique_preds = len(set(predictions.flatten().cpu().tolist()))
                blank_ratio = (predictions == 1).float().mean().item()
                
                print(f"Iter {iteration:3d} | Loss: {loss.item():.4f} | "
                      f"Unique tokens: {unique_preds:4d} | "
                      f"Blank%: {blank_ratio*100:.1f}% | "
                      f"Sample pred: {pred_sample[:10]}")
    
    print("\n" + "="*60)
    print("Overfitting Test Results:")
    print("="*60)
    
    # Final analysis
    with torch.no_grad():
        features = mini_batch['features'].to(device)
        feature_lengths = mini_batch['feature_lengths'].to(device)
        log_probs, output_lengths = model(features, feature_lengths)
        predictions = log_probs.argmax(dim=-1).transpose(0, 1)
        
        unique_preds = len(set(predictions.flatten().cpu().tolist()))
        blank_ratio = (predictions == 1).float().mean().item()
        
        # Show first sample prediction
        pred_sample = predictions[0].cpu().tolist()
        target_sample = mini_batch['targets'][0].cpu().tolist()
        
        print(f"\nFinal Statistics:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Unique predictions: {unique_preds}")
        print(f"  Blank token ratio: {blank_ratio*100:.1f}%")
        print(f"\nSample 1:")
        print(f"  Target:     {target_sample[:20]}")
        print(f"  Prediction: {pred_sample[:20]}")
    
    print("\n" + "="*60)
    print("Diagnosis:")
    print("="*60)
    
    if blank_ratio > 0.95:
        print("❌ FAILED: Model predicting >95% blanks")
        print("\n   Root Cause: Features lack discriminative information")
        print("   → Body pose alone insufficient for sign language")
        print("   → Need hand keypoints (primary signal)")
        print("   → Need face keypoints (grammatical markers)")
        print("\n   Recommended Fix:")
        print("   1. Re-extract features using MediaPipe Holistic")
        print("   2. Include 21×2 hand landmarks + 468 face landmarks")
        print("   3. This matches your research proposal (543 total landmarks)")
    elif blank_ratio < 0.5 and unique_preds > 50:
        print("✅ PASSED: Model can learn from features")
        print("\n   → Features contain sufficient information")
        print("   → Problem is training configuration (LR, regularization)")
        print("\n   Recommended Fix:")
        print("   1. Try learning rate: 3e-5 (10x lower)")
        print("   2. Try different optimizer (SGD with momentum)")
        print("   3. Check data augmentation strategy")
    else:
        print("⚠️  PARTIAL: Some learning but weak signal")
        print(f"\n   Current stats:")
        print(f"   → Blank ratio: {blank_ratio*100:.1f}%")
        print(f"   → Unique predictions: {unique_preds}")
        print("\n   Likely cause: Features have limited information")
        print("   → Consider enriching features with hand/face keypoints")


if __name__ == "__main__":
    overfitting_test()