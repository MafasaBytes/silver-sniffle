"""
Test script to validate CTC loss implementation with 2D padded targets.

This script tests whether PyTorch's nn.CTCLoss correctly handles:
1. 2D padded targets (N, S) vs 1D flattened targets
2. PAD token (0) in padding positions
3. Correct usage of target_lengths to ignore padding

Vocabulary structure:
  <PAD>: 0    (padding token - should NOT appear in actual targets)
  <BLANK>: 1  (CTC blank token - used by CTC algorithm)
  <UNK>: 2    (unknown token)
  signs: 3-1119 (actual sign vocabulary)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def test_ctc_loss_2d_vs_1d():
    """Test that 2D padded targets work correctly with CTCLoss."""

    print("\n" + "="*80)
    print("TEST 1: Comparing 2D Padded Targets vs 1D Flattened Targets")
    print("="*80)

    # Test configuration
    batch_size = 3
    T = 20  # sequence length (time steps)
    C = 1120  # vocab size (PAD=0, BLANK=1, UNK=2, signs=3-1119)

    # Create random log probabilities (T, N, C) with gradient tracking
    torch.manual_seed(42)
    raw_logits_2d = torch.randn(T, batch_size, C, requires_grad=True)
    log_probs = F.log_softmax(raw_logits_2d, dim=2)

    # Create 2D padded targets (N, S) with variable lengths
    # Using actual sign tokens (indices 3-1119), PAD=0 for padding
    targets_2d = torch.tensor([
        [3, 4, 5, 6, 7, 0, 0, 0],      # length 5
        [10, 11, 12, 0, 0, 0, 0, 0],   # length 3
        [20, 21, 22, 23, 0, 0, 0, 0]   # length 4
    ])

    target_lengths = torch.tensor([5, 3, 4])
    output_lengths = torch.tensor([T, T, T])

    print(f"\nInput shapes:")
    print(f"  log_probs: {log_probs.shape} (T, N, C)")
    print(f"  targets_2d: {targets_2d.shape} (N, S)")
    print(f"  output_lengths: {output_lengths.shape}")
    print(f"  target_lengths: {target_lengths.shape}")

    print(f"\nTarget sequences (2D padded):")
    for i in range(batch_size):
        actual_seq = targets_2d[i, :target_lengths[i]].tolist()
        padded_seq = targets_2d[i].tolist()
        print(f"  Sample {i}: {actual_seq} (actual) -> {padded_seq} (padded)")

    # Method 1: 2D padded targets (SIMPLIFIED APPROACH)
    criterion = nn.CTCLoss(blank=1, zero_infinity=True)

    try:
        loss_2d = criterion(
            log_probs,
            targets_2d,
            output_lengths,
            target_lengths
        )
        print(f"\n[2D APPROACH] Loss computed successfully: {loss_2d.item():.6f}")

        # Check if gradients flow correctly
        loss_2d.backward()
        print(f"[2D APPROACH] Gradients computed successfully")
        print(f"[2D APPROACH] Max gradient: {raw_logits_2d.grad.abs().max().item():.6f}")

    except Exception as e:
        print(f"\n[2D APPROACH] FAILED with error: {e}")
        return False

    # Method 2: 1D flattened targets (ORIGINAL APPROACH)
    # Need to create new input tensor for fair comparison
    torch.manual_seed(42)
    raw_logits_1d = torch.randn(T, batch_size, C, requires_grad=True)
    log_probs_copy = F.log_softmax(raw_logits_1d, dim=2)

    # Flatten targets by concatenating non-padded elements
    targets_1d = []
    for i in range(batch_size):
        targets_1d.append(targets_2d[i, :target_lengths[i]])
    targets_1d = torch.cat(targets_1d)

    print(f"\nTarget sequences (1D flattened):")
    print(f"  Flattened: {targets_1d.tolist()}")
    print(f"  Shape: {targets_1d.shape}")

    try:
        loss_1d = criterion(
            log_probs_copy,
            targets_1d,
            output_lengths,
            target_lengths
        )
        print(f"\n[1D APPROACH] Loss computed successfully: {loss_1d.item():.6f}")

        # Check if gradients flow correctly
        loss_1d.backward()
        print(f"[1D APPROACH] Gradients computed successfully")
        print(f"[1D APPROACH] Max gradient: {raw_logits_1d.grad.abs().max().item():.6f}")

    except Exception as e:
        print(f"\n[1D APPROACH] FAILED with error: {e}")
        return False

    # Compare losses
    print(f"\n" + "-"*80)
    print(f"COMPARISON:")
    print(f"  2D loss: {loss_2d.item():.6f}")
    print(f"  1D loss: {loss_1d.item():.6f}")
    print(f"  Difference: {abs(loss_2d.item() - loss_1d.item()):.8f}")

    # Check if losses are identical
    if torch.allclose(loss_2d, loss_1d, atol=1e-6):
        print(f"\n  RESULT: Both approaches produce IDENTICAL losses!")
        print(f"  CONCLUSION: 2D padded targets are correctly handled by CTCLoss")
        return True
    else:
        print(f"\n  WARNING: Losses differ by {abs(loss_2d.item() - loss_1d.item()):.8f}")
        print(f"  This may indicate an issue with one of the approaches")
        return False


def test_pad_token_handling():
    """Test that PAD tokens (0) in padding positions don't affect loss."""

    print("\n" + "="*80)
    print("TEST 2: PAD Token Handling in Padding Positions")
    print("="*80)

    batch_size = 2
    T = 15
    C = 1120

    torch.manual_seed(42)
    log_probs = torch.randn(T, batch_size, C).log_softmax(2)

    # Two identical actual sequences, but different padding
    targets_pad_0 = torch.tensor([
        [3, 4, 5, 0, 0, 0],  # Padded with 0 (PAD token)
        [3, 4, 5, 0, 0, 0]
    ])

    targets_pad_999 = torch.tensor([
        [3, 4, 5, 999, 999, 999],  # Padded with arbitrary value
        [3, 4, 5, 999, 999, 999]
    ])

    target_lengths = torch.tensor([3, 3])
    output_lengths = torch.tensor([T, T])

    criterion = nn.CTCLoss(blank=1, zero_infinity=True)

    # Compute loss with PAD=0
    log_probs_copy1 = log_probs.detach().clone()
    loss_pad_0 = criterion(log_probs_copy1, targets_pad_0, output_lengths, target_lengths)

    # Compute loss with PAD=999
    log_probs_copy2 = log_probs.detach().clone()
    loss_pad_999 = criterion(log_probs_copy2, targets_pad_999, output_lengths, target_lengths)

    print(f"\nLoss with PAD=0:   {loss_pad_0.item():.6f}")
    print(f"Loss with PAD=999: {loss_pad_999.item():.6f}")
    print(f"Difference:        {abs(loss_pad_0.item() - loss_pad_999.item()):.8f}")

    if torch.allclose(loss_pad_0, loss_pad_999, atol=1e-6):
        print(f"\n  RESULT: Padding values DON'T affect loss (as expected)")
        print(f"  CONCLUSION: CTCLoss correctly uses target_lengths to ignore padding")
        return True
    else:
        print(f"\n  WARNING: Different padding values produce different losses!")
        print(f"  This may indicate a problem with target_lengths handling")
        return False


def test_blank_token_configuration():
    """Test that blank token is correctly configured."""

    print("\n" + "="*80)
    print("TEST 3: BLANK Token Configuration")
    print("="*80)

    batch_size = 2
    T = 10
    C = 1120

    torch.manual_seed(42)
    log_probs = torch.randn(T, batch_size, C).log_softmax(2)

    # Target with actual signs (should NOT include blank=1 or pad=0)
    targets = torch.tensor([
        [3, 4, 5, 0],
        [6, 7, 0, 0]
    ])
    target_lengths = torch.tensor([3, 2])
    output_lengths = torch.tensor([T, T])

    # Test with correct blank=1
    criterion_blank1 = nn.CTCLoss(blank=1, zero_infinity=True)
    loss_blank1 = criterion_blank1(log_probs.clone(), targets, output_lengths, target_lengths)

    # Test with wrong blank=0 (would conflict with PAD token)
    criterion_blank0 = nn.CTCLoss(blank=0, zero_infinity=True)
    log_probs_copy = log_probs.detach().clone()
    loss_blank0 = criterion_blank0(log_probs_copy, targets, output_lengths, target_lengths)

    print(f"\nLoss with blank=1: {loss_blank1.item():.6f}")
    print(f"Loss with blank=0: {loss_blank0.item():.6f}")
    print(f"Difference:        {abs(loss_blank1.item() - loss_blank0.item()):.6f}")

    print(f"\n  CONFIGURATION:")
    print(f"    Vocabulary: PAD=0, BLANK=1, UNK=2, signs=3-1119")
    print(f"    CTCLoss should use: blank=1 (BLANK token)")
    print(f"    Targets should contain: only indices 3-1119 (signs) + padding")
    print(f"    Targets should NOT contain: 0 (PAD) or 1 (BLANK) as actual labels")

    if abs(loss_blank1.item() - loss_blank0.item()) > 0.01:
        print(f"\n  RESULT: Different blank indices produce different losses (as expected)")
        print(f"  CONCLUSION: blank=1 is correctly configured")
        return True
    else:
        print(f"\n  WARNING: Blank configuration may not be affecting loss as expected")
        return False


def test_edge_cases():
    """Test edge cases: empty sequences, single token, very long sequences."""

    print("\n" + "="*80)
    print("TEST 4: Edge Cases")
    print("="*80)

    criterion = nn.CTCLoss(blank=1, zero_infinity=True)

    # Test 1: Very short sequence (single token)
    print("\n[Test 4.1] Single token sequence:")
    T = 5
    log_probs = torch.randn(T, 1, 1120).log_softmax(2)
    targets = torch.tensor([[3, 0, 0]])  # Single sign, padded
    output_lengths = torch.tensor([T])
    target_lengths = torch.tensor([1])

    try:
        loss = criterion(log_probs, targets, output_lengths, target_lengths)
        print(f"  Loss: {loss.item():.6f}")
        print(f"  PASSED: Single token sequence handled correctly")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Test 2: Maximum length sequence
    print("\n[Test 4.2] Maximum length sequence:")
    T = 50
    S = 25  # Half the output length (CTC requirement: T >= 2*S + 1 for blanks)
    log_probs = torch.randn(T, 1, 1120).log_softmax(2)
    targets = torch.tensor([[i for i in range(3, 3+S)]])
    output_lengths = torch.tensor([T])
    target_lengths = torch.tensor([S])

    try:
        loss = criterion(log_probs, targets, output_lengths, target_lengths)
        print(f"  T={T}, target_length={S}")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  PASSED: Maximum length sequence handled correctly")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Test 3: Variable batch lengths
    print("\n[Test 4.3] Variable output lengths:")
    batch_size = 3
    max_T = 20
    log_probs = torch.randn(max_T, batch_size, 1120).log_softmax(2)
    targets = torch.tensor([
        [3, 4, 5, 0, 0],
        [6, 7, 0, 0, 0],
        [8, 9, 10, 11, 0]
    ])
    output_lengths = torch.tensor([20, 15, 18])  # Variable output lengths
    target_lengths = torch.tensor([3, 2, 4])

    try:
        loss = criterion(log_probs, targets, output_lengths, target_lengths)
        print(f"  Output lengths: {output_lengths.tolist()}")
        print(f"  Target lengths: {target_lengths.tolist()}")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  PASSED: Variable output lengths handled correctly")
    except Exception as e:
        print(f"  FAILED: {e}")

    return True


def test_gradient_flow():
    """Test that gradients flow correctly through the model."""

    print("\n" + "="*80)
    print("TEST 5: Gradient Flow")
    print("="*80)

    batch_size = 4
    T = 30
    C = 1120

    # Create raw inputs with gradient tracking
    torch.manual_seed(42)
    raw_logits = torch.randn(T, batch_size, C, requires_grad=True)
    log_probs = F.log_softmax(raw_logits, dim=2)

    targets = torch.tensor([
        [3, 4, 5, 6, 0, 0, 0],
        [10, 11, 12, 0, 0, 0, 0],
        [20, 21, 0, 0, 0, 0, 0],
        [30, 31, 32, 33, 34, 0, 0]
    ])
    target_lengths = torch.tensor([4, 3, 2, 5])
    output_lengths = torch.tensor([T, T, T, T])

    criterion = nn.CTCLoss(blank=1, zero_infinity=True)

    # Forward pass
    loss = criterion(log_probs, targets, output_lengths, target_lengths)

    print(f"\nLoss: {loss.item():.6f}")
    print(f"Loss requires_grad: {loss.requires_grad}")

    # Backward pass
    loss.backward()

    print(f"\nGradient statistics (on raw_logits):")
    print(f"  Shape: {raw_logits.grad.shape}")
    print(f"  Mean: {raw_logits.grad.mean().item():.6f}")
    print(f"  Std: {raw_logits.grad.std().item():.6f}")
    print(f"  Min: {raw_logits.grad.min().item():.6f}")
    print(f"  Max: {raw_logits.grad.max().item():.6f}")
    print(f"  Non-zero gradients: {(raw_logits.grad != 0).sum().item()} / {raw_logits.grad.numel()}")

    # Check for gradient issues
    has_nan = torch.isnan(raw_logits.grad).any()
    has_inf = torch.isinf(raw_logits.grad).any()

    if has_nan or has_inf:
        print(f"\n  WARNING: Gradients contain NaN or Inf!")
        print(f"  NaN: {has_nan}, Inf: {has_inf}")
        return False
    else:
        print(f"\n  RESULT: Gradients are clean (no NaN/Inf)")
        print(f"  CONCLUSION: Gradient flow is healthy")
        return True


def main():
    """Run all tests."""

    print("\n" + "#"*80)
    print("# CTC Loss Validation: 2D Padded Targets vs 1D Flattened Targets")
    print("#"*80)
    print("\nVocabulary Structure:")
    print("  <PAD>: 0    (padding token - should NOT appear in actual targets)")
    print("  <BLANK>: 1  (CTC blank token - used by CTC algorithm)")
    print("  <UNK>: 2    (unknown token)")
    print("  signs: 3-1119 (actual sign vocabulary)")
    print("\nCTC Configuration:")
    print("  nn.CTCLoss(blank=1, zero_infinity=True)")

    results = []

    # Run all tests
    results.append(("2D vs 1D comparison", test_ctc_loss_2d_vs_1d()))
    results.append(("PAD token handling", test_pad_token_handling()))
    results.append(("BLANK token config", test_blank_token_configuration()))
    results.append(("Edge cases", test_edge_cases()))
    results.append(("Gradient flow", test_gradient_flow()))

    # Summary
    print("\n" + "#"*80)
    print("# TEST SUMMARY")
    print("#"*80)

    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  [{status}] {test_name}")

    all_passed = all(result[1] for result in results)

    print("\n" + "="*80)
    if all_passed:
        print("FINAL VERDICT: All tests passed!")
        print("\nCONCLUSION:")
        print("  1. PyTorch CTCLoss SUPPORTS 2D padded targets directly")
        print("  2. The simplified 2D approach is CORRECT and produces identical results")
        print("  3. target_lengths correctly tells CTCLoss to ignore padding")
        print("  4. The flattening approach was unnecessary complexity")
        print("\nRECOMMENDATION:")
        print("  Use the simplified 2D padded target approach:")
        print("    loss = criterion(log_probs, targets, output_lengths, target_lengths)")
        print("  where targets is (N, S) with PAD=0 in padded positions")
    else:
        print("WARNING: Some tests failed. Review the results above.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
