"""
Code snippets for monitoring CTC loss and target validation during training.

These snippets can be added to your training script (src/train_bilstm.py)
to catch potential issues early.
"""

import torch
import torch.nn as nn


# ==============================================================================
# SNIPPET 1: First Batch Validation (Add to CTCTrainer.train() method)
# ==============================================================================
def validate_first_batch(self):
    """
    Run validation on the first training batch to catch setup issues early.
    Add this at the beginning of the train() method.
    """
    print(f"\n{'='*60}")
    print("First Batch Validation")
    print(f"{'='*60}")

    # Get first batch
    first_batch = next(iter(self.train_loader))
    features = first_batch['features'].to(self.device)
    targets = first_batch['targets'].to(self.device)
    feature_lengths = first_batch['feature_lengths'].to(self.device)
    target_lengths = first_batch['target_lengths'].to(self.device)

    # Check target validity
    print("\nTarget validation:")
    for i in range(min(3, targets.size(0))):  # Check first 3 samples
        actual_targets = targets[i, :target_lengths[i]]
        print(f"  Sample {i}:")
        print(f"    Target length: {target_lengths[i].item()}")
        print(f"    Min index: {actual_targets.min().item()}")
        print(f"    Max index: {actual_targets.max().item()}")
        print(f"    Sequence: {actual_targets.tolist()[:10]}...")  # First 10 tokens

        # Assertions
        assert actual_targets.min() >= 3, \
            f"Sample {i} contains PAD/BLANK/UNK in actual sequence!"
        assert actual_targets.max() < self.config['vocab_size'], \
            f"Sample {i} has target index >= vocab_size!"

    # Forward pass
    self.model.train()
    log_probs, output_lengths = self.model(features, feature_lengths)
    loss = self.criterion(log_probs, targets, output_lengths, target_lengths)

    # Check loss
    print("\nLoss validation:")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss is finite: {torch.isfinite(loss).item()}")
    print(f"  Loss requires_grad: {loss.requires_grad}")

    assert torch.isfinite(loss), "First batch produced non-finite loss!"

    # Check sequence length constraint (T >= 2*S + 1 for CTC)
    print("\nSequence length constraints:")
    for i in range(min(3, targets.size(0))):
        T = output_lengths[i].item()
        S = target_lengths[i].item()
        min_T = 2 * S + 1
        print(f"  Sample {i}: T={T}, S={S}, min_T={min_T}, OK={T >= min_T}")
        if T < min_T:
            print(f"    WARNING: Output length too short for CTC!")

    # Backward pass (test gradients)
    self.optimizer.zero_grad()
    loss.backward()

    # Check gradients
    grad_norms = []
    for name, param in self.model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)

    print("\nGradient validation:")
    print(f"  Parameters with gradients: {len(grad_norms)}")
    print(f"  Mean grad norm: {sum(grad_norms)/len(grad_norms):.6f}")
    print(f"  Max grad norm: {max(grad_norms):.6f}")
    print(f"  Has NaN gradients: {any(torch.isnan(p.grad).any() for p in self.model.parameters() if p.grad is not None)}")

    print(f"\n{'='*60}")
    print("First Batch Validation: PASSED")
    print(f"{'='*60}\n")


# ==============================================================================
# SNIPPET 2: Periodic Target Validation (Add to train_epoch() method)
# ==============================================================================
def validate_batch_targets(targets, target_lengths, vocab_size, batch_idx):
    """
    Validate targets in a batch. Call this periodically (e.g., every 100 batches).

    Usage in train_epoch():
        if batch_idx % 100 == 0:
            validate_batch_targets(targets, target_lengths, self.config['vocab_size'], batch_idx)
    """
    batch_size = targets.size(0)

    for i in range(batch_size):
        actual_targets = targets[i, :target_lengths[i]]

        # Check for PAD/BLANK/UNK in actual sequence
        if actual_targets.min() < 3:
            print(f"\nWARNING in batch {batch_idx}, sample {i}:")
            print(f"  Found PAD/BLANK/UNK token in actual sequence!")
            print(f"  Min index: {actual_targets.min().item()}")
            print(f"  Sequence: {actual_targets.tolist()}")
            raise ValueError("Invalid target sequence detected!")

        # Check for out-of-vocab indices
        if actual_targets.max() >= vocab_size:
            print(f"\nWARNING in batch {batch_idx}, sample {i}:")
            print(f"  Found out-of-vocab index!")
            print(f"  Max index: {actual_targets.max().item()}")
            print(f"  Vocab size: {vocab_size}")
            raise ValueError("Target index exceeds vocabulary size!")


# ==============================================================================
# SNIPPET 3: CTC Sequence Length Checker (Add to train_epoch() method)
# ==============================================================================
def check_ctc_length_constraint(output_lengths, target_lengths, batch_idx):
    """
    Check that CTC sequence length constraint is satisfied: T >= 2*S + 1

    Usage in train_epoch():
        check_ctc_length_constraint(output_lengths, target_lengths, batch_idx)
    """
    batch_size = output_lengths.size(0)

    for i in range(batch_size):
        T = output_lengths[i].item()
        S = target_lengths[i].item()
        min_T = 2 * S + 1

        if T < min_T:
            print(f"\nWARNING in batch {batch_idx}, sample {i}:")
            print(f"  CTC constraint violated: T < 2*S + 1")
            print(f"  T (output length): {T}")
            print(f"  S (target length): {S}")
            print(f"  Minimum T required: {min_T}")
            print(f"  This may cause CTC alignment issues!")


# ==============================================================================
# SNIPPET 4: Loss Monitoring (Add to train_epoch() method)
# ==============================================================================
def monitor_loss_health(loss, batch_idx, window_size=100):
    """
    Monitor loss for anomalies (NaN, Inf, sudden spikes).

    Usage in train_epoch():
        if not hasattr(self, 'loss_history'):
            self.loss_history = []

        monitor_loss_health(loss, batch_idx, loss_history=self.loss_history)
    """
    # Check for non-finite loss
    if not torch.isfinite(loss):
        print(f"\n{'!'*60}")
        print(f"CRITICAL ERROR in batch {batch_idx}:")
        print(f"  Loss is {'NaN' if torch.isnan(loss) else 'Inf'}!")
        print(f"  This indicates a serious training issue.")
        print(f"{'!'*60}\n")
        raise ValueError("Non-finite loss detected!")

    # Check for extreme loss values
    loss_val = loss.item()
    if loss_val > 1000.0:
        print(f"\nWARNING in batch {batch_idx}:")
        print(f"  Loss is very large: {loss_val:.2f}")
        print(f"  This may indicate a problem.")


# ==============================================================================
# SNIPPET 5: Gradient Monitoring (Add to train_epoch() after backward())
# ==============================================================================
def monitor_gradient_health(model, batch_idx, check_interval=100):
    """
    Monitor gradient statistics to detect vanishing/exploding gradients.

    Usage in train_epoch() (after loss.backward(), before optimizer.step()):
        if batch_idx % 100 == 0:
            monitor_gradient_health(self.model, batch_idx)
    """
    if batch_idx % check_interval != 0:
        return

    grad_norms = []
    grad_max = 0.0
    grad_min = float('inf')

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            grad_max = max(grad_max, param.grad.abs().max().item())
            grad_min = min(grad_min, param.grad.abs().min().item())

    if grad_norms:
        mean_norm = sum(grad_norms) / len(grad_norms)
        max_norm = max(grad_norms)

        print(f"\nGradient stats (batch {batch_idx}):")
        print(f"  Mean grad norm: {mean_norm:.6f}")
        print(f"  Max grad norm: {max_norm:.6f}")
        print(f"  Max grad value: {grad_max:.6f}")
        print(f"  Min grad value: {grad_min:.6f}")

        # Check for vanishing gradients
        if mean_norm < 1e-7:
            print(f"  WARNING: Gradients are very small (vanishing gradients?)")

        # Check for exploding gradients (before clipping)
        if max_norm > 100.0:
            print(f"  WARNING: Gradients are very large (exploding gradients?)")


# ==============================================================================
# SNIPPET 6: Complete Enhanced train_epoch() Method
# ==============================================================================
def train_epoch_enhanced(self, epoch):
    """
    Enhanced training loop with comprehensive monitoring.
    Replace the existing train_epoch() method with this version.
    """
    self.model.train()
    total_loss = 0

    # Initialize loss history if not exists
    if not hasattr(self, 'loss_history'):
        self.loss_history = []

    pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        features = batch['features'].to(self.device)
        targets = batch['targets'].to(self.device)
        feature_lengths = batch['feature_lengths'].to(self.device)
        target_lengths = batch['target_lengths'].to(self.device)

        # VALIDATION: Check targets every 100 batches
        if batch_idx % 100 == 0:
            validate_batch_targets(targets, target_lengths,
                                 self.config['vocab_size'], batch_idx)

        # Forward pass
        log_probs, output_lengths = self.model(features, feature_lengths)

        # VALIDATION: Check CTC length constraint
        if batch_idx % 100 == 0:
            check_ctc_length_constraint(output_lengths, target_lengths, batch_idx)

        # CTC loss with 2D targets
        loss = self.criterion(
            log_probs,           # (T, N, C)
            targets,             # (N, S) - padded targets
            output_lengths,      # (N,)
            target_lengths       # (N,)
        )

        # VALIDATION: Monitor loss health
        monitor_loss_health(loss, batch_idx)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # VALIDATION: Monitor gradients every 100 batches
        if batch_idx % 100 == 0:
            monitor_gradient_health(self.model, batch_idx)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['grad_clip']
        )

        self.optimizer.step()

        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        # Log to TensorBoard
        global_step = epoch * len(self.train_loader) + batch_idx
        self.writer.add_scalar('train/loss_step', loss.item(), global_step)

        # Store loss history
        self.loss_history.append(loss.item())
        if len(self.loss_history) > 1000:  # Keep last 1000
            self.loss_history.pop(0)

    avg_loss = total_loss / len(self.train_loader)
    return avg_loss


# ==============================================================================
# SNIPPET 7: Add to CTCTrainer.__init__() for first batch validation
# ==============================================================================
def enhanced_init_addition(self):
    """
    Add this at the end of CTCTrainer.__init__() to run first batch validation.
    """
    # ... existing init code ...

    print(f"\nRunning first batch validation...")
    self.validate_first_batch()  # Validate setup with first batch


# ==============================================================================
# EXAMPLE: How to integrate into train_bilstm.py
# ==============================================================================
"""
1. Add validate_first_batch() as a method to CTCTrainer class

2. Call it at the end of __init__():
   def __init__(self, config):
       # ... existing init code ...
       self.validate_first_batch()  # Add this line

3. Replace train_epoch() with train_epoch_enhanced()

4. Add the utility functions as standalone functions or class methods

This will give you comprehensive monitoring during training!
"""


# ==============================================================================
# SNIPPET 8: Overfitting Test (Separate script to run)
# ==============================================================================
def test_overfit_single_batch():
    """
    Sanity check: Test that the model can overfit a single batch.
    This verifies that the model architecture and loss function work correctly.

    Run this as a separate script before full training.
    """
    import torch
    import torch.nn as nn
    from models.bilstm import create_bilstm_model
    from phoenix_dataset import create_dataloaders

    print("\n" + "="*60)
    print("Overfitting Test: Single Batch")
    print("="*60)

    # Load data
    train_loader, _, _ = create_dataloaders(
        data_root='data/raw_data/phoenix-2014-signerindependent-SI5',
        features_root='data/processed',
        batch_size=32,
        num_workers=0,
        max_sequence_length=241
    )

    # Get one batch
    batch = next(iter(train_loader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features = batch['features'].to(device)
    targets = batch['targets'].to(device)
    feature_lengths = batch['feature_lengths'].to(device)
    target_lengths = batch['target_lengths'].to(device)

    print(f"\nBatch info:")
    print(f"  Batch size: {features.size(0)}")
    print(f"  Max sequence length: {features.size(1)}")
    print(f"  Feature dim: {features.size(2)}")

    # Create model
    model = create_bilstm_model(
        input_dim=177,
        hidden_dim=256,
        num_layers=2,
        vocab_size=1120,
        dropout=0.3,
        device=str(device)
    )

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=1, zero_infinity=True)

    # Train on this batch for 200 iterations
    print(f"\nTraining on single batch for 200 iterations...")
    print(f"{'Iter':<6} {'Loss':<12} {'Status'}")
    print(f"{'-'*30}")

    initial_loss = None
    for i in range(200):
        model.train()
        optimizer.zero_grad()

        log_probs, output_lengths = model(features, feature_lengths)
        loss = criterion(log_probs, targets, output_lengths, target_lengths)

        if i == 0:
            initial_loss = loss.item()

        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        if i % 20 == 0:
            status = "✓" if loss.item() < initial_loss * 0.5 else ""
            print(f"{i:<6} {loss.item():<12.4f} {status}")

    final_loss = loss.item()

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Reduction:    {(1 - final_loss/initial_loss)*100:.1f}%")

    if final_loss < initial_loss * 0.3:  # 70% reduction
        print(f"\n  STATUS: PASSED ✓")
        print(f"  Model can successfully learn from data.")
    else:
        print(f"\n  STATUS: FAILED ✗")
        print(f"  Model failed to overfit single batch.")
        print(f"  This may indicate an issue with the architecture or loss function.")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    print(__doc__)
    print("\nThis file contains code snippets for monitoring.")
    print("See the comments in each snippet for usage instructions.")
    print("\nTo run the overfitting test:")
    print("  python -c 'from ctc_monitoring_snippets import test_overfit_single_batch; test_overfit_single_batch()'")
