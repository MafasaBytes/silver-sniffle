"""
Training script for BiLSTM-CTC baseline model with comprehensive metrics.
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import time
import numpy as np

from models.bilstm import create_bilstm_model
from phoenix_dataset import create_dataloaders


class CTCTrainer:
    """Trainer class for BiLSTM-CTC model."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directories
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.log_dir = Path(config['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard
        self.writer = SummaryWriter(self.log_dir)

        # Create model
        print(f"\nCreating BiLSTM model...")
        self.model = create_bilstm_model(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            vocab_size=config['vocab_size'],
            dropout=config['dropout'],
            device=str(self.device)
        )

        # Create dataloaders
        print(f"\nLoading datasets...")
        self.train_loader, self.dev_loader, self.test_loader = create_dataloaders(
            data_root=config['data_root'],
            features_root=config['features_root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            max_sequence_length=config['max_sequence_length']
        )

        # CTC Loss (blank token index = 1)
        self.criterion = nn.CTCLoss(blank=1, zero_infinity=True, reduction='mean')

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config['scheduler_patience'],
            # verbose=True
        )

        # Training state
        self.start_epoch = 0
        self.best_dev_loss = float('inf')
        self.patience_counter = 0

        # Metrics tracking
        self.train_metrics_history = []
        self.dev_metrics_history = []

        print(f"\nTraining configuration:")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Max epochs: {config['num_epochs']}")
        print(f"  Early stopping patience: {config['early_stop_patience']}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Dev batches: {len(self.dev_loader)}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print()

    def compute_gradient_metrics(self):
        """Compute gradient statistics for monitoring."""
        total_norm = 0.0
        grad_norms = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                grad_norms.append(param_norm)
                total_norm += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        return {
            'total_norm': total_norm,
            'mean_norm': np.mean(grad_norms) if grad_norms else 0.0,
            'max_norm': np.max(grad_norms) if grad_norms else 0.0,
            'min_norm': np.min(grad_norms) if grad_norms else 0.0,
        }

    def compute_weight_metrics(self):
        """Compute model weight statistics."""
        weight_norms = []
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                weight_norms.append(param.data.norm(2).item())
        
        return {
            'mean_weight_norm': np.mean(weight_norms) if weight_norms else 0.0,
            'max_weight_norm': np.max(weight_norms) if weight_norms else 0.0,
        }

    def compute_frame_accuracy(self, log_probs, targets, target_lengths):
        """Compute frame-level accuracy (greedy decoding)."""
        # Greedy decode: argmax along vocab dimension
        predictions = log_probs.argmax(dim=-1)  # (T, N)
        predictions = predictions.transpose(0, 1)  # (N, T)
        
        correct = 0
        total = 0
        
        for i in range(predictions.size(0)):
            pred_seq = predictions[i].cpu().numpy()
            target_seq = targets[i, :target_lengths[i]].cpu().numpy()
            
            # Remove blank tokens (index 1) and consecutive duplicates
            pred_collapsed = []
            prev = None
            for p in pred_seq:
                if p != 1 and p != prev:  # Skip blank and duplicates
                    pred_collapsed.append(p)
                    prev = p
            
            # Compare (simple approximation)
            min_len = min(len(pred_collapsed), len(target_seq))
            if min_len > 0:
                correct += sum(p == t for p, t in zip(pred_collapsed[:min_len], target_seq[:min_len]))
                total += len(target_seq)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def train_epoch(self, epoch):
        """Train for one epoch with comprehensive metrics."""
        self.model.train()
        
        # Metrics accumulators
        total_loss = 0
        loss_list = []
        frame_accuracies = []
        batch_times = []
        
        epoch_start_time = time.time()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
            
            # Move batch to device
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            feature_lengths = batch['feature_lengths'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)

            # Forward pass
            log_probs, output_lengths = self.model(features, feature_lengths)

            # CTC loss
            loss = self.criterion(
                log_probs,           # (T, N, C)
                targets,             # (N, S)
                output_lengths,      # (N,)
                target_lengths       # (N,)
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Compute gradient metrics BEFORE clipping
            grad_metrics = self.compute_gradient_metrics()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )

            self.optimizer.step()

            # Compute frame accuracy
            with torch.no_grad():
                frame_acc = self.compute_frame_accuracy(log_probs, targets, target_lengths)
                frame_accuracies.append(frame_acc)

            # Update metrics
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            loss_item = loss.item()
            total_loss += loss_item
            loss_list.append(loss_item)
            avg_loss = total_loss / (batch_idx + 1)

            # Compute throughput
            samples_per_sec = features.size(0) / batch_time
            tokens_per_sec = feature_lengths.sum().item() / batch_time

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{frame_acc:.3f}',
                'smp/s': f'{samples_per_sec:.1f}'
            })

            # Log to TensorBoard (step-level)
            global_step = epoch * len(self.train_loader) + batch_idx
            
            if batch_idx % 10 == 0:  # Log every 10 steps to reduce overhead
                self.writer.add_scalar('train/loss_step', loss_item, global_step)
                self.writer.add_scalar('train/frame_accuracy_step', frame_acc, global_step)
                self.writer.add_scalar('train/samples_per_sec', samples_per_sec, global_step)
                self.writer.add_scalar('train/tokens_per_sec', tokens_per_sec, global_step)
                
                # Gradient metrics
                self.writer.add_scalar('train/gradient_norm', grad_metrics['total_norm'], global_step)
                self.writer.add_scalar('train/gradient_norm_mean', grad_metrics['mean_norm'], global_step)
                self.writer.add_scalar('train/gradient_norm_max', grad_metrics['max_norm'], global_step)

        # Epoch-level metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = np.mean(loss_list)
        avg_frame_acc = np.mean(frame_accuracies)
        
        # Weight metrics
        weight_metrics = self.compute_weight_metrics()
        
        epoch_metrics = {
            'loss_mean': avg_loss,
            'loss_std': np.std(loss_list),
            'loss_min': np.min(loss_list),
            'loss_max': np.max(loss_list),
            'frame_accuracy': avg_frame_acc,
            'epoch_time': epoch_time,
            'avg_batch_time': np.mean(batch_times),
            'weight_norm_mean': weight_metrics['mean_weight_norm'],
            'weight_norm_max': weight_metrics['max_weight_norm'],
        }
        
        return epoch_metrics

    def validate(self, epoch):
        """Validate on dev set with comprehensive metrics."""
        self.model.eval()
        
        # Metrics accumulators
        total_loss = 0
        loss_list = []
        frame_accuracies = []
        inference_times = []

        with torch.no_grad():
            for batch in tqdm(self.dev_loader, desc="Validating"):
                start_time = time.time()
                
                # Move batch to device
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)
                feature_lengths = batch['feature_lengths'].to(self.device)
                target_lengths = batch['target_lengths'].to(self.device)

                # Forward pass
                log_probs, output_lengths = self.model(features, feature_lengths)

                # CTC loss
                loss = self.criterion(
                    log_probs,
                    targets,
                    output_lengths,
                    target_lengths
                )

                # Frame accuracy
                frame_acc = self.compute_frame_accuracy(log_probs, targets, target_lengths)
                frame_accuracies.append(frame_acc)

                # Timing
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                loss_item = loss.item()
                total_loss += loss_item
                loss_list.append(loss_item)

        # Compute metrics
        avg_loss = np.mean(loss_list)
        avg_frame_acc = np.mean(frame_accuracies)
        avg_inference_time = np.mean(inference_times)
        
        # Calculate FPS
        total_frames = sum(len(batch['features']) for batch in self.dev_loader)
        total_time = sum(inference_times)
        fps = total_frames / total_time if total_time > 0 else 0
        
        val_metrics = {
            'loss_mean': avg_loss,
            'loss_std': np.std(loss_list),
            'frame_accuracy': avg_frame_acc,
            'inference_time': avg_inference_time,
            'fps': fps,
        }
        
        return val_metrics

    def save_checkpoint(self, epoch, dev_metrics, is_best=False):
        """Save model checkpoint with metrics."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'dev_metrics': dev_metrics,
            'config': self.config,
            'train_metrics_history': self.train_metrics_history,
            'dev_metrics_history': self.dev_metrics_history,
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'epoch_{epoch+1:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_dev_loss = checkpoint['dev_metrics']['loss_mean']
        self.train_metrics_history = checkpoint.get('train_metrics_history', [])
        self.dev_metrics_history = checkpoint.get('dev_metrics_history', [])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def train(self):
        """Main training loop with comprehensive logging."""
        print(f"\nStarting training...")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print(f"{'-'*60}")

            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_metrics_history.append(train_metrics)

            # Validate
            dev_metrics = self.validate(epoch)
            self.dev_metrics_history.append(dev_metrics)

            # Log all metrics to TensorBoard
            # Training metrics
            self.writer.add_scalar('epoch/train_loss_mean', train_metrics['loss_mean'], epoch)
            self.writer.add_scalar('epoch/train_loss_std', train_metrics['loss_std'], epoch)
            self.writer.add_scalar('epoch/train_frame_accuracy', train_metrics['frame_accuracy'], epoch)
            self.writer.add_scalar('epoch/train_time', train_metrics['epoch_time'], epoch)
            self.writer.add_scalar('epoch/weight_norm_mean', train_metrics['weight_norm_mean'], epoch)
            
            # Validation metrics
            self.writer.add_scalar('epoch/dev_loss_mean', dev_metrics['loss_mean'], epoch)
            self.writer.add_scalar('epoch/dev_loss_std', dev_metrics['loss_std'], epoch)
            self.writer.add_scalar('epoch/dev_frame_accuracy', dev_metrics['frame_accuracy'], epoch)
            self.writer.add_scalar('epoch/dev_fps', dev_metrics['fps'], epoch)
            
            # Learning rate
            self.writer.add_scalar('epoch/learning_rate', 
                                 self.optimizer.param_groups[0]['lr'], epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss:     {train_metrics['loss_mean']:.4f} ± {train_metrics['loss_std']:.4f}")
            print(f"  Train Frame Acc: {train_metrics['frame_accuracy']:.3f}")
            print(f"  Dev Loss:       {dev_metrics['loss_mean']:.4f} ± {dev_metrics['loss_std']:.4f}")
            print(f"  Dev Frame Acc:   {dev_metrics['frame_accuracy']:.3f}")
            print(f"  Dev FPS:        {dev_metrics['fps']:.1f}")
            print(f"  LR:             {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Epoch Time:     {train_metrics['epoch_time']:.1f}s")

            # Learning rate scheduler
            self.scheduler.step(dev_metrics['loss_mean'])

            # Check if best model
            is_best = dev_metrics['loss_mean'] < self.best_dev_loss
            if is_best:
                print(f"  *** New best dev loss: {dev_metrics['loss_mean']:.4f} ***")
                self.best_dev_loss = dev_metrics['loss_mean']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            if (epoch + 1) % self.config['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, dev_metrics, is_best=is_best)

            # Early stopping
            if self.patience_counter >= self.config['early_stop_patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best dev loss: {self.best_dev_loss:.4f}")
                break

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best dev loss: {self.best_dev_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")

        # Save final metrics
        metrics_file = self.checkpoint_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump({
                'train_metrics': self.train_metrics_history,
                'dev_metrics': self.dev_metrics_history,
            }, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train BiLSTM-CTC baseline')

    # Model architecture
    parser.add_argument('--input-dim', type=int, default=177)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--vocab-size', type=int, default=1120)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Data
    parser.add_argument('--data-root', type=str,
                        default='data/raw_data/phoenix-2014-signerindependent-SI5')
    parser.add_argument('--features-root', type=str, default='data/processed')
    parser.add_argument('--max-sequence-length', type=int, default=241)

    # Training
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--num-workers', type=int, default=4)

    # Scheduler
    parser.add_argument('--scheduler-patience', type=int, default=3)

    # Early stopping
    parser.add_argument('--early-stop-patience', type=int, default=5)

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str,
                        default='models/bilstm_baseline')
    parser.add_argument('--log-dir', type=str,
                        default='logs/bilstm_baseline')
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Convert to config dict
    config = vars(args)

    # Print configuration
    print("\n" + "="*60)
    print("BiLSTM-CTC Baseline Training")
    print("="*60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Save configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = Path(config['checkpoint_dir']) / f'config_{timestamp}.json'
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to: {config_file}")

    # Create trainer
    trainer = CTCTrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()