"""
Evaluation script for BiLSTM-CTC model.
Calculates WER, SER, and other metrics on dev/test sets.
"""

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import time
import numpy as np
from datetime import datetime

from models.bilstm import BiLSTMModel
from phoenix_dataset import create_dataloaders
from utils.ctc_decoder import greedy_decode_batch, beam_search_decode_batch
from utils.metrics import calculate_wer, calculate_ser, format_metrics_report


class ModelEvaluator:
    """Evaluator class for BiLSTM-CTC model."""

    def __init__(self, model, dataloader, dataset, device, decode_method='greedy', beam_width=10):
        """
        Initialize evaluator.

        Args:
            model: Trained BiLSTM model
            dataloader: DataLoader for evaluation
            dataset: Dataset object (for vocabulary access)
            device: torch.device
            decode_method: 'greedy' or 'beam_search'
            beam_width: Beam width for beam search
        """
        self.model = model
        self.dataloader = dataloader
        self.dataset = dataset
        self.device = device
        self.decode_method = decode_method
        self.beam_width = beam_width

        # Get vocabulary
        self.vocab = dataset.vocab
        self.idx_to_sign = {idx: sign for sign, idx in self.vocab.items()}

    def decode_predictions(self, log_probs, lengths):
        """
        Decode CTC outputs to sequences.

        Args:
            log_probs: (T, N, vocab_size) log probabilities
            lengths: (N,) sequence lengths

        Returns:
            List of decoded sequences (list of token IDs)
        """
        if self.decode_method == 'greedy':
            return greedy_decode_batch(log_probs, lengths, blank_id=1)
        elif self.decode_method == 'beam_search':
            results = beam_search_decode_batch(log_probs, lengths, self.beam_width, blank_id=1)
            return [seq for seq, _ in results]
        else:
            raise ValueError(f"Unknown decode method: {self.decode_method}")

    def evaluate(self):
        """
        Run evaluation on the dataset.

        Returns:
            Dictionary with evaluation results
        """
        self.model.eval()

        all_references = []
        all_hypotheses = []
        all_sample_ids = []

        inference_times = []
        total_frames = 0

        print(f"\nEvaluating with {self.decode_method} decoding...")
        print(f"{'='*60}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
                start_time = time.time()

                # Move to device
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)
                feature_lengths = batch['feature_lengths'].to(self.device)
                target_lengths = batch['target_lengths'].to(self.device)
                sample_ids = batch['sample_ids']

                # Forward pass
                log_probs, output_lengths = self.model(features, feature_lengths)

                # Decode predictions
                hypotheses = self.decode_predictions(log_probs, output_lengths)

                # Extract references (remove padding)
                references = []
                for i in range(targets.size(0)):
                    ref = targets[i, :target_lengths[i]].cpu().tolist()
                    references.append(ref)

                # Store results
                all_references.extend(references)
                all_hypotheses.extend(hypotheses)
                all_sample_ids.extend(sample_ids)

                # Timing
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                total_frames += features.size(0)

        # Calculate metrics
        print("\nCalculating metrics...")
        wer, wer_stats = calculate_wer(all_references, all_hypotheses)
        ser, ser_stats = calculate_ser(all_references, all_hypotheses)

        # Timing statistics
        total_time = sum(inference_times)
        avg_time_per_batch = np.mean(inference_times)
        fps = total_frames / total_time if total_time > 0 else 0

        results = {
            'wer': wer,
            'wer_stats': wer_stats,
            'ser': ser,
            'ser_stats': ser_stats,
            'inference': {
                'total_time': total_time,
                'avg_batch_time': avg_time_per_batch,
                'fps': fps,
                'decode_method': self.decode_method,
                'beam_width': self.beam_width if self.decode_method == 'beam_search' else None,
            },
            'predictions': {
                'sample_ids': all_sample_ids,
                'references': all_references,
                'hypotheses': all_hypotheses,
            }
        }

        return results

    def format_sample_predictions(self, results, num_samples=5):
        """
        Format sample predictions for display.

        Args:
            results: Evaluation results dictionary
            num_samples: Number of samples to display

        Returns:
            Formatted string
        """
        output = f"\n{'='*60}\n"
        output += f"Sample Predictions (first {num_samples})\n"
        output += f"{'='*60}\n\n"

        for i in range(min(num_samples, len(results['predictions']['sample_ids']))):
            sample_id = results['predictions']['sample_ids'][i]
            reference = results['predictions']['references'][i]
            hypothesis = results['predictions']['hypotheses'][i]

            # Decode to text
            ref_text = self.decode_sequence_to_text(reference)
            hyp_text = self.decode_sequence_to_text(hypothesis)

            output += f"Sample {i+1}: {sample_id}\n"
            output += f"  Reference:  {ref_text}\n"
            output += f"  Hypothesis: {hyp_text}\n"
            output += f"  Match: {'✓' if reference == hypothesis else '✗'}\n\n"

        return output

    def decode_sequence_to_text(self, sequence):
        """
        Decode token IDs to text.

        Args:
            sequence: List of token IDs

        Returns:
            Space-separated string of signs
        """
        tokens = []
        for idx in sequence:
            if idx in self.idx_to_sign:
                sign = self.idx_to_sign[idx]
                # Skip special tokens
                if sign not in ['<PAD>', '<BLANK>', '<UNK>']:
                    tokens.append(sign)
            else:
                tokens.append(f"<UNK_{idx}>")

        return ' '.join(tokens)


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: torch.device

    Returns:
        Loaded model and config
    """
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint['config']

    # Create model
    model = BiLSTMModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        vocab_size=config['vocab_size'],
        dropout=config['dropout']
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best dev loss: {checkpoint.get('dev_metrics', {}).get('loss_mean', 'N/A')}")

    return model, config


def save_results(results, output_dir, split_name, decode_method):
    """
    Save evaluation results to JSON file.

    Args:
        results: Evaluation results dictionary
        output_dir: Output directory
        split_name: Dataset split name
        decode_method: Decoding method used
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results (without predictions to keep file size small)
    results_file = output_dir / f'{split_name}_{decode_method}_results.json'

    # Create summary (without full predictions)
    summary = {
        'split': split_name,
        'decode_method': decode_method,
        'wer': results['wer'],
        'wer_stats': results['wer_stats'],
        'ser': results['ser'],
        'ser_stats': results['ser_stats'],
        'inference': results['inference'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {results_file}")

    # Save detailed predictions separately
    predictions_file = output_dir / f'{split_name}_{decode_method}_predictions.json'
    with open(predictions_file, 'w') as f:
        json.dump(results['predictions'], f, indent=2)

    print(f"Predictions saved to: {predictions_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate BiLSTM-CTC model')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')

    # Data
    parser.add_argument('--data-root', type=str,
                       default='data/raw_data/phoenix-2014-signerindependent-SI5')
    parser.add_argument('--features-root', type=str, default='data/processed')
    parser.add_argument('--split', type=str, default='test',
                       choices=['dev', 'test'], help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)

    # Decoding
    parser.add_argument('--decode-method', type=str, default='greedy',
                       choices=['greedy', 'beam_search'],
                       help='CTC decoding method')
    parser.add_argument('--beam-width', type=int, default=10,
                       help='Beam width for beam search')

    # Output
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Directory to save results')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of sample predictions to display')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Load data
    print(f"\nLoading {args.split} dataset...")
    _, dev_loader, test_loader = create_dataloaders(
        data_root=args.data_root,
        features_root=args.features_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_sequence_length=config.get('max_sequence_length', None)
    )

    # Select dataloader and dataset
    if args.split == 'dev':
        dataloader = dev_loader
        dataset = dev_loader.dataset
    else:
        dataloader = test_loader
        dataset = test_loader.dataset

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Vocabulary size: {len(dataset.vocab)}")

    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        device=device,
        decode_method=args.decode_method,
        beam_width=args.beam_width
    )

    # Run evaluation
    results = evaluator.evaluate()

    # Print results
    print(format_metrics_report(
        results['wer_stats'],
        results['ser_stats'],
        split_name=args.split.capitalize()
    ))

    print(f"\nInference Statistics:")
    print(f"  Total time: {results['inference']['total_time']:.2f}s")
    print(f"  Avg batch time: {results['inference']['avg_batch_time']:.4f}s")
    print(f"  FPS: {results['inference']['fps']:.1f}")
    print(f"  Decode method: {results['inference']['decode_method']}")
    if results['inference']['beam_width']:
        print(f"  Beam width: {results['inference']['beam_width']}")

    # Print sample predictions
    print(evaluator.format_sample_predictions(results, args.num_samples))

    # Save results
    save_results(results, args.output_dir, args.split, args.decode_method)

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()