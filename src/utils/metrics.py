"""
Evaluation metrics for sign language recognition.
Implements WER (Word Error Rate) and SER (Sign Error Rate).
"""

import numpy as np
from typing import List, Tuple
import editdistance


def calculate_wer(references: List[List[int]], hypotheses: List[List[int]]) -> Tuple[float, dict]:
    """
    Calculate Word Error Rate (WER).

    WER = (Substitutions + Deletions + Insertions) / Total_Reference_Words

    Args:
        references: List of reference sequences (ground truth)
        hypotheses: List of hypothesis sequences (predictions)

    Returns:
        (wer, detailed_stats)
    """
    assert len(references) == len(hypotheses), "References and hypotheses must have same length"

    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_reference_length = 0
    total_distance = 0

    for ref, hyp in zip(references, hypotheses):
        # Calculate edit distance
        distance = editdistance.eval(ref, hyp)
        total_distance += distance
        total_reference_length += len(ref)

        # Detailed error counts (approximation using DP)
        # For exact counts, need to backtrack through DP matrix
        ref_len = len(ref)
        hyp_len = len(hyp)

        # Simple approximation
        if hyp_len > ref_len:
            total_insertions += (hyp_len - ref_len)
        elif hyp_len < ref_len:
            total_deletions += (ref_len - hyp_len)

        # Remaining errors are substitutions
        total_substitutions += distance - abs(ref_len - hyp_len)

    wer = (total_distance / total_reference_length * 100) if total_reference_length > 0 else 0.0

    stats = {
        'wer': wer,
        'total_errors': total_distance,
        'substitutions': total_substitutions,
        'deletions': total_deletions,
        'insertions': total_insertions,
        'total_words': total_reference_length,
        'num_sequences': len(references),
    }

    return wer, stats


def calculate_ser(references: List[List[int]], hypotheses: List[List[int]]) -> Tuple[float, dict]:
    """
    Calculate Sign Error Rate (SER) - sentence-level accuracy.

    SER = Number of incorrect sequences / Total sequences

    Args:
        references: List of reference sequences
        hypotheses: List of hypothesis sequences

    Returns:
        (ser, detailed_stats)
    """
    assert len(references) == len(hypotheses)

    correct = 0
    total = len(references)

    for ref, hyp in zip(references, hypotheses):
        if ref == hyp:
            correct += 1

    ser = ((total - correct) / total * 100) if total > 0 else 0.0
    accuracy = (correct / total * 100) if total > 0 else 0.0

    stats = {
        'ser': ser,
        'sentence_accuracy': accuracy,
        'correct_sequences': correct,
        'total_sequences': total,
        'incorrect_sequences': total - correct,
    }

    return ser, stats


def calculate_per_class_accuracy(
    references: List[List[int]],
    hypotheses: List[List[int]],
    num_classes: int
) -> dict:
    """
    Calculate per-class recognition accuracy.

    Args:
        references: List of reference sequences
        hypotheses: List of hypothesis sequences
        num_classes: Total number of classes

    Returns:
        Dictionary with per-class statistics
    """
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for ref, hyp in zip(references, hypotheses):
        # Count ground truth occurrences
        for token in ref:
            if token < num_classes:
                class_total[token] += 1

        # Count correct predictions
        for r_token, h_token in zip(ref, hyp):
            if r_token == h_token and r_token < num_classes:
                class_correct[r_token] += 1

    # Calculate per-class accuracy
    class_accuracy = np.divide(
        class_correct,
        class_total,
        out=np.zeros_like(class_correct),
        where=class_total > 0
    )

    return {
        'class_accuracy': class_accuracy,
        'class_correct': class_correct,
        'class_total': class_total,
        'mean_class_accuracy': np.mean(class_accuracy[class_total > 0]) * 100 if np.any(class_total > 0) else 0.0,
    }


def format_metrics_report(wer_stats: dict, ser_stats: dict, split_name: str = "Test") -> str:
    """
    Format metrics into a readable report.

    Args:
        wer_stats: WER statistics dictionary
        ser_stats: SER statistics dictionary
        split_name: Name of the dataset split

    Returns:
        Formatted string report
    """
    report = f"\n{'='*60}\n"
    report += f"{split_name} Set Evaluation Results\n"
    report += f"{'='*60}\n\n"

    report += f"Word Error Rate (WER):\n"
    report += f"  WER: {wer_stats['wer']:.2f}%\n"
    report += f"  Total Errors: {wer_stats['total_errors']}\n"
    report += f"    - Substitutions: {wer_stats['substitutions']}\n"
    report += f"    - Deletions: {wer_stats['deletions']}\n"
    report += f"    - Insertions: {wer_stats['insertions']}\n"
    report += f"  Total Words: {wer_stats['total_words']}\n"
    report += f"  Sequences: {wer_stats['num_sequences']}\n\n"

    report += f"Sentence Error Rate (SER):\n"
    report += f"  SER: {ser_stats['ser']:.2f}%\n"
    report += f"  Sentence Accuracy: {ser_stats['sentence_accuracy']:.2f}%\n"
    report += f"  Correct Sequences: {ser_stats['correct_sequences']}/{ser_stats['total_sequences']}\n\n"

    report += f"{'='*60}\n"

    return report