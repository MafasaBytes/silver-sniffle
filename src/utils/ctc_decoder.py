"""
CTC Decoder utilities for sequence decoding.
Implements greedy decoding and beam search for CTC outputs.
"""

import torch
import numpy as np
from typing import List, Tuple


def greedy_decode(log_probs: torch.Tensor, blank_id: int = 1) -> List[int]:
    """
    Greedy CTC decoding - argmax at each timestep.

    Args:
        log_probs: (T, vocab_size) log probabilities
        blank_id: Index of CTC blank token

    Returns:
        Decoded sequence (list of token IDs with blanks/duplicates removed)
    """
    # Get argmax predictions
    predictions = log_probs.argmax(dim=-1).cpu().numpy()  # (T,)

    # Collapse repeated tokens and remove blanks
    decoded = []
    prev = None
    for pred in predictions:
        if pred != blank_id and pred != prev:
            decoded.append(int(pred))
        prev = pred

    return decoded


def greedy_decode_batch(log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int = 1) -> List[List[int]]:
    """
    Batch greedy decoding.

    Args:
        log_probs: (T, N, vocab_size) log probabilities
        lengths: (N,) sequence lengths
        blank_id: CTC blank token index

    Returns:
        List of decoded sequences for each sample in batch
    """
    T, N, _ = log_probs.shape
    decoded_batch = []

    for i in range(N):
        # Extract single sequence
        seq_len = lengths[i].item()
        seq_log_probs = log_probs[:seq_len, i, :]  # (seq_len, vocab_size)

        # Decode
        decoded = greedy_decode(seq_log_probs, blank_id)
        decoded_batch.append(decoded)

    return decoded_batch


def beam_search_decode(log_probs: torch.Tensor, beam_width: int = 10, blank_id: int = 1) -> Tuple[List[int], float]:
    """
    Beam search CTC decoding.

    Args:
        log_probs: (T, vocab_size) log probabilities
        beam_width: Number of beams to maintain
        blank_id: CTC blank token index

    Returns:
        (decoded_sequence, log_probability)
    """
    T, vocab_size = log_probs.shape
    log_probs = log_probs.cpu().numpy()

    # Initialize beams: (prefix, (p_blank, p_non_blank))
    # prefix = sequence so far, p_blank = prob ending in blank, p_non_blank = prob ending in non-blank
    beams = {
        (): (0.0, float('-inf'))  # Empty prefix: p_blank=1 (log=0), p_non_blank=0 (log=-inf)
    }

    for t in range(T):
        new_beams = {}

        # Expand each beam
        for prefix, (p_blank, p_non_blank) in beams.items():
            # Current total probability
            p_total = np.logaddexp(p_blank, p_non_blank)

            # Extend with blank
            new_p_blank = p_total + log_probs[t, blank_id]
            if prefix not in new_beams:
                new_beams[prefix] = (new_p_blank, float('-inf'))
            else:
                old_p_blank, old_p_non_blank = new_beams[prefix]
                new_beams[prefix] = (np.logaddexp(old_p_blank, new_p_blank), old_p_non_blank)

            # Extend with non-blank tokens
            for c in range(vocab_size):
                if c == blank_id:
                    continue

                new_prefix = prefix + (c,)

                # If extending with same token, must come from blank state
                if len(prefix) > 0 and prefix[-1] == c:
                    new_p_non_blank = p_blank + log_probs[t, c]
                else:
                    # Can come from either blank or non-blank state
                    new_p_non_blank = p_total + log_probs[t, c]

                if new_prefix not in new_beams:
                    new_beams[new_prefix] = (float('-inf'), new_p_non_blank)
                else:
                    old_p_blank, old_p_non_blank = new_beams[new_prefix]
                    new_beams[new_prefix] = (old_p_blank, np.logaddexp(old_p_non_blank, new_p_non_blank))

        # Prune beams - keep top beam_width
        beams = dict(sorted(
            new_beams.items(),
            key=lambda x: np.logaddexp(x[1][0], x[1][1]),
            reverse=True
        )[:beam_width])

    # Get best beam
    best_prefix = max(beams.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1]))
    decoded = list(best_prefix[0])
    log_prob = np.logaddexp(best_prefix[1][0], best_prefix[1][1])

    return decoded, log_prob


def beam_search_decode_batch(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    beam_width: int = 10,
    blank_id: int = 1
) -> List[Tuple[List[int], float]]:
    """
    Batch beam search decoding.

    Args:
        log_probs: (T, N, vocab_size)
        lengths: (N,)
        beam_width: Beam width
        blank_id: CTC blank token

    Returns:
        List of (decoded_sequence, log_prob) for each sample
    """
    T, N, _ = log_probs.shape
    results = []

    for i in range(N):
        seq_len = lengths[i].item()
        seq_log_probs = log_probs[:seq_len, i, :]

        decoded, log_prob = beam_search_decode(seq_log_probs, beam_width, blank_id)
        results.append((decoded, log_prob))

    return results