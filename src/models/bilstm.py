"""
BiLSTM-CTC model for sign language recognition (Phase I baseline).

Architecture follows Koller et al. (2015) approach with handcrafted features.
Target WER: 35-45% baseline performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM with CTC loss for continuous sign language recognition.

    Args:
        input_dim: Input feature dimensionality (default: 177)
        hidden_dim: LSTM hidden state size (default: 256)
        num_layers: Number of stacked BiLSTM layers (default: 2)
        vocab_size: Output vocabulary size (default: 1120)
        dropout: Dropout probability (default: 0.3)
    """

    def __init__(
        self, input_dim: int = 177, hidden_dim: int = 256, num_layers: int = 2, vocab_size: int = 1120, dropout: float = 0.3):
        super(BiLSTMModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.dropout = dropout

        # Bidirectional LSTM (output is 2*hidden_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
          )

        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/orthogonal initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'fc.weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through BiLSTM-CTC model.

        Args:
            x: (batch_size, max_seq_len, input_dim)
            lengths: (batch_size,) actual sequence lengths

        Returns:
            log_probs: (max_seq_len, batch_size, vocab_size) for CTC
            output_lengths: (batch_size,) output sequence lengths
        """
        batch_size, max_seq_len, _ = x.shape

        # Pack padded sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # BiLSTM forward
        packed_output, (h_n, c_n) = self.lstm(packed_input)

        # Unpack
        lstm_out, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=max_seq_len
        )

        # Dropout and projection
        lstm_out = self.dropout_layer(lstm_out)
        logits = self.fc(lstm_out)
        log_probs = F.log_softmax(logits, dim=-1)

        # Transpose to time-first for CTC
        log_probs = log_probs.transpose(0, 1).contiguous()

        return log_probs, output_lengths

    def predict(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference with greedy decoding."""
        self.eval()
        with torch.no_grad():
            log_probs, output_lengths = self.forward(x, lengths)
            predictions = log_probs.argmax(dim=-1).transpose(0, 1)
            scores = log_probs.max(dim=-1)[0].sum(dim=0)
        return predictions, scores

    def get_num_params(self) -> dict:
        """Calculate parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}

    def get_model_size_mb(self) -> float:
        """Estimate model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


def create_bilstm_model(
    input_dim: int = 177,
    hidden_dim: int = 256,
    num_layers: int = 2,
    vocab_size: int = 1120,
    dropout: float = 0.3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> BiLSTMModel:
      """Factory function to create BiLSTM model."""
      model = BiLSTMModel(input_dim, hidden_dim, num_layers, vocab_size, dropout)
      model = model.to(device)

      params = model.get_num_params()
      size_mb = model.get_model_size_mb()

      print(f"\n{'='*60}")
      print("BiLSTM Model Summary")
      print(f"{'='*60}")
      print(f"Architecture:")
      print(f"  Input dim:     {input_dim}")
      print(f"  Hidden dim:    {hidden_dim}")
      print(f"  Num layers:    {num_layers}")
      print(f"  Vocab size:    {vocab_size}")
      print(f"  Dropout:       {dropout}")
      print(f"  Bidirectional: True")
      print(f"\nParameters:")
      print(f"  Total:         {params['total']:,}")
      print(f"  Trainable:     {params['trainable']:,}")
      print(f"\nModel size:      {size_mb:.2f} MB")
      print(f"Device:          {device}")
      print(f"{'='*60}\n")

      return model


if __name__ == "__main__":
    print("Testing BiLSTM model...")

    model = create_bilstm_model()
    x = torch.randn(4, 150, 177)
    lengths = torch.tensor([150, 120, 100, 80])

    device = next(model.parameters()).device
    x, lengths = x.to(device), lengths.to(device)

    log_probs, output_lengths = model(x, lengths)
    print(f"Output shape: {log_probs.shape}")
    print("BiLSTM model test passed!")
