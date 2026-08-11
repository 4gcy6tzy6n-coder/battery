"""TriStateLite joint probabilistic battery network.

A GRU encoder reads the causal fast window (left-padded; valid rows are
extracted and right-packed so padding never corrupts the recurrence), a
mask-aware mean pool summarizes the slow completed-cycle history, and a fused
representation feeds three ordered quantile heads (SOC, SOH bounded; log-TTE
non-negative). ``head_dropout`` makes inference-time MC sampling meaningful.
"""

from __future__ import annotations

import torch
from torch import nn

from tristatelite.models.quantile_heads import BoundedQuantileHead, PositiveQuantileHead


class TCNEncoder(nn.Module):
    """Causal dilated-convolution encoder over the fast window.

    Dilations grow by powers of two, so the final position's receptive field
    covers the whole window with O(log L) layers. Faster than a GRU on CPU.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        blocks: list[nn.Module] = []
        for layer_index in range(num_layers):
            dilation = 2**layer_index
            in_channels = input_dim if layer_index == 0 else hidden_dim
            conv = nn.Conv1d(
                in_channels,
                hidden_dim,
                kernel_size=3,
                padding=dilation,  # causal: pad only the left via padding=dilation
                dilation=dilation,
            )
            blocks.append(
                nn.Sequential(conv, nn.ReLU(), nn.Dropout(dropout))
            )
        self.net = nn.Sequential(*blocks)

    def forward(self, fast_x: torch.Tensor) -> torch.Tensor:
        # fast_x: [B, L, F] -> [B, F, L]
        x = fast_x.transpose(1, 2)
        x = self.net(x)
        # Anchor representation: the final timestep of the causal stack.
        return x[:, :, -1]


class PointHead(nn.Module):
    """Single-point output head for the deterministic MSE baseline."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)


class UnorderedQuantileHead(nn.Module):
    """Independent per-level quantile outputs with no ordering guarantee.

    Used by the ``unordered`` baseline; predictions are sorted post-hoc at
    evaluation, mirroring common non-crossing-constrained quantile regression.
    """

    def __init__(self, in_dim: int, num_quantiles: int) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if num_quantiles < 1:
            raise ValueError("num_quantiles must be positive")
        self.linear = nn.Linear(in_dim, num_quantiles)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)


class TriStateLiteNet(nn.Module):
    """GRU encoder + slow-history fusion + three ordered quantile heads."""

    def __init__(
        self,
        fast_input_dim: int,
        slow_input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_quantiles: int = 3,
        head_mode: str = "ordered",
        encoder_type: str = "gru",
    ) -> None:
        super().__init__()
        if fast_input_dim <= 0 or slow_input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input and hidden dimensions must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if head_mode not in ("ordered", "unordered", "point"):
            raise ValueError("head_mode must be ordered, unordered, or point")
        if encoder_type not in ("gru", "tcn"):
            raise ValueError("encoder_type must be 'gru' or 'tcn'")
        self.num_quantiles = num_quantiles
        self.head_mode = head_mode
        self.encoder_type = encoder_type
        self.requires_quantile_sort = head_mode == "unordered"
        self.hidden_dim = hidden_dim
        if encoder_type == "gru":
            # GRU dropout only applies between stacked layers.
            self.gru = nn.GRU(
                fast_input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.gru = TCNEncoder(fast_input_dim, hidden_dim, num_layers, dropout)
        self.fusion = nn.Linear(hidden_dim + slow_input_dim, hidden_dim)
        self.head_dropout = nn.Dropout(dropout)
        if head_mode == "ordered":
            self.soc_head: nn.Module = BoundedQuantileHead(
                hidden_dim, num_quantiles=num_quantiles
            )
            self.soh_head: nn.Module = BoundedQuantileHead(
                hidden_dim, num_quantiles=num_quantiles
            )
            self.tte_head: nn.Module = PositiveQuantileHead(
                hidden_dim, num_quantiles=num_quantiles
            )
        elif head_mode == "unordered":
            self.soc_head = UnorderedQuantileHead(hidden_dim, num_quantiles)
            self.soh_head = UnorderedQuantileHead(hidden_dim, num_quantiles)
            self.tte_head = UnorderedQuantileHead(hidden_dim, num_quantiles)
        else:
            self.soc_head = PointHead(hidden_dim)
            self.soh_head = PointHead(hidden_dim)
            self.tte_head = PointHead(hidden_dim)

    def _encode_fast(
        self, fast_x: torch.Tensor, fast_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return the anchor representation, ignoring left padding."""
        if self.encoder_type == "tcn":
            return self.gru(fast_x)
        batch, length, dim = fast_x.shape
        lengths = fast_mask.sum(dim=1).clamp(min=1)
        # Reorder each sample so valid rows are contiguous at the front,
        # right-padding with zeros, then pack so the recurrence sees only truth.
        reordered = fast_x.new_zeros(batch, length, dim)
        for index in range(batch):
            count = int(lengths[index])
            reordered[index, :count] = fast_x[index, fast_mask[index]]
        packed = nn.utils.rnn.pack_padded_sequence(
            reordered, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return hidden[-1]

    def _pool_slow(
        self, slow_x: torch.Tensor, slow_mask: torch.Tensor
    ) -> torch.Tensor:
        weights = slow_mask.to(slow_x.dtype)
        valid = weights.sum(dim=1, keepdim=True).clamp(min=1)
        return (slow_x * weights.unsqueeze(-1)).sum(dim=1) / valid

    def forward(
        self,
        fast_x: torch.Tensor,
        fast_mask: torch.Tensor,
        slow_x: torch.Tensor,
        slow_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_state = self._encode_fast(fast_x, fast_mask)
        slow_pooled = self._pool_slow(slow_x, slow_mask)
        fused = torch.tanh(self.fusion(torch.cat([anchor_state, slow_pooled], dim=-1)))
        fused = self.head_dropout(fused)
        return self.soc_head(fused), self.soh_head(fused), self.tte_head(fused)
