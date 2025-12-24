import torch
import torch.nn as nn


class ExemplarModule(nn.Module):
    """
    Standard encoder block using PyTorch 2.0+ conventions.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        dropout_rate (float): Dropout probability.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Seq, Dim].

        Returns:
            torch.Tensor: Output tensor of shape [Batch, Seq, Dim].
        """
        # Shape assertion for debugging/documentation
        assert x.dim() == 3, f"Expected 3D input [B, S, D], got {x.shape}"

        # Residual connection
        return self.norm(x + self.net(x))
