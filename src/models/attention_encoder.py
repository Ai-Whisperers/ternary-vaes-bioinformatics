"""Attention-based Encoder for 9-Operation Ternary Sequences.

Phase 3.2 Enhancement - Self-Attention Over Ternary Operations
=============================================================

The standard ImprovedEncoder treats 9 ternary operations as a flat vector.
The AttentionEncoder treats them as a sequence and uses self-attention
to capture relationships between operation positions.

Key capabilities:
1. **Position-aware encoding**: Each of 9 positions gets position embeddings
2. **Self-attention**: Operations attend to each other to learn dependencies
3. **Multi-head attention**: Different heads capture different relationship types
4. **Hierarchical processing**: Multiple attention layers with residual connections
5. **Backward compatibility**: Can replace ImprovedEncoder with same interface

Architecture:
```
Input: (batch, 9) ternary operations {-1, 0, 1}
↓
Position Embeddings + Operation Embeddings
↓
Multi-layer Self-Attention (with residual connections)
↓
Global Pooling (attention-weighted sum)
↓
MLP → (mu, logvar)
```

Expected improvements:
- Better capture of operation dependencies
- Position-aware understanding
- More expressive latent representations
- +0.10 hierarchy correlation potential

Author: Claude Code
Date: 2026-01-14
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import MLPBuilder for standardized architecture
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.nn_factory import create_encoder_mlp


class PositionalEncoding(nn.Module):
    """Positional encoding for 9-operation sequences."""

    def __init__(self, d_model: int, max_len: int = 9):
        super().__init__()
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Use sinusoidal encoding (like Transformer)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of module state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Position-encoded tensor
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class OperationEmbedding(nn.Module):
    """Embedding layer for ternary operations that preserves gradients."""

    def __init__(self, d_model: int):
        super().__init__()
        # Instead of discrete embedding, use learnable transformation
        # This preserves gradient flow through the input values
        self.input_transform = nn.Linear(1, d_model)

        # Additional non-linear transformation for expressiveness
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Layer norm for stable embeddings
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert ternary operations to embeddings.

        Args:
            x: Ternary operations (batch, 9) with values {-1, 0, 1}

        Returns:
            Embedded operations (batch, 9, d_model)
        """
        # Expand last dimension for linear transformation: (batch, 9) → (batch, 9, 1)
        x_expanded = x.unsqueeze(-1)

        # Linear transformation preserves gradients: (batch, 9, 1) → (batch, 9, d_model)
        embedded = self.input_transform(x_expanded)

        # Additional transformation for expressiveness
        embedded = self.transform(embedded)

        # Normalize
        return self.norm(embedded)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for operation sequences."""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Multi-head self-attention.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Attended tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()

        # Linear projections in batch from d_model => h * d_k
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        attended = self._scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and put through final linear layer
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )

        return self.w_o(attended)

    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute scaled dot-product attention."""
        d_k = Q.size(-1)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        return torch.matmul(attn_weights, V)


class AttentionBlock(nn.Module):
    """Transformer-style attention block with residual connections."""

    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = None, dropout: float = 0.1):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model  # Standard transformer ratio

        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),  # Using SiLU for consistency with other components
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class AttentionPooling(nn.Module):
    """Attention-based global pooling for sequence-to-vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attention_weights = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool sequence using learned attention weights.

        Args:
            x: Input sequence (batch, seq_len, d_model)
            mask: Optional mask for sequence positions

        Returns:
            Pooled vector (batch, d_model)
        """
        # Compute attention weights
        attn_scores = self.attention_weights(x).squeeze(-1)  # (batch, seq_len)

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)  # (batch, d_model)

        return pooled


class AttentionEncoder(nn.Module):
    """Attention-based encoder for 9-operation ternary sequences.

    Replaces flat MLP encoder with attention-based sequence processing
    to better capture dependencies between operation positions.
    """

    def __init__(
        self,
        input_dim: int = 9,  # 9 ternary operations
        latent_dim: int = 16,
        d_model: int = 128,  # Model dimension for attention
        num_heads: int = 8,  # Number of attention heads
        num_layers: int = 3,  # Number of attention layers
        dropout: float = 0.1,
        logvar_min: float = -10.0,
        logvar_max: float = 2.0,
    ):
        """Initialize attention-based encoder.

        Args:
            input_dim: Input dimension (should be 9 for ternary operations)
            latent_dim: Latent space dimension
            d_model: Model dimension for attention layers
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            dropout: Dropout probability
            logvar_min: Minimum logvar clamp
            logvar_max: Maximum logvar clamp
        """
        super().__init__()

        assert input_dim == 9, f"AttentionEncoder designed for 9 operations, got {input_dim}"

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        # Operation embedding and positional encoding
        self.operation_embedding = OperationEmbedding(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=9)

        # Multi-layer self-attention
        self.attention_layers = nn.ModuleList([
            AttentionBlock(d_model, num_heads, d_ff=4*d_model, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Global pooling
        self.pooling = AttentionPooling(d_model)

        # Final projection to latent space (using MLPBuilder)
        self.projection = create_encoder_mlp(
            input_dim=d_model,
            hidden_dims=[d_model//2],
            latent_dim=d_model//2,
            dropout=dropout,
        )

        # Latent distribution heads
        self.fc_mu = nn.Linear(d_model//2, latent_dim)
        self.fc_logvar = nn.Linear(d_model//2, latent_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through attention encoder.

        Args:
            x: Ternary operations (batch, 9) with values in {-1, 0, 1}

        Returns:
            Tuple of (mu, logvar) tensors with logvar clamped
        """
        batch_size = x.size(0)

        # Convert to embeddings: (batch, 9) → (batch, 9, d_model)
        x_embedded = self.operation_embedding(x)

        # Add positional encoding
        x_pos = self.positional_encoding(x_embedded)

        # Apply attention layers
        for attention_layer in self.attention_layers:
            x_pos = attention_layer(x_pos)

        # Global pooling: (batch, 9, d_model) → (batch, d_model)
        x_pooled = self.pooling(x_pos)

        # Project to latent space
        x_proj = self.projection(x_pooled)

        # Compute mu and logvar
        mu = self.fc_mu(x_proj)
        logvar = self.fc_logvar(x_proj)

        # Clamp logvar for stability
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)

        return mu, logvar

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability.

        Args:
            x: Ternary operations (batch, 9)

        Returns:
            Attention weights from final pooling layer (batch, 9)
        """
        batch_size = x.size(0)

        # Forward pass through embedding and attention
        x_embedded = self.operation_embedding(x)
        x_pos = self.positional_encoding(x_embedded)

        for attention_layer in self.attention_layers:
            x_pos = attention_layer(x_pos)

        # Get attention weights from pooling
        attn_scores = self.pooling.attention_weights(x_pos).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)

        return attn_weights


class HybridAttentionEncoder(nn.Module):
    """Hybrid encoder that combines attention and MLP approaches.

    For comparison and ablation studies.
    """

    def __init__(
        self,
        input_dim: int = 9,
        latent_dim: int = 16,
        d_model: int = 64,  # Smaller for hybrid
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        logvar_min: float = -10.0,
        logvar_max: float = 2.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        # Attention path
        self.attention_encoder = AttentionEncoder(
            input_dim=input_dim,
            latent_dim=d_model,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # MLP path (traditional)
        self.mlp_encoder = create_encoder_mlp(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            latent_dim=d_model,
            dropout=dropout,
        )

        # Fusion layer
        self.fusion = create_encoder_mlp(
            input_dim=d_model * 2,  # Concat attention + MLP
            hidden_dims=d_model,
            latent_dim=d_model,
            dropout=dropout,
        )

        # Output heads
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through hybrid encoder."""
        # Get features from attention path
        attn_mu, attn_logvar = self.attention_encoder(x)
        attn_features = attn_mu  # Use mu as features

        # Get features from MLP path
        mlp_features = self.mlp_encoder(x)

        # Fuse features
        fused = torch.cat([attn_features, mlp_features], dim=1)
        fused_features = self.fusion(fused)

        # Compute final mu and logvar
        mu = self.fc_mu(fused_features)
        logvar = self.fc_logvar(fused_features)

        # Clamp logvar
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)

        return mu, logvar


# Convenience functions
def create_attention_encoder(
    latent_dim: int = 16,
    d_model: int = 128,
    num_heads: int = 8,
    num_layers: int = 3,
    dropout: float = 0.1,
) -> AttentionEncoder:
    """Create attention encoder with sensible defaults."""
    return AttentionEncoder(
        input_dim=9,
        latent_dim=latent_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )


def create_lightweight_attention_encoder(
    latent_dim: int = 16,
    dropout: float = 0.1,
) -> AttentionEncoder:
    """Create lightweight attention encoder for performance."""
    return AttentionEncoder(
        input_dim=9,
        latent_dim=latent_dim,
        d_model=64,  # Smaller model
        num_heads=4,
        num_layers=2,
        dropout=dropout,
    )


def create_hybrid_encoder(
    latent_dim: int = 16,
    dropout: float = 0.1,
) -> HybridAttentionEncoder:
    """Create hybrid attention + MLP encoder."""
    return HybridAttentionEncoder(
        input_dim=9,
        latent_dim=latent_dim,
        dropout=dropout,
    )


__all__ = [
    "AttentionEncoder",
    "HybridAttentionEncoder",
    "PositionalEncoding",
    "OperationEmbedding",
    "MultiHeadAttention",
    "AttentionBlock",
    "AttentionPooling",
    "create_attention_encoder",
    "create_lightweight_attention_encoder",
    "create_hybrid_encoder",
]