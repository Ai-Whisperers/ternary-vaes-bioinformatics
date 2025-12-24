from torch import nn
import torch


class MockFrozenModule(nn.Module):
    """Shared mock for frozen VAE components."""

    def __init__(self, output_shape, return_tuple=False):
        super().__init__()
        self.output_shape = output_shape
        self.return_tuple = return_tuple

    def forward(self, x):
        batch_size = x.shape[0]
        if self.return_tuple:
            # For encoder: returns (mu, logvar)
            return (
                torch.randn(batch_size, *self.output_shape),
                torch.randn(batch_size, *self.output_shape),
            )
        return torch.randn(batch_size, *self.output_shape)

    @classmethod
    def from_v5_5_checkpoint(cls, *args, **kwargs):
        return cls(output_shape=(16,), return_tuple=True)
