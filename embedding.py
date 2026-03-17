import torch
import torch.nn as nn

import config


class ComplexEmbedding(nn.Module):
    def __init__(self, dim_2n: int = config.DIM_2N, d_model: int = config.D_MODEL):
        super().__init__()
        self.input_dim = int(dim_2n) * 2
        hidden_dim = max(d_model, self.input_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        x_real = torch.view_as_real(x_complex.to(torch.complex64)).flatten(-2)
        return self.projection(x_real)
