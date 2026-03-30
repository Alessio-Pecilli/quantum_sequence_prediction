import torch
import torch.nn as nn

import config


def pack_clamped_state_features(x_complex: torch.Tensor, *, dim_2n: int) -> torch.Tensor:
    """
    Rappresentazione reale clampata:
    [Re(psi_0..psi_{N-1}), Im(psi_1..psi_{N-1})] -> 2N-1 feature.
    """
    if not torch.is_complex(x_complex):
        raise ValueError(f"pack_clamped_state_features richiede tensori complessi, dtype={x_complex.dtype}")
    if x_complex.shape[-1] != int(dim_2n):
        raise ValueError(
            f"Ultima dimensione non valida: atteso dim_2n={dim_2n}, ricevuto {x_complex.shape[-1]}"
        )
    real = torch.real(x_complex)
    imag_tail = torch.imag(x_complex[..., 1:])
    return torch.cat([real, imag_tail], dim=-1)


def unpack_clamped_state_features(features: torch.Tensor, *, dim_2n: int) -> torch.Tensor:
    feature_dim = 2 * int(dim_2n) - 1
    if features.shape[-1] != feature_dim:
        raise ValueError(
            f"Feature size non valida: atteso {feature_dim}, ricevuto {features.shape[-1]}"
        )
    real = features[..., :dim_2n]
    imag_tail = features[..., dim_2n:]
    imag = torch.cat([torch.zeros_like(real[..., :1]), imag_tail], dim=-1)
    return torch.complex(real, imag)


class ComplexEmbedding(nn.Module):
    def __init__(self, dim_2n: int = config.DIM_2N, d_model: int = config.D_MODEL):
        super().__init__()
        self.dim_2n = int(dim_2n)
        self.input_dim = 2 * self.dim_2n - 1
        hidden_dim = max(d_model, self.input_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x_complex):
            x_real = pack_clamped_state_features(x_complex.to(torch.complex64), dim_2n=self.dim_2n)
        else:
            x_real = x_complex
            if x_real.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Feature size non valida: atteso {self.input_dim}, ricevuto {x_real.shape[-1]}"
                )
        return self.projection(x_real.to(torch.float32))
