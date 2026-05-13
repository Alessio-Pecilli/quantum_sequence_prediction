import math

import torch
import torch.nn as nn

import config


def pack_clamped_state_features(x_complex: torch.Tensor, *, dim_2n: int) -> torch.Tensor:
    """
    Rappresentazione reale clampata legacy:
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
        hidden_dim = max(int(d_model), self.input_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(d_model)),
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


class TTNEncoder(nn.Module):
    def __init__(
        self,
        num_qubits: int = config.N_QUBITS,
        latent_dim: int = config.TTN_LATENT_DIM,
        d_model: int = config.D_MODEL,
    ):
        super().__init__()
        self.num_qubits = int(num_qubits)
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits deve essere >= 1, ricevuto: {self.num_qubits}")

        self.dim_2n = 2 ** self.num_qubits
        self.num_layers = int(math.log2(self.dim_2n))
        if 2 ** self.num_layers != self.dim_2n:
            raise ValueError(
                f"dim_2n={self.dim_2n} non e' una potenza di 2, impossibile costruire TTN perfetto."
            )

        self.latent_dim = int(latent_dim)
        self.input_layer = nn.Linear(4, self.latent_dim)
        self.input_skip = nn.Linear(2, self.latent_dim)
        self.input_norm = nn.LayerNorm(self.latent_dim)
        self.tree_layers = nn.ModuleList(
            nn.Linear(2 * self.latent_dim, self.latent_dim) for _ in range(self.num_layers - 1)
        )
        self.tree_norms = nn.ModuleList(nn.LayerNorm(self.latent_dim) for _ in range(self.num_layers - 1))
        self.pre_output_norm = nn.LayerNorm(self.latent_dim)
        self.output_projection = nn.Linear(self.latent_dim, int(d_model))
        self.activation = nn.GELU()

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x_complex):
            raise ValueError(f"TTNEncoder richiede tensori complessi, ricevuto dtype={x_complex.dtype}")
        if x_complex.shape[-1] != self.dim_2n:
            raise ValueError(
                f"Ultima dimensione non valida: atteso dim_2n={self.dim_2n}, ricevuto {x_complex.shape[-1]}"
            )

        x = torch.view_as_real(x_complex.to(torch.complex64))
        leading_shape = x.shape[:-2]
        x = x.reshape(-1, self.dim_2n, 2).to(torch.float32)

        for layer_idx in range(self.num_layers):
            batch_size, length, feat_dim = x.shape
            if length % 2 != 0:
                raise ValueError(
                    f"Lunghezza non pari al livello {layer_idx}: ricevuto L={length}, atteso multiplo di 2."
                )

            if layer_idx == 0:
                pair_tensor = x.reshape(batch_size, length // 2, 2, feat_dim)
                merged = pair_tensor.reshape(batch_size, length // 2, 2 * feat_dim)
                x = self.activation(self.input_layer(merged))
                skip = self.input_skip(pair_tensor.mean(dim=2))
                x = x + skip
                x = self.input_norm(x)
            else:
                pair_tensor = x.reshape(batch_size, length // 2, 2, feat_dim)
                merged = pair_tensor.reshape(batch_size, length // 2, 2 * feat_dim)
                skip = pair_tensor.mean(dim=2)
                x = self.activation(self.tree_layers[layer_idx - 1](merged))
                x = x + skip
                x = self.tree_norms[layer_idx - 1](x)

        if x.shape[1] != 1:
            raise RuntimeError(f"Riduzione TTN incompleta: atteso L=1, ottenuto L={x.shape[1]}")

        x = x.squeeze(1)
        x = self.pre_output_norm(x)
        x = self.output_projection(x)
        return x.reshape(*leading_shape, x.shape[-1])


FlatCoefficientTTNEncoder = TTNEncoder
