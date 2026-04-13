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


class QuantumStateEncoder(nn.Module):
    def __init__(
        self,
        dim_2n: int = config.DIM_2N,
        embedding_dim: int = config.EMBEDDING_DIM,
        hidden_dim: int = config.AUTOENCODER_HIDDEN_DIM,
    ):
        super().__init__()
        self.dim_2n = int(dim_2n)
        self.input_dim = 2 * self.dim_2n - 1
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = max(int(hidden_dim), self.embedding_dim * 4)
        self.hidden_dim_half = max(self.hidden_dim // 2, self.embedding_dim * 2)
        self.hidden_dim_quarter = max(self.hidden_dim // 4, self.embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim_half),
            nn.SiLU(),
            nn.Linear(self.hidden_dim_half, self.hidden_dim_quarter),
            nn.SiLU(),
            nn.Linear(self.hidden_dim_quarter, self.embedding_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(states):
            features = pack_clamped_state_features(x_complex=states.to(torch.complex64), dim_2n=self.dim_2n)
        else:
            features = states.to(torch.float32)
            if features.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Feature size non valida: atteso {self.input_dim}, ricevuto {features.shape[-1]}"
                )
        return self.network(features.to(torch.float32))


class QuantumStateDecoder(nn.Module):
    def __init__(
        self,
        dim_2n: int = config.DIM_2N,
        embedding_dim: int = config.EMBEDDING_DIM,
        hidden_dim: int = config.AUTOENCODER_HIDDEN_DIM,
    ):
        super().__init__()
        self.dim_2n = int(dim_2n)
        self.embedding_dim = int(embedding_dim)
        self.feature_dim = 2 * self.dim_2n - 1
        self.hidden_dim = max(int(hidden_dim), self.embedding_dim * 4)
        self.hidden_dim_half = max(self.hidden_dim // 2, self.embedding_dim * 2)
        self.hidden_dim_quarter = max(self.hidden_dim // 4, self.embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim_quarter),
            nn.LayerNorm(self.hidden_dim_quarter),
            nn.SiLU(),
            nn.Linear(self.hidden_dim_quarter, self.hidden_dim_half),
            nn.SiLU(),
            nn.Linear(self.hidden_dim_half, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )

    def forward(self, latent_states: torch.Tensor) -> torch.Tensor:
        features = self.network(latent_states.to(torch.float32))
        return unpack_clamped_state_features(features, dim_2n=self.dim_2n)


class QuantumStateAutoencoder(nn.Module):
    def __init__(
        self,
        dim_2n: int = config.DIM_2N,
        embedding_dim: int = config.EMBEDDING_DIM,
        hidden_dim: int = config.AUTOENCODER_HIDDEN_DIM,
    ):
        super().__init__()
        self.encoder = QuantumStateEncoder(
            dim_2n=dim_2n,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
        self.decoder = QuantumStateDecoder(
            dim_2n=dim_2n,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states)

    def decode(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_states)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_states = self.encode(states)
        reconstructed_states = self.decode(latent_states)
        return latent_states, reconstructed_states
