from __future__ import annotations

import math

import torch
import torch.nn as nn

import config
from embedding import ComplexEmbedding


def normalize_state(states: torch.Tensor) -> torch.Tensor:
    return states / torch.linalg.vector_norm(states, dim=-1, keepdim=True).clamp(min=1e-8)


def quantum_fidelity(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = normalize_state(predicted)
    target_norm = normalize_state(target)
    overlap = torch.sum(target_norm.conj() * pred_norm, dim=-1)
    fidelity = torch.abs(overlap) ** 2
    return fidelity.clamp(0.0, 1.0)


class NegativeLogFidelityLoss(nn.Module):
    def __init__(self, epsilon: float = config.LOG_FIDELITY_EPS):
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        fidelity = quantum_fidelity(predicted, target)
        loss = -torch.log(fidelity.clamp(min=self.epsilon)).mean()
        return loss, fidelity.mean(), fidelity


class ComplexMSELoss(nn.Module):
    """
    Allinea ampiezze e fase (non solo la fidelity) con MSE sulle componenti reali/immaginarie.
    """

    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        # predicted/target sono attesi come stati complessi normalizzati.
        pred_real = torch.view_as_real(predicted)
        target_real = torch.view_as_real(target)
        loss = torch.nn.functional.mse_loss(pred_real, target_real)

        # Fidelity solo per reporting/plot: non influenza direttamente il gradiente.
        with torch.no_grad():
            fidelity = quantum_fidelity(predicted, target)

        return loss, fidelity.mean(), fidelity


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.encoding[: hidden.shape[1]]


class QuantumSequencePredictor(nn.Module):
    def __init__(
        self,
        dim_2n: int = config.DIM_2N,
        d_model: int = config.D_MODEL,
        num_heads: int = config.NUM_HEADS,
        num_layers: int = config.NUM_LAYERS,
        dim_feedforward: int = config.DIM_FEEDFORWARD,
        dropout: float = config.DROPOUT,
        max_seq_len: int = config.SEQ_LEN,
    ):
        super().__init__()
        self.dim_2n = int(dim_2n)
        self.embedding = ComplexEmbedding(dim_2n=dim_2n, d_model=d_model)
        self.position_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * self.dim_2n),
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, context_states: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(context_states)
        hidden = self.position_encoding(hidden)
        seq_len = int(hidden.shape[1])
        hidden = self.transformer(hidden, mask=self.causal_mask[:seq_len, :seq_len])
        hidden = self.output_norm(hidden)
        raw = self.output_head(hidden)
        real_part, imag_part = raw.chunk(2, dim=-1)
        predicted = torch.complex(real_part, imag_part)
        return normalize_state(predicted)
