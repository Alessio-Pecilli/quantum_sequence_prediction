import torch
import torch.nn as nn

import config
from embedding import ComplexEmbedding
from observables import batch_observables_diff


def _dynamo_disable(fn):
    disable = getattr(torch, "_dynamo", None)
    if disable is not None and hasattr(disable, "disable"):
        return disable.disable(fn)
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "disable"):
        return compiler.disable(fn)
    return fn


@_dynamo_disable
def _complex_to_real_features(x_complex: torch.Tensor) -> torch.Tensor:
    # (B, T, D) complex -> (B, T, 2D) float without triggering Inductor complex lowering.
    return torch.view_as_real(x_complex).flatten(-2)


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) - Su, Lu et al. 2021.

    Codifica la posizione temporale ruotando coppie di dimensioni dello
    spazio latente. L'informazione posizionale viene iniettata direttamente
    nel prodotto scalare dell'attenzione, senza parametri aggiuntivi.

    Per un vettore h di dimensione d, raggruppiamo le feature in d/2 coppie
    (h_{2i}, h_{2i+1}) e applichiamo una rotazione di angolo t * theta_i dove:
      theta_i = 1 / (base^(2i/d))
    """

    def __init__(self, d_model=config.D_MODEL, max_seq_len=config.SEQ_LEN, base=config.ROPE_BASE):
        super().__init__()
        assert d_model % 2 == 0, f"d_model deve essere pari per RoPE, ricevuto {d_model}"

        half_d = d_model // 2
        theta = 1.0 / (base ** (torch.arange(0, half_d).float() / half_d))
        positions = torch.arange(0, max_seq_len).float()
        angles = torch.outer(positions, theta)  # (seq_len, d/2)

        self.register_buffer("cos_cached", angles.cos())  # (seq_len, d/2)
        self.register_buffer("sin_cached", angles.sin())  # (seq_len, d/2)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        ritorna: (batch_size, seq_len, d_model) con posizione codificata
        """
        seq_len = x.shape[1]

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos

        out = torch.stack([x_even_rot, x_odd_rot], dim=-1)
        out = out.flatten(-2)
        return out


class QuantumFidelityLoss(nn.Module):
    """
    Loss per stati quantistici complessi.

    - Backprop: infidelity 1 - F
    - Logging/Early stopping: mean fidelity F = |<target | pred>|^2
    """

    def __init__(self):
        super().__init__()

    def forward(self, state_pred, state_target):
        overlap = torch.sum(state_target.conj() * state_pred, dim=-1)
        fidelity = torch.abs(overlap) ** 2
        mean_fidelity = fidelity.mean()
        loss = 1.0 - mean_fidelity
        return loss, mean_fidelity


class PhysicsInformedLoss(nn.Module):
    """
    Loss composita:
      - infidelity globale (1 - fidelity), pesata da LAMBDA_FID
      - MSE sulle osservabili (m^z, m^x, c^z) opzionali
    """

    def __init__(
        self,
        z_eigs: torch.Tensor,
        zz_nn_eigs: torch.Tensor,
        zz_all_eigs: torch.Tensor,
        x_flip_idx: list[torch.Tensor],
        lambda_fid: float = config.LAMBDA_FID,
        lambda_mz: float = config.LAMBDA_MZ,
        lambda_mx: float = config.LAMBDA_MX,
        lambda_cz: float = config.LAMBDA_CZ,
    ):
        super().__init__()
        self.lambda_fid = float(lambda_fid)
        self.lambda_mz = float(lambda_mz)
        self.lambda_mx = float(lambda_mx)
        self.lambda_cz = float(lambda_cz)

        self._use_obs = any(w > 0.0 for w in (self.lambda_mz, self.lambda_mx, self.lambda_cz))
        self.z_eigs = z_eigs
        self.zz_nn_eigs = zz_nn_eigs
        self.zz_all_eigs = zz_all_eigs
        self.x_flip_idx = list(x_flip_idx)

    @staticmethod
    def _flatten_states(states: torch.Tensor) -> torch.Tensor:
        if states.ndim < 2:
            raise ValueError(f"states deve avere almeno 2 dimensioni, ricevuto {tuple(states.shape)}")
        return states.reshape(-1, states.shape[-1])

    def forward(self, state_pred: torch.Tensor, state_target: torch.Tensor):
        if state_pred.shape != state_target.shape:
            raise ValueError(
                f"state_pred e state_target devono avere la stessa shape: "
                f"{tuple(state_pred.shape)} vs {tuple(state_target.shape)}"
            )

        pred_flat = self._flatten_states(state_pred)
        target_flat = self._flatten_states(state_target)

        overlap = torch.sum(target_flat.conj() * pred_flat, dim=-1)
        fidelity = torch.abs(overlap) ** 2
        mean_fidelity = fidelity.mean()

        total_loss = self.lambda_fid * (1.0 - mean_fidelity)

        if self._use_obs:
            mz_pred, mx_pred, cz_pred, _, _ = batch_observables_diff(
                pred_flat, self.z_eigs, self.zz_nn_eigs, self.zz_all_eigs, self.x_flip_idx
            )
            mz_tgt, mx_tgt, cz_tgt, _, _ = batch_observables_diff(
                target_flat, self.z_eigs, self.zz_nn_eigs, self.zz_all_eigs, self.x_flip_idx
            )

            if self.lambda_mz > 0.0:
                total_loss = total_loss + self.lambda_mz * torch.mean((mz_pred - mz_tgt) ** 2)
            if self.lambda_mx > 0.0:
                total_loss = total_loss + self.lambda_mx * torch.mean((mx_pred - mx_tgt) ** 2)
            if self.lambda_cz > 0.0:
                total_loss = total_loss + self.lambda_cz * torch.mean((cz_pred - cz_tgt) ** 2)

        return total_loss, mean_fidelity


class QuantumStatePredictor(nn.Module):
    def __init__(self, dim_2n=config.DIM_2N, d=config.D_MODEL, num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS):
        super().__init__()
        self.d = d
        self.dim_2n = dim_2n

        # 1) Embedding complesso->reale (immutato)
        self.embedding = ComplexEmbedding(dim_2n=dim_2n, d_model=d)

        # 1b) Positional encoding (immutato)
        self.rope = RotaryPositionalEncoding(d_model=d, max_seq_len=config.SEQ_LEN)

        # 2) Core Transformer encoder (immutato)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(config.SEQ_LEN)
        self.register_buffer("causal_mask", causal_mask)

        # 3) Output head: 2 * dim^2 real values (real+imag della matrice dim x dim)
        self.output_head = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, 2 * dim_2n * dim_2n),
        )
        # ZERO-INIT per la stabilita' dell'esponenziale di matrice
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, x_complex):
        """
        x_complex: (batch, seq, dim) complesso
        ritorna:   (batch, seq, dim) complesso con evoluzione unitaria
        """
        # A) Complesso -> reale per il backbone
        x_real = _complex_to_real_features(x_complex)

        # B) Proiezione + RoPE + Transformer
        h = self.embedding.projection(x_real)
        h = self.rope(h)

        seq_len = h.shape[1]
        out = self.transformer(h, mask=self.causal_mask[:seq_len, :seq_len])

        # C) Parametri della matrice complessa generica M
        out_raw = self.output_head(out)  # (B, S, 2*dim^2)
        real_raw, imag_raw = torch.chunk(out_raw, chunks=2, dim=-1)  # (B, S, dim^2), (B, S, dim^2)

        bsz, ssz = real_raw.shape[:2]
        real_matrix = real_raw.reshape(bsz, ssz, self.dim_2n, self.dim_2n)
        imag_matrix = imag_raw.reshape(bsz, ssz, self.dim_2n, self.dim_2n)

        # D) M complessa e simmetrizzazione hermitiana
        m_complex = real_matrix + 1j * imag_matrix
        h_eff = m_complex + m_complex.adjoint()

        # E) Operatore unitario U = exp(-i H_eff)
        u_op = torch.matrix_exp(-1j * h_eff)

        # F) Evoluzione dello stato: psi_next = U @ psi_current
        psi_next = torch.einsum("bsij,bsj->bsi", u_op, x_complex)
        return psi_next
