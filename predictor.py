import torch
import torch.nn as nn

import config
from embedding import ComplexEmbedding


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


@_dynamo_disable
def _normalize_and_pack_complex(real_raw: torch.Tensor, imag_raw: torch.Tensor) -> torch.Tensor:
    # Normalizzazione L2 a norma unitaria: ||psi||^2 = 1 (in FP32 per stabilità + AMP compatibility).
    real_f32 = real_raw.float()
    imag_f32 = imag_raw.float()
    norm = torch.sqrt((real_f32.square() + imag_f32.square()).sum(dim=-1, keepdim=True)).clamp(min=1e-8)
    real_f32 = real_f32 / norm
    imag_f32 = imag_f32 / norm
    return torch.complex(real_f32, imag_f32)


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) â€” Su, Lu et al. 2021.

    Codifica la posizione temporale ruotando coppie di dimensioni dello
    spazio latente. L'informazione posizionale viene iniettata direttamente
    nel prodotto scalare dell'attenzione, senza parametri aggiuntivi.

    Per un vettore h di dimensione d, raggruppiamo le feature in d/2 coppie
    (h_{2i}, h_{2i+1}) e applichiamo una rotazione di angolo t * Î¸_i dove:
      Î¸_i = 1 / (base^(2i/d))
    """

    def __init__(self, d_model=config.D_MODEL, max_seq_len=config.SEQ_LEN, base=config.ROPE_BASE):
        super().__init__()
        assert d_model % 2 == 0, f"d_model deve essere pari per RoPE, ricevuto {d_model}"

        # Frequenze: Î¸_i = 1 / (base^(2i/d))  per i = 0, 1, ..., d/2 - 1
        half_d = d_model // 2
        theta = 1.0 / (base ** (torch.arange(0, half_d).float() / half_d))

        # Posizioni temporali: t = 0, 1, ..., max_seq_len - 1
        positions = torch.arange(0, max_seq_len).float()

        # Angoli: (max_seq_len, d/2) â€” ogni posizione ha d/2 angoli
        angles = torch.outer(positions, theta)  # (seq_len, d/2)

        # Pre-calcoliamo cos e sin (non sono parametri trainabili)
        self.register_buffer("cos_cached", angles.cos())  # (seq_len, d/2)
        self.register_buffer("sin_cached", angles.sin())  # (seq_len, d/2)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model) â€” embedding reale post-proiezione
        Ritorna: (batch_size, seq_len, d_model) â€” con posizione codificata
        """
        seq_len = x.shape[1]

        cos = self.cos_cached[:seq_len]  # (seq_len, d/2)
        sin = self.sin_cached[:seq_len]  # (seq_len, d/2)

        # Split nelle coppie pari/dispari
        x_even = x[..., 0::2]  # (batch, seq, d/2)
        x_odd = x[..., 1::2]  # (batch, seq, d/2)

        # Rotazione 2D: [cos Î¸, -sin Î¸; sin Î¸, cos Î¸] applicata a ogni coppia
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos

        # Ricombina alternando le dimensioni
        out = torch.stack([x_even_rot, x_odd_rot], dim=-1)  # (batch, seq, d/2, 2)
        out = out.flatten(-2)  # (batch, seq, d)

        return out


class QuantumFidelityLoss(nn.Module):
    """
    Loss per stati quantistici complessi.

    - Backprop: MSE (distanza L2) tra stato predetto e target.
    - Logging/Early stopping: mean fidelity F = |<target | pred>|^2.
    """

    def __init__(self):
        super().__init__()

    def forward(self, state_pred, state_target):
        # Loss = infidelity = 1 - F, dove F = |<target|pred>|² (overlap al quadrato).
        # Metrica naturale per stati quantistici: 0 = perfetto, 1 = ortogonali.
        overlap = torch.sum(state_target.conj() * state_pred, dim=-1)  # (batch, seq)
        fidelity = torch.abs(overlap) ** 2  # (batch, seq)
        mean_fidelity = fidelity.mean()
        loss = 1.0 - mean_fidelity

        return loss, mean_fidelity


class QuantumStatePredictor(nn.Module):
    def __init__(self, dim_2n=config.DIM_2N, d=config.D_MODEL, num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS):
        super().__init__()
        self.d = d
        self.dim_2n = dim_2n

        # ------------------------------------------
        # 1. Modulo di Embedding (Da Complesso a Reale latente d)
        # ------------------------------------------
        self.embedding = ComplexEmbedding(dim_2n=dim_2n, d_model=d)

        # 1b. Positional Encoding: RoPE (codifica la posizione temporale)
        self.rope = RotaryPositionalEncoding(d_model=d, max_seq_len=config.SEQ_LEN)

        # 2. Core: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 2b. Maschera causale: posizione t vede solo 0..t (predizione autoregressiva)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(config.SEQ_LEN)
        self.register_buffer("causal_mask", causal_mask)

        # 3. Output Head: MLP 2-layer per decompressione graduale d â†’ 2*dim_2n
        self.output_head = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, 2 * dim_2n),
        )

    def forward(self, x_complex):
        """
        x_complex: Input tensor di shape (batch_size, seq_len, dim_2n) (DEVE essere torch.complex64 o complex128)
        Ritorna: Tensore complesso di shape (batch_size, seq_len, dim_2n)
        """
        # A. Convertiamo complesso -> reale fuori dal grafo compilato (TorchInductor non supporta complessi).
        x_real = _complex_to_real_features(x_complex)

        # A2. Proiezione verso lo spazio latente d-dimensionale
        h = self.embedding.projection(x_real)

        # A2. Iniezione informazione posizionale temporale (RoPE)
        h = self.rope(h)

        # B. Passaggio nel Transformer (con maschera causale)
        seq_len = h.shape[1]
        out = self.transformer(h, mask=self.causal_mask[:seq_len, :seq_len])

        # C. Estrazione raw features (dimensione 2*dim_2n)
        out_raw = self.output_head(out)

        # D. Split: parte reale e immaginaria
        real_raw, imag_raw = torch.chunk(out_raw, chunks=2, dim=-1)

        # --- STATO COMPLESSO + VINCOLO FISICO ---
        # Pack + normalizzazione fuori dal grafo compilato (evita lowering complessi).
        return _normalize_and_pack_complex(real_raw, imag_raw)
