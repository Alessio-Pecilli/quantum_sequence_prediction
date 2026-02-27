import torch
import torch.nn as nn

import config
from embedding import ComplexEmbedding


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) — Su, Lu et al. 2021.
    
    Codifica la posizione temporale ruotando coppie di dimensioni dello
    spazio latente. L'informazione posizionale viene iniettata direttamente
    nel prodotto scalare dell'attenzione, senza parametri aggiuntivi.
    
    Per un vettore h di dimensione d, raggruppiamo le feature in d/2 coppie
    (h_{2i}, h_{2i+1}) e applichiamo una rotazione di angolo t * θ_i dove:
      θ_i = 1 / (base^(2i/d))
    """
    def __init__(self, d_model=config.D_MODEL, max_seq_len=config.SEQ_LEN, base=config.ROPE_BASE):
        super().__init__()
        assert d_model % 2 == 0, f"d_model deve essere pari per RoPE, ricevuto {d_model}"
        
        # Frequenze: θ_i = 1 / (base^(2i/d))  per i = 0, 1, ..., d/2 - 1
        half_d = d_model // 2
        theta = 1.0 / (base ** (torch.arange(0, half_d).float() / half_d))
        
        # Posizioni temporali: t = 0, 1, ..., max_seq_len - 1
        positions = torch.arange(0, max_seq_len).float()
        
        # Angoli: (max_seq_len, d/2) — ogni posizione ha d/2 angoli
        angles = torch.outer(positions, theta)  # (seq_len, d/2)
        
        # Pre-calcoliamo cos e sin (non sono parametri trainabili)
        self.register_buffer("cos_cached", angles.cos())  # (seq_len, d/2)
        self.register_buffer("sin_cached", angles.sin())  # (seq_len, d/2)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model) — embedding reale post-proiezione
        Ritorna: (batch_size, seq_len, d_model) — con posizione codificata
        """
        seq_len = x.shape[1]
        
        cos = self.cos_cached[:seq_len]  # (seq_len, d/2)
        sin = self.sin_cached[:seq_len]  # (seq_len, d/2)
        
        # Split nelle coppie pari/dispari
        x_even = x[..., 0::2]  # (batch, seq, d/2)
        x_odd  = x[..., 1::2]  # (batch, seq, d/2)
        
        # Rotazione 2D: [cos θ, -sin θ; sin θ, cos θ] applicata a ogni coppia
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot  = x_even * sin + x_odd * cos
        
        # Ricombina alternando le dimensioni
        out = torch.stack([x_even_rot, x_odd_rot], dim=-1)  # (batch, seq, d/2, 2)
        out = out.flatten(-2)  # (batch, seq, d)
        
        return out


class QuantumFidelityLoss(nn.Module):
    """
    Calcola l'errore basato sulla Fidelity: F = |<target | pred>|^2.
    La loss da minimizzare è 1 - F.
    """
    def __init__(self):
        super().__init__()

    def forward(self, state_pred, state_target):
        # Prodotto interno <target | pred> (somma lungo la dimensione d delle feature)
        # Usiamo .conj() per ottenere il "bra" del target
        overlap = torch.sum(state_target.conj() * state_pred, dim=-1)
        
        # Modulo quadro
        fidelity = torch.abs(overlap) ** 2
        
        # Media su batch e sequenza
        mean_fidelity = fidelity.mean()
        
        # Loss da minimizzare
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
        encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output Head: Proietta da d a 2*dim_2n (dim_2n ampiezze + dim_2n fasi)
        self.output_head = nn.Linear(d, 2 * dim_2n)

    def forward(self, x_complex):
        """
        x_complex: Input tensor di shape (batch_size, seq_len, dim_2n) (DEVE essere torch.complex64 o complex128)
        Ritorna: Tensore complesso di shape (batch_size, seq_len, dim_2n)
        """
        # A. Elaborazione iniziale: trasformiamo il tensore complesso in un embedding reale d-dimensionale
        h = self.embedding(x_complex)
        
        # A2. Iniezione informazione posizionale temporale (RoPE)
        h = self.rope(h)
        
        # B. Passaggio nel Transformer
        out = self.transformer(h)
        
        # C. Estrazione raw features (dimensione 2*dim_2n)
        out_raw = self.output_head(out)
        
        # D. Split a metà lungo l'ultima dimensione
        amplitudes_raw, phases_raw = torch.chunk(out_raw, chunks=2, dim=-1)
        
        # --- VINCOLI FISICI ---
        # Ampiezze: normalizzate (somma dei quadrati = 1) e positive
        amplitudes = torch.sqrt(torch.softmax(amplitudes_raw, dim=-1))
        
        # Fasi: mappate in [-pi, pi] per stabilità
        phases = torch.sigmoid(phases_raw) * 2 * torch.pi
        
        # --- STATO COMPLESSO FINALE ---
        # Costruzione a * e^(i * phi)
        complex_state = amplitudes * torch.exp(1j * phases)
        
        return complex_state