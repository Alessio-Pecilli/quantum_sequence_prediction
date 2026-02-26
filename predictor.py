import torch
import torch.nn as nn

import config
from embedding import ComplexEmbedding


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
        
        # ------------------------------------------
        # 1. Modulo di Embedding (Da Complesso a Reale latente d)
        # ------------------------------------------
        self.embedding = ComplexEmbedding(dim_2n=dim_2n, d_model=d)
        
        # 2. Core: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output Head: Proietta da d a 2d (d ampiezze + d fasi)
        self.output_head = nn.Linear(d, 2 * d)

    def forward(self, x_complex):
        """
        x_complex: Input tensor di shape (batch_size, seq_len, dim_2n) (DEVE essere torch.complex64 o complex128)
        Ritorna: Tensore complesso di shape (batch_size, seq_len, d)
        """
        # A. Elaborazione iniziale: trasformiamo il tensore complesso in un embedding reale d-dimensionale
        h = self.embedding(x_complex)
        
        # B. Passaggio nel Transformer
        out = self.transformer(h)
        
        # C. Estrazione raw features (dimensione 2d)
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