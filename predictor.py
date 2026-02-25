import torch
import torch.nn as nn

import config


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
    def __init__(self, d=config.D_MODEL, num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS):
        super().__init__()
        self.d = d
        
        # ------------------------------------------
        # TODO: Modulo di Embedding (Word + Positional)
        # Qui implementerai la proiezione dal tuo vettore
        # di 2^n elementi allo spazio d-dimensionale.
        # ------------------------------------------
        
        # Core: Transformer Encoder (batch_first=True per shape [batch, seq, features])
        encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output Head: Proietta da d a 2d (d ampiezze + d fasi)
        self.output_head = nn.Linear(d, 2 * d)

    def forward(self, x):
        """
        x: Input tensor di shape (batch_size, seq_len, d)
        Ritorna: Tensore complesso di shape (batch_size, seq_len, d)
        """
        # TODO: h = self.embedding_module(x_original)
        h = x # Per ora l'input è già considerato nello spazio d
        
        # Passaggio nel Transformer
        out = self.transformer(h)
        
        # Estrazione raw features (dimensione 2d)
        out_raw = self.output_head(out)
        
        # Split a metà lungo l'ultima dimensione
        amplitudes_raw, phases_raw = torch.chunk(out_raw, chunks=2, dim=-1)
        
        # --- VINCOLI FISICI ---
        # Ampiezze: normalizzate (somma dei quadrati = 1) e positive
        amplitudes = torch.sqrt(torch.softmax(amplitudes_raw, dim=-1))
        
        # Fasi: mappate in [-pi, pi] per stabilità
        phases = torch.tanh(phases_raw) * torch.pi
        
        # --- STATO COMPLESSO FINALE ---
        # Costruzione a * e^(i * phi)
        complex_state = amplitudes * torch.exp(1j * phases)
        
        return complex_state