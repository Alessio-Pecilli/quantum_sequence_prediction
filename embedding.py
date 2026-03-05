import torch
import torch.nn as nn
import config


class ComplexEmbedding(nn.Module):
    """
    Traduce uno stato quantistico complesso in un vettore reale latente.
    Estrae ampiezze e fasi, le concatena e applica una proiezione lineare.
    """
    def __init__(self, dim_2n=config.DIM_2N, d_model=config.D_MODEL):
        super().__init__()
        # In ingresso abbiamo (ampiezze + fasi) concatenate lungo l'ultima dimensione,
        # quindi il layer lineare deve accettare il doppio delle feature (dim_2n * 2)
        self.projection = nn.Linear(dim_2n * 2, d_model)

    def forward(self, x_complex):
        """
        x_complex: Tensore complesso di shape (batch_size, seq_len, dim_2n)
        """
        # 1. Estrazione parti reale e immaginaria (continua, nessuna discontinuità)
        #    Rispetto a amp/fase: evita instabilità quando amp→0 (fase indefinita)
        #    e fornisce gradienti più lisci alla rete
        real_part = x_complex.real
        imag_part = x_complex.imag
        
        # 2. Concatenazione: (batch, seq, dim_2n) → (batch, seq, dim_2n * 2)
        x_real = torch.cat([real_part, imag_part], dim=-1)
        
        # 3. Proiezione Lineare verso lo spazio latente d
        h = self.projection(x_real)
        
        return h