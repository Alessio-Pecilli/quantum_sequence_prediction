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
        # 1. Estrazione fisica: Modulo (ampiezza) e Angolo (fase)
        amplitudes = torch.abs(x_complex)
        phases = torch.angle(x_complex)
        
        # 2. Concatenazione. 
        # Da shape (batch, seq, dim_2n) passiamo a (batch, seq, dim_2n * 2)
        x_real = torch.cat([amplitudes, phases], dim=-1)
        
        # 3. Proiezione Lineare verso lo spazio latente d
        # Ora la rete lavora con feature puramente reali in R^d
        h = self.projection(x_real)
        
        return h