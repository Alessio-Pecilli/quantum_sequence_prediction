import torch
from torch.utils.data import Dataset, DataLoader

import config


class QuantumStateDataset(Dataset):
    def __init__(self, inputs_data, targets_data):
        """
        Qui passi i tuoi dati reali.
        inputs_data: Tensore di shape (N_samples, seq_len, d) - Reale
        targets_data: Tensore di shape (N_samples, seq_len, d) - Complesso
        """
        self.inputs = inputs_data
        self.targets = targets_data

        # Un check di sicurezza (best practice)
        assert len(self.inputs) == len(self.targets), "Mismatch tra input e target!"

    def __len__(self):
        # Ritorna il numero totale di campioni nel dataset
        return len(self.inputs)

    def __getitem__(self, idx):
        # Estrae un singolo campione. Il DataLoader chiamerà questo metodo
        # per assemblare i batch in automatico.
        x = self.inputs[idx]
        y_target = self.targets[idx]

        return x, y_target


# --- MATRICI DI PAULI BASE ---
# Definite rigorosamente in campo complesso
I = torch.eye(2, dtype=torch.complex64)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

def get_pauli_string(operator_list):
    """
    Costruisce l'operatore denso 2^n x 2^n tramite prodotto di Kronecker.
    operator_list: lista di tensori 2x2 (es. [I, X, I, I] per applicare X al secondo qubit su 4)
    """
    res = operator_list[0]
    for op in operator_list[1:]:
        res = torch.kron(res, op)
    return res

def build_tfim_hamiltonian(n_qubits, J, g):
    """
    MODULO ISOLATO: Costruisce l'Hamiltoniana TFIM per n_qubits.
    In futuro, se cambi sistema fisico, modifichi solo il corpo di questa funzione.
    Ritorna: Tensore (2^n, 2^n) complesso.
    """
    dim = 2**n_qubits
    H = torch.zeros((dim, dim), dtype=torch.complex64)
    
    # Termine di interazione (ZZ) sui vicini (condizioni al contorno aperte)
    for i in range(n_qubits - 1):
        ops = [I] * n_qubits
        ops[i] = Z
        ops[i+1] = Z
        H -= J * get_pauli_string(ops)
        
    # Termine di campo trasverso (X) su ogni qubit
    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = X
        H -= g * get_pauli_string(ops)
        
    return H

def generate_quantum_dynamics_dataset(
    n_qubits=config.N_QUBITS,
    B=config.N_HAMILTONIANS,
    S=config.N_STATES_PER_H,
    seq_len=config.SEQ_LEN,
    dt=config.DT,
    J_range=config.J_RANGE,
    g_range=config.G_RANGE,
):
    """
    Genera il dataset B * S traiettorie.
    n_qubits: Numero di qubit del sistema
    B: Numero di Hamiltoniane casuali da campionare
    S: Numero di stati iniziali casuali per ogni Hamiltoniana
    seq_len: Lunghezza della sequenza di training (quanti step di evoluzione predirre)
    dt: Passo temporale dell'evoluzione delta_t
    """
    dim = 2**n_qubits
    total_samples = B * S
    
    # Pre-allochiamo i tensori finali per massima efficienza in RAM
    # Shape: (B*S, seq_len, 2^n)
    inputs_data = torch.empty((total_samples, seq_len, dim), dtype=torch.complex64)
    targets_data = torch.empty((total_samples, seq_len, dim), dtype=torch.complex64)
    
    sample_idx = 0
    
    for b in range(B):
        # 1. Estrazione parametri casuali per l'Hamiltoniana
        J_b = torch.empty(1).uniform_(*J_range).item()
        g_b = torch.empty(1).uniform_(*g_range).item()
        
        # 2. Costruzione H e operatore unitario di evoluzione U
        H_b = build_tfim_hamiltonian(n_qubits, J=J_b, g=g_b)
        
        # U = exp(-i * H * dt)
        # torch.matrix_exp calcola l'esponenziale di matrice in modo esatto
        U_b = torch.matrix_exp(-1j * H_b * dt) 
        
        for s in range(S):
            # 3. Estrazione stato iniziale casuale (normalizzato)
            psi_real = torch.randn(dim)
            psi_imag = torch.randn(dim)
            psi = torch.complex(psi_real, psi_imag)
            psi = psi / torch.norm(psi) # Normalizzazione fisica a 1
            
            # 4. Evoluzione temporale per creare la traiettoria
            # Ci servono seq_len + 1 stati per creare le coppie (input_t, target_t+1)
            trajectory = []
            current_state = psi
            
            for t in range(seq_len + 1):
                trajectory.append(current_state)
                # Evoluzione: |psi(t+dt)> = U |psi(t)>
                current_state = torch.matmul(U_b, current_state)
                
            # Trasformiamo la lista in un tensore di shape (seq_len + 1, 2^n)
            trajectory_tensor = torch.stack(trajectory)
            
            # 5. Split tramite shift temporale:
            # Input: da t=0 a t=seq_len-1
            # Target: da t=1 a t=seq_len
            inputs_data[sample_idx] = trajectory_tensor[:-1]
            targets_data[sample_idx] = trajectory_tensor[1:]
            
            sample_idx += 1
            
    return inputs_data, targets_data