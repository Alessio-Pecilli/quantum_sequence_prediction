# ===== Configurazione Modello =====

# Dimensione dello spazio di embedding (2^n per n qubit)
D_MODEL = 64

# Numero di teste di attenzione nel Transformer
NUM_HEADS = 4

# Numero di layer del Transformer Encoder
NUM_LAYERS = 2

# ===== Configurazione Sistema Quantistico =====

# Numero di qubit del sistema
N_QUBITS = 4

# Dimensione dello spazio di Hilbert (2^n)
DIM_2N = 2 ** N_QUBITS

# Parametri dell'Hamiltoniana TFIM
J_RANGE = (0.5, 1.5)
G_RANGE = (0.5, 1.5)

# Passo temporale dell'evoluzione
DT = 0.1

# ===== Configurazione Dataset =====

# Numero di Hamiltoniane casuali da campionare (B)
N_HAMILTONIANS = 100

# Numero di stati iniziali per Hamiltoniana (S)
N_STATES_PER_H = 100

# Numero totale di campioni nel dataset
N_TOTALE = N_HAMILTONIANS * N_STATES_PER_H

# Lunghezza della sequenza temporale
SEQ_LEN = 10

# ===== Configurazione Training =====

# Percentuale di dati per il training (il resto va al test)
TRAIN_SPLIT = 0.8

# Dimensione del batch
BATCH_SIZE = 32

# Numero di epoche di addestramento
EPOCHS = 50

# Learning rate dell'ottimizzatore
LEARNING_RATE = 1e-3

# Device (gpu o cpu) - determinato al momento dell'uso
def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = get_device()
