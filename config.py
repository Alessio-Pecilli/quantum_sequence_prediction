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

# Base per RoPE (Rotary Positional Encoding)
ROPE_BASE = 10000.0

# ===== Configurazione Sistema Fisico =====

# Tipo di Hamiltoniana (fissa per tutto l'esperimento)
# Opzioni supportate: "TFIM"
HAMILTONIAN_TYPE = "TFIM"

# ===== Configurazione Dataset =====

# --- Training ---
# Numero di Hamiltoniane casuali da campionare (B)
B_TRAIN = 5
# Numero di stati iniziali per Hamiltoniana (S)
S_TRAIN = 10

# --- Test ---
# Nuove Hamiltoniane e nuovi stati (generati indipendentemente dal training)
B_TEST = 2
S_TEST = 10

# Lunghezza della sequenza temporale
SEQ_LEN = 10

# Dimensione del batch (su CPU batch grandi riducono l'overhead per iterazione)
BATCH_SIZE = 16

# Numero di epoche di addestramento
EPOCHS = 100

# ===== Configurazione Ottimizzatore =====

# Learning rate iniziale (peak dopo warmup)
LEARNING_RATE = 1e-3

# Weight decay per AdamW (regolarizzazione L2)
WEIGHT_DECAY = 1e-4

# Gradient clipping (max norm) — stabilizza il training dei Transformer
GRAD_CLIP_MAX_NORM = 1.0

# ===== Configurazione Learning Rate Scheduler =====

# Warmup: numero di epoche con rampa lineare da 0 → LEARNING_RATE
LR_WARMUP_EPOCHS = 5

# Tipo di scheduler dopo il warmup:
#   "cosine"            → Cosine Annealing fino a LR_MIN
#   "plateau"           → ReduceLROnPlateau (reattivo alla loss)
#   "cosine+plateau"    → Cosine Annealing + fallback ReduceLROnPlateau
LR_SCHEDULER_TYPE = "cosine+plateau"

# LR minimo raggiungibile (floor per qualsiasi scheduler)
LR_MIN = 1e-6

# ReduceLROnPlateau: patience (epoche senza miglioramento prima di ridurre LR)
LR_PLATEAU_PATIENCE = 8

# ReduceLROnPlateau: fattore di riduzione (new_lr = lr * factor)
LR_PLATEAU_FACTOR = 0.5

# ===== Configurazione Early Stopping =====

# Attiva/disattiva early stopping
EARLY_STOPPING_ENABLED = True

# Metrica da monitorare ("test_loss" o "test_fidelity")
EARLY_STOPPING_METRIC = "test_fidelity"

# Patience: epoche senza miglioramento prima di fermarsi
EARLY_STOPPING_PATIENCE = 25

# Delta minimo per considerare un miglioramento significativo
EARLY_STOPPING_MIN_DELTA = 1e-5

# ===== Configurazione Checkpointing =====

# Salva il miglior modello durante il training
SAVE_BEST_MODEL = True

# Path per il checkpoint del miglior modello
BEST_MODEL_PATH = "results/best_model.pt"

# Salva checkpoint periodici ogni N epoche (0 = disabilitato)
CHECKPOINT_EVERY_N_EPOCHS = 0

# ===== Configurazione EMA (Exponential Moving Average) =====

# Attiva EMA dei pesi del modello per una valutazione più stabile
EMA_ENABLED = True

# Decay factor per EMA (tipicamente 0.999 - 0.9999)
EMA_DECAY = 0.999

# ===== Mixed Precision Training =====

# Usa AMP (Automatic Mixed Precision) se disponibile su CUDA
AMP_ENABLED = True

# ===== Gradient Accumulation =====

# Numero di step di accumulo prima di aggiornare i pesi
# Simula un batch_size effettivo = BATCH_SIZE * GRAD_ACCUMULATION_STEPS
GRAD_ACCUMULATION_STEPS = 1

# ===== Compilazione Modello =====

# Usa torch.compile() per ottimizzare il modello (PyTorch 2.0+)
# Riduce overhead Python e fonde operazioni. Prima epoca piu' lenta (compilazione).
TORCH_COMPILE = True

# Backend per torch.compile: "inductor" (richiede compilatore C++), "aot_eager" (no dipendenze esterne)
TORCH_COMPILE_BACKEND = "aot_eager"

# ===== DataLoader =====

# Numero di worker per il caricamento dati (0 = main process)
# Su CPU con dati gia' in memoria, 0 e' ottimale (evita overhead IPC)
NUM_WORKERS = 0

# Device (gpu o cpu) - determinato al momento dell'uso
def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = get_device()
