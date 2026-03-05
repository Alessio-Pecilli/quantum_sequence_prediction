# ===== Configurazione Modello =====

# Dimensione dello spazio di embedding (2^n per n qubit)
# Per 10 qubit (2048 feature reali) → 320 riduce la compressione a ~6.4:1
D_MODEL = 128

# Numero di teste di attenzione nel Transformer
# 320 / 8 = 40 dim per testa (cattura pattern temporali diversi)
NUM_HEADS = 4

# Numero di layer del Transformer Encoder
# 6 layer per dinamiche quantistiche complesse con residual connections
NUM_LAYERS = 4

# Dimensione hidden del Feed-Forward Network nel Transformer
# Rapporto standard 4× d_model (Vaswani et al.)
DIM_FEEDFORWARD = 512

# Dropout nel Transformer (regolarizzazione)
# Ridotto: il modello era in underfitting (gap train-test piccolo, fidelity bassa)
DROPOUT = 0.05

# ===== Configurazione Sistema Quantistico =====

# Numero di qubit del sistema
N_QUBITS = 6

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
# Aumentato per migliore generalizzazione e diversita' del dataset
B_TRAIN = 100
# Numero di stati iniziali per Hamiltoniana (S)
S_TRAIN = 30

# --- Test ---
# Nuove Hamiltoniane e nuovi stati (generati indipendentemente dal training)
B_TEST = 20
S_TEST = 20

# Lunghezza della sequenza temporale
SEQ_LEN = 100

# Dimensione del batch (su CPU batch grandi riducono l'overhead per iterazione)
BATCH_SIZE = 32

# Numero di epoche di addestramento
EPOCHS = 200

# ===== Configurazione Ottimizzatore =====

# Learning rate iniziale (peak dopo warmup)
# Aumentato: il modello era in underfitting, serve piu' spinta
LEARNING_RATE = 8e-4

# Weight decay per AdamW (regolarizzazione L2)
# Ridotto: meno regolarizzazione per contrastare l'underfitting
WEIGHT_DECAY = 1e-4

# Gradient clipping (max norm) — stabilizza il training dei Transformer
GRAD_CLIP_MAX_NORM = 1.0

# ===== Configurazione Learning Rate Scheduler =====

# Warmup: numero di epoche con rampa lineare da 0 → LEARNING_RATE
# 5 ep. per avviare rapidamente l'apprendimento
LR_WARMUP_EPOCHS = 5

# Tipo di scheduler dopo il warmup:
#   "cosine"            → Cosine Annealing fino a LR_MIN
#   "plateau"           → ReduceLROnPlateau (reattivo alla loss)
#   "cosine+plateau"    → Cosine Annealing + fallback ReduceLROnPlateau
# Solo cosine: plateau puo' ridurre il LR troppo presto causando stagnazione
LR_SCHEDULER_TYPE = "cosine"

# LR minimo raggiungibile (floor per qualsiasi scheduler)
LR_MIN = 1e-5

# ReduceLROnPlateau: patience (epoche senza miglioramento prima di ridurre LR)
LR_PLATEAU_PATIENCE = 15

# ReduceLROnPlateau: fattore di riduzione (new_lr = lr * factor)
LR_PLATEAU_FACTOR = 0.5

# ===== Configurazione Early Stopping =====

# Attiva/disattiva early stopping
EARLY_STOPPING_ENABLED = True

# Metrica da monitorare ("test_loss" o "test_fidelity")
# Loss e' piu' stabile della fidelity (meno rumorosa) per il criterio di stop
EARLY_STOPPING_METRIC = "test_loss"

# Patience: epoche senza miglioramento prima di fermarsi
# 60 ep. per dare tempo al cosine schedule di esplorare
EARLY_STOPPING_PATIENCE = 60

# Delta minimo per considerare un miglioramento significativo
EARLY_STOPPING_MIN_DELTA = 1e-6

# ===== Configurazione Checkpointing =====

# Salva il miglior modello durante il training
SAVE_BEST_MODEL = True

# Path per il checkpoint del miglior modello
BEST_MODEL_PATH = "results/best_model.pt"

# Salva checkpoint periodici ogni N epoche (0 = disabilitato)
CHECKPOINT_EVERY_N_EPOCHS = 0

# ===== Configurazione Resume Training =====

# Se True, PROVA a riprendere il training dall'ultimo checkpoint salvato.
# Il sistema verifica automaticamente la compatibilità dell'architettura:
#   - Se i parametri salvati sono compatibili (stessa architettura) → riprende
#   - Se NON sono compatibili (es. d_model diverso, n_qubits diverso) → parte da zero
# Se False, parte SEMPRE da zero ignorando qualsiasi checkpoint esistente.
RESUME_TRAINING = True

# Path del checkpoint di fallback (salvato automaticamente a ogni epoca)
LAST_CHECKPOINT_PATH = "results/last_checkpoint.pt"

# ===== Configurazione EMA (Exponential Moving Average) =====

# Attiva EMA dei pesi del modello per una valutazione più stabile
EMA_ENABLED = True

# Decay factor per EMA (tipicamente 0.999 - 0.9999)
# Piu' alto per media piu' liscia con training piu' lungo
EMA_DECAY = 0.9995

# ===== Mixed Precision Training =====

# Usa AMP (Automatic Mixed Precision) se disponibile su CUDA
AMP_ENABLED = True

# ===== Memory Management =====

# Modalita' conservativa: riduce i picchi di memoria a costo di throughput.
MEMORY_SAFE_MODE = True

# Micro-batch reale processato dal modello per volta.
# 0 = usa direttamente BATCH_SIZE senza split.
MICRO_BATCH_SIZE = 8

# Cleanup periodico Python GC durante train/eval (step).
# 0 = disabilitato.
GC_COLLECT_EVERY_N_STEPS = 25

# Svuota la cache CUDA periodicamente (utile su sessioni lunghe).
# 0 = disabilitato.
CUDA_EMPTY_CACHE_EVERY_N_STEPS = 50

# ===== Gradient Accumulation =====

# Numero di step di accumulo prima di aggiornare i pesi
# Simula un batch_size effettivo = BATCH_SIZE * GRAD_ACCUMULATION_STEPS
# Con BATCH_SIZE=32, effettivo=96 per gradienti piu' stabili
GRAD_ACCUMULATION_STEPS = 3

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

# Pin memory accelera il trasferimento CPU->GPU quando il device e' CUDA.
PIN_MEMORY = True

# Limite massimo di campioni da conservare per grafici distribuzionali.
# Serve a evitare crescita memoria durante valutazione finale.
MAX_FIDELITY_PLOT_SAMPLES = 4096

# Device (gpu o cpu) - determinato al momento dell'uso
def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = get_device()
