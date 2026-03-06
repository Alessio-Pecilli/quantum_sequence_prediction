# ===== Configurazione Modello =====

# Nota: alcuni parametri (es. SEQ_LEN, T1, T2) possono essere sovrascritti via env var.
# Convenzione:
#   - QSP_SEQ_LEN      -> lunghezza totale della sequenza generata dal dataset (max_seq_len del modello)
#   - QSP_N_QUBITS     -> numero di qubit del sistema
#   - QSP_T1           -> lookback window / context length (numero di stati osservati)
#   - QSP_T2           -> forecast horizon (numero di step futuri da predire in rollout)
#   - QSP_B_TRAIN      -> n. Hamiltoniane training
#   - QSP_S_TRAIN      -> n. stati iniziali per Hamiltoniana (train)
#   - QSP_B_TEST       -> n. Hamiltoniane test
#   - QSP_S_TEST       -> n. stati iniziali per Hamiltoniana (test)
#   - QSP_BATCH_SIZE   -> batch size
#   - QSP_EPOCHS       -> epoche training
#   - QSP_OBS_PLOT_SAMPLES -> n. traiettorie per plot osservabili
import os


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"Env var {name} deve essere un intero, ricevuto: {raw!r}") from e

# Dimensione dello spazio di embedding (2^n per n qubit)
# Per 10 qubit (2048 feature reali) → 512 riduce la compressione a ~4:1
D_MODEL = 512

# Numero di teste di attenzione nel Transformer
# 512 / 8 = 64 dim per testa (cattura pattern temporali diversi)
NUM_HEADS = 8

# Numero di layer del Transformer Encoder
# 6 layer per dinamiche quantistiche complesse con residual connections
NUM_LAYERS = 6

# Dimensione hidden del Feed-Forward Network nel Transformer
# Rapporto standard 4× d_model (Vaswani et al.)
DIM_FEEDFORWARD = 2048

# Dropout nel Transformer (regolarizzazione)
# Ridotto: il modello era in underfitting (gap train-test piccolo, fidelity bassa)
DROPOUT = 0.05

# ===== Configurazione Sistema Quantistico =====

# Numero di qubit del sistema
# Default ridotto a 6 per test veloci; usa env QSP_N_QUBITS per cambiare.
N_QUBITS = _env_int("QSP_N_QUBITS", 6)

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
B_TRAIN = _env_int("QSP_B_TRAIN", 100)
# Numero di stati iniziali per Hamiltoniana (S)
S_TRAIN = _env_int("QSP_S_TRAIN", 30)

# --- Test ---
# Nuove Hamiltoniane e nuovi stati (generati indipendentemente dal training)
B_TEST = _env_int("QSP_B_TEST", 20)
S_TEST = _env_int("QSP_S_TEST", 20)

# Lunghezza totale della sequenza temporale generata (max seq len del modello).
# Nota: per valutazioni "rollout" serve che SEQ_LEN + 1 >= T1 + T2 (perche' nel dataset ci sono SEQ_LEN+1 stati).
SEQ_LEN = _env_int("QSP_SEQ_LEN", 100)
if SEQ_LEN < 2:
    raise ValueError(f"SEQ_LEN deve essere >= 2, ricevuto: {SEQ_LEN}")

# Struttura finestra temporale per rollout / testing:
#   T1: lookback window (context)
#   T2: forecast horizon (step futuri da predire autoregressivamente)
_t1_default = min(10, SEQ_LEN - 1)
T1 = _env_int("QSP_T1", _t1_default)
_t2_default = max(1, SEQ_LEN - T1)
T2 = _env_int("QSP_T2", _t2_default)

if T1 < 1:
    raise ValueError(f"T1 deve essere >= 1, ricevuto: {T1}")
if T2 < 1:
    raise ValueError(f"T2 deve essere >= 1, ricevuto: {T2}")
if T1 + T2 > SEQ_LEN + 1:
    raise ValueError(
        f"Configurazione non valida: T1+T2={T1 + T2} > SEQ_LEN+1={SEQ_LEN + 1}. "
        f"Aumenta SEQ_LEN (env QSP_SEQ_LEN) o riduci T1/T2 (env QSP_T1/QSP_T2)."
    )

# Dimensione del batch (su CPU batch grandi riducono l'overhead per iterazione)
BATCH_SIZE = _env_int("QSP_BATCH_SIZE", 32)

# Numero di epoche di addestramento
EPOCHS = _env_int("QSP_EPOCHS", 200)

# ===== Configurazione Ottimizzatore =====

# Learning rate iniziale (peak dopo warmup)
# Alto per compensare gradienti deboli della fidelity loss in dim=1024
LEARNING_RATE = 2e-3

# Weight decay per AdamW (regolarizzazione L2)
# Ridotto: meno regolarizzazione per contrastare l'underfitting
WEIGHT_DECAY = 1e-4

# Gradient clipping (max norm) — stabilizza il training dei Transformer
GRAD_CLIP_MAX_NORM = 1.0

# ===== Configurazione Learning Rate Scheduler =====

# Warmup: numero di epoche con rampa lineare da 0 → LEARNING_RATE
# 10 ep. con LR alto per stabilizzare l'inizio
LR_WARMUP_EPOCHS = 10

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
RESUME_TRAINING = False

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
MICRO_BATCH_SIZE = 4

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

# ===== Plot Osservabili (rollout) =====
# Se True, il training pipeline genera anche grafici di osservabili fisiche
# (magnetizzazioni e correlazioni) confrontando rollout del modello vs traiettoria esatta.
OBSERVABLE_PLOTS_ENABLED = True

# Numero di traiettorie (dal training set) usate per stimare media/std degli osservabili.
# Se <= 0 usa tutto il training set (puo' essere lento).
OBSERVABLE_PLOT_SAMPLES = _env_int("QSP_OBS_PLOT_SAMPLES", 256)

# Device (gpu o cpu) - determinato al momento dell'uso
def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = get_device()
