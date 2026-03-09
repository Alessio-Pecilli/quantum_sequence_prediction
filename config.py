import os
import math

# ===== Funzioni di Utility per Env Vars =====

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"Env var {name} deve essere un intero, ricevuto: {raw!r}") from e

def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower()

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Env var {name} deve essere booleana, ricevuto: {raw!r}")


# ===== Configurazione Sistema Quantistico =====

# Numero di qubit del sistema.
N_QUBITS = _env_int("QSP_N_QUBITS", 4)
DIM_2N = 2**N_QUBITS

# ===== Configurazione Modello (Custom per CPU Test) =====

D_MODEL = _env_int("QSP_D_MODEL", 128)
NUM_HEADS = _env_int("QSP_NUM_HEADS", 4)
NUM_LAYERS = _env_int("QSP_NUM_LAYERS", 2)
# Riduciamo il feedforward per alleggerire la CPU
DIM_FEEDFORWARD = _env_int("QSP_DIM_FEEDFORWARD", D_MODEL * 2)

if D_MODEL <= 0 or NUM_HEADS <= 0 or D_MODEL % NUM_HEADS != 0:
    raise ValueError("Configurazione D_MODEL/NUM_HEADS non valida.")

# Dropout a 0 per facilitare la convergenza sul dataset piccolo
DROPOUT = 0.0

# Parametri dell'Hamiltoniana TFIM (Range aperti per generalizzazione)
J_RANGE = (0.5, 1.5)
G_RANGE = (0.5, 1.5)

DT = 0.1
ROPE_BASE = 10000.0
HAMILTONIAN_TYPE = "TFIM"

# ===== Configurazione Dataset (3 Hamiltoniane per CPU) =====

B_TRAIN = _env_int("QSP_B_TRAIN", 3)
S_TRAIN = _env_int("QSP_S_TRAIN", 20)

B_TEST = _env_int("QSP_B_TEST", 2)
S_TEST = _env_int("QSP_S_TEST", 5)

# Sequenze corte per elaborazione veloce
SEQ_LEN = _env_int("QSP_SEQ_LEN", 40)
T1 = _env_int("QSP_T1", 5)
T2 = _env_int("QSP_T2", 35)

if T1 + T2 > SEQ_LEN + 1:
    raise ValueError(f"Configurazione non valida: T1+T2={T1 + T2} > SEQ_LEN+1={SEQ_LEN + 1}.")

TRAINING_MODE = _env_str("QSP_TRAINING_MODE", "rollout_window")

# Batch ridotto, ideale per CPU
BATCH_SIZE = _env_int("QSP_BATCH_SIZE", 20)
EPOCHS = _env_int("QSP_EPOCHS", 200)

# ===== Configurazione Ottimizzatore =====
# LR leggermente più alto per convergere in fretta
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP_MAX_NORM = 1.0

# ===== Configurazione Learning Rate Scheduler =====
LR_WARMUP_EPOCHS = 5
LR_SCHEDULER_TYPE = "cosine"
LR_MIN = 1e-5
LR_PLATEAU_PATIENCE = 15
LR_PLATEAU_FACTOR = 0.5

# ===== Configurazione Early Stopping e Checkpoint =====
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_METRIC = "test_loss"
EARLY_STOPPING_PATIENCE = 60
EARLY_STOPPING_MIN_DELTA = 1e-6

SAVE_BEST_MODEL = True
BEST_MODEL_PATH = "results/best_model.pt"
# Disattiviamo i checkpoint continui per non usurare il disco / rallentare l'I/O
CHECKPOINT_EVERY_N_EPOCHS = 0
RESUME_TRAINING = False
LAST_CHECKPOINT_PATH = "results/last_checkpoint.pt"

# ===== Configurazione EMA e Precisione =====
EMA_ENABLED = False
AMP_ENABLED = False

# ===== Memory Management =====
MEMORY_SAFE_MODE = True
MICRO_BATCH_SIZE = 0
GC_COLLECT_EVERY_N_STEPS = 50
CUDA_EMPTY_CACHE_EVERY_N_STEPS = 0
GRAD_ACCUMULATION_STEPS = 1

# ===== Compilazione e DataLoader =====
TORCH_COMPILE = False
TORCH_COMPILE_BACKEND = "inductor"

NUM_WORKERS = 0
PIN_MEMORY = False
MAX_FIDELITY_PLOT_SAMPLES = 60

# ===== Plot Osservabili =====
OBSERVABLE_PLOTS_ENABLED = True
OBSERVABLE_PLOT_SAMPLES = _env_int("QSP_OBS_PLOT_SAMPLES", 5)

# ===== Logging / Diagnostica =====
VERBOSE_STARTUP_LOGS = False
DATASET_LOG_EVERY_N_HAMILTONIANS = 1
TEMPORAL_LOG_EXAMPLES = 2
TRAIN_LOG_EVERY_N_STEPS = 50
EVAL_LOG_EVERY_N_STEPS = 50
LOG_BATCH_STATS = False
EMA_DECAY = 0.9995
LOG_MEMORY_STATS = False
SYNC_CUDA_TIMINGS = False

# ===== Device e Auto-tuning =====
def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = get_device()

# --- Stampa di Recap della configurazione ---
print(f"=====================================================")
print(f" CPU-FAST TEST CONFIG (3 HAMILTONIANE) ")
print(f"=====================================================")
print(f" Qubits: {N_QUBITS} | Dim Hilbert: {DIM_2N//2}")
print(f" Dataset: {B_TRAIN} Hamiltoniane x {S_TRAIN} Stati")
print(f" Transformer: {NUM_LAYERS} Layer, {NUM_HEADS} Heads, d={D_MODEL}")
print(f" Device: {DEVICE.upper()}")
print(f"=====================================================")