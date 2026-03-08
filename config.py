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


def _default_model_hyperparams(n_qubits: int) -> tuple[int, int, int]:
    if n_qubits <= 4:
        return 128, 4, 4
    if n_qubits <= 6:
        return 256, 8, 5
    return 512, 8, 6


# Dimensione dello spazio di embedding (2^n per n qubit)
# Per 10 qubit (2048 feature reali) â†’ 512 riduce la compressione a ~4:1
# ===== Configurazione Sistema Quantistico =====

# Numero di qubit del sistema.
# Default impostato a 4: spazio di Hilbert di dimensione 16.
N_QUBITS = _env_int("QSP_N_QUBITS", 4)
DIM_2N = 2**N_QUBITS
# ===== Configurazione Modello =====

_default_d_model, _default_num_heads, _default_num_layers = _default_model_hyperparams(N_QUBITS)
# Per 4 qubit (16 ampiezze complesse -> 32 feature reali) 128 e' gia' capiente.
D_MODEL = _env_int("QSP_D_MODEL", _default_d_model)

# Numero di teste di attenzione nel Transformer
# 512 / 8 = 64 dim per testa (cattura pattern temporali diversi)
NUM_HEADS = _env_int("QSP_NUM_HEADS", _default_num_heads)

# Numero di layer del Transformer Encoder
# 6 layer per dinamiche quantistiche complesse con residual connections
NUM_LAYERS = _env_int("QSP_NUM_LAYERS", _default_num_layers)

# Dimensione hidden del Feed-Forward Network nel Transformer
# Rapporto standard 4Ã— d_model (Vaswani et al.)
DIM_FEEDFORWARD = _env_int("QSP_DIM_FEEDFORWARD", D_MODEL * 4)

if D_MODEL <= 0:
    raise ValueError(f"D_MODEL deve essere > 0, ricevuto: {D_MODEL}")
if NUM_HEADS <= 0:
    raise ValueError(f"NUM_HEADS deve essere > 0, ricevuto: {NUM_HEADS}")
if D_MODEL % NUM_HEADS != 0:
    raise ValueError(
        f"D_MODEL={D_MODEL} deve essere divisibile per NUM_HEADS={NUM_HEADS}"
    )

# Dropout nel Transformer (regolarizzazione)
# Ridotto: il modello era in underfitting (gap train-test piccolo, fidelity bassa)
DROPOUT = 0.05

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

# Modalita' di training:
#   - "rollout_window": finestre di lunghezza T1 -> next-step, allineato al rollout autoregressivo
#   - "full_sequence": teacher forcing sull'intera sequenza shiftata
TRAINING_MODE = _env_str("QSP_TRAINING_MODE", "rollout_window")
if TRAINING_MODE not in {"rollout_window", "full_sequence"}:
    raise ValueError(
        f"QSP_TRAINING_MODE non supportato: {TRAINING_MODE!r}. "
        "Valori ammessi: 'rollout_window', 'full_sequence'."
    )

# Dimensione del batch. T4 (16GB) regge batch=64 con d_model=512, seq_len=100.
# Piu' grande = meno iterazioni per epoca = piu' veloce.
BATCH_SIZE = _env_int("QSP_BATCH_SIZE", 64)

# Numero di epoche di addestramento
EPOCHS = _env_int("QSP_EPOCHS", 200)

# ===== Configurazione Ottimizzatore =====

# Learning rate iniziale (peak dopo warmup)
# Loss = infidelity (1 - F), range [0, 1]. LR standard per Transformer.
LEARNING_RATE = 1e-3

# Weight decay per AdamW (regolarizzazione L2)
# Ridotto: meno regolarizzazione per contrastare l'underfitting
WEIGHT_DECAY = 1e-4

# Gradient clipping (max norm) â€” stabilizza il training dei Transformer
GRAD_CLIP_MAX_NORM = 1.0

# ===== Configurazione Learning Rate Scheduler =====

# Warmup: numero di epoche con rampa lineare da 0 â†’ LEARNING_RATE
# 10 ep. con LR alto per stabilizzare l'inizio
LR_WARMUP_EPOCHS = 10

# Tipo di scheduler dopo il warmup:
#   "cosine"            â†’ Cosine Annealing fino a LR_MIN
#   "plateau"           â†’ ReduceLROnPlateau (reattivo alla loss)
#   "cosine+plateau"    â†’ Cosine Annealing + fallback ReduceLROnPlateau
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
# Colab puo' disconnettere in qualsiasi momento: salvare spesso!
CHECKPOINT_EVERY_N_EPOCHS = 5

# ===== Configurazione Resume Training =====

# Se True, PROVA a riprendere il training dall'ultimo checkpoint salvato.
# Il sistema verifica automaticamente la compatibilitÃ  dell'architettura:
#   - Se i parametri salvati sono compatibili (stessa architettura) â†’ riprende
#   - Se NON sono compatibili (es. d_model diverso, n_qubits diverso) â†’ parte da zero
# Se False, parte SEMPRE da zero ignorando qualsiasi checkpoint esistente.
# Su Colab: True per riprendere dopo disconnessioni.
RESUME_TRAINING = True

# Path del checkpoint di fallback (salvato automaticamente a ogni epoca)
LAST_CHECKPOINT_PATH = "results/last_checkpoint.pt"

# ===== Configurazione EMA (Exponential Moving Average) =====

# Attiva EMA dei pesi del modello per una valutazione piÃ¹ stabile
EMA_ENABLED = True

# Decay factor per EMA (tipicamente 0.999 - 0.9999)
# Piu' alto per media piu' liscia con training piu' lungo
EMA_DECAY = 0.9995

# ===== Mixed Precision Training =====

# Usa AMP (Automatic Mixed Precision) se disponibile su CUDA
AMP_ENABLED = True

# ===== Memory Management =====

# Modalita' conservativa: riduce i picchi di memoria a costo di throughput.
MEMORY_SAFE_MODE = False

# Micro-batch reale processato dal modello per volta.
# 0 = usa direttamente BATCH_SIZE senza split.
MICRO_BATCH_SIZE = 0

# Cleanup periodico Python GC durante train/eval (step).
# Previene memory creep su sessioni Colab lunghe.
GC_COLLECT_EVERY_N_STEPS = 50

# Svuota la cache CUDA periodicamente (utile su sessioni lunghe).
# Evita OOM su T4 16GB.
CUDA_EMPTY_CACHE_EVERY_N_STEPS = 100

# ===== Gradient Accumulation =====

# Numero di step di accumulo prima di aggiornare i pesi
# Simula un batch_size effettivo = BATCH_SIZE * GRAD_ACCUMULATION_STEPS
# Con BATCH_SIZE=64 su T4 non serve accumulare: 1 = zero overhead.
GRAD_ACCUMULATION_STEPS = 1

# ===== Compilazione Modello =====

# Usa torch.compile() per ottimizzare il modello (PyTorch 2.0+)
# Riduce overhead Python e fonde operazioni. Prima epoca piu' lenta (compilazione).
# NOTA: su CPU viene disabilitato automaticamente (vedi auto-tuning in fondo).
TORCH_COMPILE = _env_bool("QSP_TORCH_COMPILE", N_QUBITS > 4)

# Backend per torch.compile: "inductor" (richiede compilatore C++), "aot_eager" (no dipendenze esterne)
TORCH_COMPILE_BACKEND = "inductor"

# ===== DataLoader =====

# Numero di worker per il caricamento dati (0 = main process)
# Su CPU con dati gia' in memoria, 0 e' ottimale (evita overhead IPC)
NUM_WORKERS = _env_int("QSP_NUM_WORKERS", 0 if os.name == "nt" else 2)

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

# ===== Logging / Diagnostica =====

# Log dettagliati su semantica temporale, dataset e tempi interni del trainer.
VERBOSE_STARTUP_LOGS = _env_bool("QSP_VERBOSE_STARTUP_LOGS", True)

# Progress della generazione dataset: 1 = una riga per Hamiltoniana.
DATASET_LOG_EVERY_N_HAMILTONIANS = _env_int("QSP_DATASET_LOG_EVERY_N_HAMILTONIANS", 1)

# Numero di esempi temporali da stampare per spiegare la mappa input -> target.
TEMPORAL_LOG_EXAMPLES = _env_int("QSP_TEMPORAL_LOG_EXAMPLES", 4)

# Log step-level nel training/eval.
TRAIN_LOG_EVERY_N_STEPS = _env_int("QSP_TRAIN_LOG_EVERY_N_STEPS", 10)
EVAL_LOG_EVERY_N_STEPS = _env_int("QSP_EVAL_LOG_EVERY_N_STEPS", 10)

# Logga shape, dtype e statistiche di norma del primo batch di ogni fase.
LOG_BATCH_STATS = _env_bool("QSP_LOG_BATCH_STATS", True)

# Logga memoria CUDA nei punti chiave del training.
LOG_MEMORY_STATS = _env_bool("QSP_LOG_MEMORY_STATS", True)

# Sincronizza CUDA quando si misurano i tempi loggati.
# Piu' accurato, ma aggiunge un po' di overhead nei batch stampati.
SYNC_CUDA_TIMINGS = _env_bool("QSP_SYNC_CUDA_TIMINGS", True)


# Device (gpu o cpu) - determinato al momento dell'uso
def get_device():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = get_device()

# ===== Auto-tuning in base al device =====
# Su CPU molte feature GPU-only causano solo overhead e memory bloat.
if DEVICE == "cpu":
    TORCH_COMPILE = False       # Inductor usa ~1GB RAM extra senza beneficio su CPU
    EMA_ENABLED = False         # Risparmia una copia completa dei pesi (~120MB)
    AMP_ENABLED = False         # AMP non ha senso su CPU
    NUM_WORKERS = 0             # Workers su Windows (spawn) duplicano il dataset in RAM
    PIN_MEMORY = False          # Solo utile per trasferimenti CPU->GPU
    MEMORY_SAFE_MODE = True     # Abilita cleanup aggressivo
    GC_COLLECT_EVERY_N_STEPS = 20
    CUDA_EMPTY_CACHE_EVERY_N_STEPS = 0
    if BATCH_SIZE > 16:
        BATCH_SIZE = 16         # Riduce picco attivazioni forward/backward
    if GRAD_ACCUMULATION_STEPS < 4:
        GRAD_ACCUMULATION_STEPS = 4  # Compensa batch piccolo: effettivo = 16*4 = 64
