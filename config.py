import os
from pathlib import Path

import torch


_ACTIVE_ENV_OVERRIDES: dict[str, dict[str, object]] = {}


def _track_env_override(name: str, raw: str, parsed_value):
    _ACTIVE_ENV_OVERRIDES[name] = {
        "raw": raw,
        "value": parsed_value,
    }


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Env var {name} deve essere un intero, ricevuto: {raw!r}") from exc
    _track_env_override(name, raw, value)
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Env var {name} deve essere un float, ricevuto: {raw!r}") from exc
    _track_env_override(name, raw, value)
    return value


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    _track_env_override(name, raw, value)
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        _track_env_override(name, raw, True)
        return True
    if value in {"0", "false", "no", "n", "off"}:
        _track_env_override(name, raw, False)
        return False
    raise ValueError(f"Env var {name} deve essere booleana, ricevuto: {raw!r}")


def get_active_env_overrides() -> dict[str, dict[str, object]]:
    return dict(_ACTIVE_ENV_OVERRIDES)


PROJECT_ROOT = Path(__file__).resolve().parent
# Directory risultati separata per la run "paper-like" (logamp+phase),
# così non sovrascriviamo i risultati del baseline.
RESULTS_DIR = PROJECT_ROOT / "results_paper_logamp_phase"
CHECKPOINT_PATH = RESULTS_DIR / "best_model.pt"
LAST_CHECKPOINT_PATH = RESULTS_DIR / "last_checkpoint.pt"
SUMMARY_PATH = RESULTS_DIR / "run_summary.json"
FIDELITY_PLOT_PATH = RESULTS_DIR / "fidelity_vs_time.png"
TRAINING_CURVES_PATH = RESULTS_DIR / "training_curves.png"
OBSERVABLES_PLOT_PATH = RESULTS_DIR / "observables_vs_time.png"
OBSERVABLES_TRAIN_PLOT_PATH = RESULTS_DIR / "observables_train_vs_rollout.png"
OBSERVABLES_TEST_PLOT_PATH = RESULTS_DIR / "observables_test_vs_rollout.png"


SEED = _env_int("QSP_SEED", 7)


N_QUBITS = _env_int("QSP_N_QUBITS", 4)
if N_QUBITS < 1:
    raise ValueError(f"N_QUBITS deve essere >= 1, ricevuto: {N_QUBITS}")

DIM_2N = 2 ** N_QUBITS


def _default_by_qubits(defaults: dict[int, int], fallback: int) -> int:
    return int(defaults.get(int(N_QUBITS), fallback))

# Numero totale di stati in ogni traiettoria, incluso psi(0).
# Default spinto per esplorare rollout piu' lunghi e verificare se, dopo il
# transiente iniziale, fidelity e osservabili tendono a riassestarsi.
NUM_STATES = _env_int("QSP_NUM_STATES", 64)
if NUM_STATES < 2:
    raise ValueError(f"NUM_STATES deve essere >= 2, ricevuto: {NUM_STATES}")

# Alias retrocompatibile: numero di transizioni supervisionate.
SEQ_LEN = NUM_STATES - 1


def _is_long_horizon() -> bool:
    # Run con rollout lungo: >=60 stati (59+ predizioni).
    return int(NUM_STATES) >= 60


TRAIN_SEQUENCES = _env_int("QSP_TRAIN_SEQUENCES", 1024)
TEST_SEQUENCES = _env_int("QSP_TEST_SEQUENCES", 256)
if TRAIN_SEQUENCES < 1 or TEST_SEQUENCES < 1:
    raise ValueError("TRAIN_SEQUENCES e TEST_SEQUENCES devono essere >= 1.")

# Alias compatibili con il codice precedente.
B_TRAIN = 1
S_TRAIN = TRAIN_SEQUENCES
B_TEST = 1
S_TEST = TEST_SEQUENCES


HAMILTONIAN_TYPE = "TFIM"
BOUNDARY_CONDITION = _env_str("QSP_BOUNDARY_CONDITION", "open")
if BOUNDARY_CONDITION != "open":
    raise ValueError(
        f"BOUNDARY_CONDITION={BOUNDARY_CONDITION!r} non supportata. "
        "Attualmente e' implementata solo la catena open-boundary."
    )

COUPLING_MEAN = _env_float("QSP_COUPLING_MEAN", 1.0)
COUPLING_STD = _env_float("QSP_COUPLING_STD", 1.0)
FIELD_STRENGTH = _env_float("QSP_FIELD_STRENGTH", 1.0)
TIME_STEP = _env_float("QSP_TIME_STEP", 1.0)
DT = TIME_STEP
if COUPLING_STD < 0.0:
    raise ValueError(f"COUPLING_STD deve essere >= 0, ricevuto: {COUPLING_STD}")
if TIME_STEP <= 0.0:
    raise ValueError(f"TIME_STEP deve essere > 0, ricevuto: {TIME_STEP}")

EVOLUTION_BACKEND = _env_str("QSP_EVOLUTION_BACKEND", "auto")
if EVOLUTION_BACKEND not in {"auto", "exact_diag", "matrix_exp"}:
    raise ValueError(
        f"EVOLUTION_BACKEND={EVOLUTION_BACKEND!r} non valido. "
        "Valori ammessi: auto, exact_diag, matrix_exp."
    )
EXACT_DIAG_MAX_DIM = _env_int("QSP_EXACT_DIAG_MAX_DIM", 256)
if EXACT_DIAG_MAX_DIM < 2:
    raise ValueError(f"EXACT_DIAG_MAX_DIM deve essere >= 2, ricevuto: {EXACT_DIAG_MAX_DIM}")


INITIAL_STATE_FAMILY = _env_str("QSP_INITIAL_STATE_FAMILY", "xyz_basis")
if INITIAL_STATE_FAMILY not in {"x_basis", "xyz_basis"}:
    raise ValueError(
        f"INITIAL_STATE_FAMILY={INITIAL_STATE_FAMILY!r} non valida. "
        "Valori ammessi: x_basis, xyz_basis."
    )
INITIAL_STATE_SAMPLE_WITH_REPLACEMENT = _env_bool("QSP_INITIAL_STATE_SAMPLE_WITH_REPLACEMENT", True)
X_BASIS_SAMPLE_WITH_REPLACEMENT = INITIAL_STATE_SAMPLE_WITH_REPLACEMENT


D_MODEL = _env_int("QSP_D_MODEL", _default_by_qubits({4: 64, 6: 256}, 256))
NUM_HEADS = _env_int("QSP_NUM_HEADS", _default_by_qubits({4: 4, 6: 4}, 4))
NUM_LAYERS = _env_int("QSP_NUM_LAYERS", _default_by_qubits({4: 2, 6: 3}, 3))
DIM_FEEDFORWARD = _env_int("QSP_DIM_FEEDFORWARD", _default_by_qubits({4: 256, 6: 1024}, 1024))
DROPOUT = _env_float("QSP_DROPOUT", 0.0)
if D_MODEL <= 0 or NUM_HEADS <= 0 or NUM_LAYERS <= 0 or DIM_FEEDFORWARD <= 0:
    raise ValueError("D_MODEL, NUM_HEADS, NUM_LAYERS e DIM_FEEDFORWARD devono essere > 0.")
if D_MODEL % NUM_HEADS != 0:
    raise ValueError("D_MODEL deve essere divisibile per NUM_HEADS.")


BATCH_SIZE = _env_int(
    "QSP_BATCH_SIZE",
    _default_by_qubits({4: 48 if _is_long_horizon() else 64, 6: 24}, 24),
)
EPOCHS = _env_int(
    "QSP_EPOCHS",
    _default_by_qubits({4: 420 if _is_long_horizon() else 180, 6: 220}, 220),
)
LEARNING_RATE = _env_float("QSP_LEARNING_RATE", 1e-4)
WEIGHT_DECAY = _env_float("QSP_WEIGHT_DECAY", 1e-4)
GRAD_CLIP_MAX_NORM = _env_float("QSP_GRAD_CLIP_MAX_NORM", 1.0)
LOG_FIDELITY_EPS = _env_float("QSP_LOG_FIDELITY_EPS", 1e-8)

# Orizzonte multi-step del nuovo training autoregressivo.
# Lo alziamo in modo netto per allenare il modello su dipendenze temporali piu'
# lunghe prima di valutare rollout estesi.
MULTISTEP_H = _env_int("QSP_MULTISTEP_H", min(20, int(SEQ_LEN)))
if not (1 <= MULTISTEP_H <= SEQ_LEN):
    raise ValueError(
        f"MULTISTEP_H deve stare in [1, SEQ_LEN={SEQ_LEN}], ricevuto: {MULTISTEP_H}"
    )

# Numero di passi che continuano a usare il contesto corretto prima del passaggio
# al contesto predetto. Lo aumentiamo per rendere meno fragile l'innesco del
# rollout quando si passa a orizzonti piu' lunghi.
MULTISTEP_TEACHER_FORCING_STEPS = _env_int(
    "QSP_MULTISTEP_TEACHER_FORCING_STEPS",
    min(6, int(MULTISTEP_H)),
)
if MULTISTEP_TEACHER_FORCING_STEPS < 0:
    raise ValueError(
        "MULTISTEP_TEACHER_FORCING_STEPS deve essere >= 0, "
        f"ricevuto: {MULTISTEP_TEACHER_FORCING_STEPS}"
    )
MULTISTEP_TRAIN_VERBOSE = _env_bool("QSP_MULTISTEP_TRAIN_VERBOSE", False)

# Training multi-step adattivo:
# resta disponibile via env, ma di default lo lasciamo spento per privilegiare
# una baseline piu' stabile e interpretabile.
ADAPTIVE_MULTISTEP_ENABLED = _env_bool("QSP_ADAPTIVE_MULTISTEP_ENABLED", False)
ADAPTIVE_STATS_EMA = _env_float("QSP_ADAPTIVE_STATS_EMA", 0.70)
ADAPTIVE_WEIGHT_ALPHA = _env_float("QSP_ADAPTIVE_WEIGHT_ALPHA", 0.80)
ADAPTIVE_WEIGHT_MIN = _env_float("QSP_ADAPTIVE_WEIGHT_MIN", 1.0)
ADAPTIVE_WEIGHT_MAX = _env_float("QSP_ADAPTIVE_WEIGHT_MAX", 2.0)
ADAPTIVE_H_MIN = _env_int("QSP_ADAPTIVE_H_MIN", max(1, min(8, int(MULTISTEP_H))))
ADAPTIVE_H_MAX = _env_int("QSP_ADAPTIVE_H_MAX", int(MULTISTEP_H))
ADAPTIVE_TEACHER_MIN = _env_int("QSP_ADAPTIVE_TEACHER_MIN", 2)
ADAPTIVE_TEACHER_MAX = _env_int("QSP_ADAPTIVE_TEACHER_MAX", min(8, int(MULTISTEP_H)))
ADAPTIVE_H_LOSS_THRESHOLD = _env_float("QSP_ADAPTIVE_H_LOSS_THRESHOLD", 0.95)
ADAPTIVE_H_FIDELITY_THRESHOLD = _env_float("QSP_ADAPTIVE_H_FIDELITY_THRESHOLD", 0.58)
ADAPTIVE_TEACHER_LOSS_THRESHOLD = _env_float("QSP_ADAPTIVE_TEACHER_LOSS_THRESHOLD", 0.85)
ADAPTIVE_TEACHER_FIDELITY_THRESHOLD = _env_float("QSP_ADAPTIVE_TEACHER_FIDELITY_THRESHOLD", 0.62)

# Parametri legacy del vecchio rollout-aux training: mantenuti per retrocompatibilita'
# di configurazione, ma non piu' usati dal nuovo training multi-step.
SCHEDULED_SAMPLING_RAMP_EPOCHS = _env_int(
    "QSP_SCHEDULED_SAMPLING_RAMP_EPOCHS",
    80 if _is_long_horizon() else 40,
)
SCHEDULED_SAMPLING_MAX_PROB = _env_float(
    "QSP_SCHEDULED_SAMPLING_MAX_PROB",
    0.80 if _is_long_horizon() else 0.65,
)

# Rollout training più pesante e curriculum più rapido (arriva prima a rollout lunghi).
ROLLOUT_AUX_WEIGHT = _env_float("QSP_ROLLOUT_AUX_WEIGHT", 1.00)
ROLLOUT_CURRICULUM_EPOCHS = _env_int(
    "QSP_ROLLOUT_CURRICULUM_EPOCHS",
    90 if _is_long_horizon() else 30,
)

# Warmup usato SOLO nel rollout-training: piu' contesto vero per rendere meno
# traumatici i primissimi passi del rollout libero.
ROLLOUT_WARMUP_STATES = _env_int(
    "QSP_ROLLOUT_WARMUP_STATES",
    max(8, min(16, int(NUM_STATES) // 6)),
)
if BATCH_SIZE < 1 or EPOCHS < 1:
    raise ValueError("BATCH_SIZE e EPOCHS devono essere >= 1.")
if LEARNING_RATE <= 0.0:
    raise ValueError(f"LEARNING_RATE deve essere > 0, ricevuto: {LEARNING_RATE}")
if WEIGHT_DECAY < 0.0 or GRAD_CLIP_MAX_NORM < 0.0:
    raise ValueError("WEIGHT_DECAY e GRAD_CLIP_MAX_NORM devono essere >= 0.")
if not (0.0 < LOG_FIDELITY_EPS < 1.0):
    raise ValueError(f"LOG_FIDELITY_EPS deve stare in (0,1), ricevuto: {LOG_FIDELITY_EPS}")
if not (0.0 <= SCHEDULED_SAMPLING_MAX_PROB <= 1.0):
    raise ValueError(
        "SCHEDULED_SAMPLING_MAX_PROB deve stare in [0,1], "
        f"ricevuto: {SCHEDULED_SAMPLING_MAX_PROB}"
    )
if SCHEDULED_SAMPLING_RAMP_EPOCHS < 1:
    raise ValueError(
        f"SCHEDULED_SAMPLING_RAMP_EPOCHS deve essere >= 1, ricevuto: {SCHEDULED_SAMPLING_RAMP_EPOCHS}"
    )
if ROLLOUT_AUX_WEIGHT < 0.0:
    raise ValueError(f"ROLLOUT_AUX_WEIGHT deve essere >= 0, ricevuto: {ROLLOUT_AUX_WEIGHT}")
if ROLLOUT_CURRICULUM_EPOCHS < 1:
    raise ValueError(
        f"ROLLOUT_CURRICULUM_EPOCHS deve essere >= 1, ricevuto: {ROLLOUT_CURRICULUM_EPOCHS}"
    )
if ROLLOUT_WARMUP_STATES < 1 or ROLLOUT_WARMUP_STATES >= NUM_STATES:
    raise ValueError(
        "ROLLOUT_WARMUP_STATES deve stare in [1, NUM_STATES-1], "
        f"ricevuto: {ROLLOUT_WARMUP_STATES} con NUM_STATES={NUM_STATES}"
    )
if not (0.0 <= ADAPTIVE_STATS_EMA < 1.0):
    raise ValueError(
        f"ADAPTIVE_STATS_EMA deve stare in [0,1), ricevuto: {ADAPTIVE_STATS_EMA}"
    )
if ADAPTIVE_WEIGHT_ALPHA < 0.0:
    raise ValueError(
        f"ADAPTIVE_WEIGHT_ALPHA deve essere >= 0, ricevuto: {ADAPTIVE_WEIGHT_ALPHA}"
    )
if ADAPTIVE_WEIGHT_MIN <= 0.0 or ADAPTIVE_WEIGHT_MAX <= 0.0:
    raise ValueError("ADAPTIVE_WEIGHT_MIN e ADAPTIVE_WEIGHT_MAX devono essere > 0.")
if ADAPTIVE_WEIGHT_MIN > ADAPTIVE_WEIGHT_MAX:
    raise ValueError("ADAPTIVE_WEIGHT_MIN non puo' superare ADAPTIVE_WEIGHT_MAX.")
if not (1 <= ADAPTIVE_H_MIN <= ADAPTIVE_H_MAX <= SEQ_LEN):
    raise ValueError(
        "ADAPTIVE_H_MIN e ADAPTIVE_H_MAX devono stare in [1, SEQ_LEN] "
        f"con ADAPTIVE_H_MIN={ADAPTIVE_H_MIN}, ADAPTIVE_H_MAX={ADAPTIVE_H_MAX}, SEQ_LEN={SEQ_LEN}"
    )
if not (0 <= ADAPTIVE_TEACHER_MIN <= ADAPTIVE_TEACHER_MAX <= ADAPTIVE_H_MAX):
    raise ValueError(
        "ADAPTIVE_TEACHER_MIN e ADAPTIVE_TEACHER_MAX devono stare in [0, ADAPTIVE_H_MAX] "
        f"con ADAPTIVE_TEACHER_MIN={ADAPTIVE_TEACHER_MIN}, "
        f"ADAPTIVE_TEACHER_MAX={ADAPTIVE_TEACHER_MAX}, ADAPTIVE_H_MAX={ADAPTIVE_H_MAX}"
    )
if ADAPTIVE_H_LOSS_THRESHOLD < 0.0 or ADAPTIVE_TEACHER_LOSS_THRESHOLD < 0.0:
    raise ValueError("Le soglie di loss adattive devono essere >= 0.")
if not (0.0 <= ADAPTIVE_H_FIDELITY_THRESHOLD <= 1.0):
    raise ValueError(
        "ADAPTIVE_H_FIDELITY_THRESHOLD deve stare in [0,1], "
        f"ricevuto: {ADAPTIVE_H_FIDELITY_THRESHOLD}"
    )
if not (0.0 <= ADAPTIVE_TEACHER_FIDELITY_THRESHOLD <= 1.0):
    raise ValueError(
        "ADAPTIVE_TEACHER_FIDELITY_THRESHOLD deve stare in [0,1], "
        f"ricevuto: {ADAPTIVE_TEACHER_FIDELITY_THRESHOLD}"
    )


EXPOSURE_BIAS_GAP_THRESHOLD = _env_float("QSP_EXPOSURE_BIAS_GAP_THRESHOLD", 0.10)
EXPOSURE_BIAS_DROP_THRESHOLD = _env_float("QSP_EXPOSURE_BIAS_DROP_THRESHOLD", 0.10)
PARTIAL_WARMUP_STEPS = _env_str("QSP_PARTIAL_WARMUP_STEPS", "auto")
PLOT_DPI = _env_int("QSP_PLOT_DPI", 160)
OBSERVABLES_TEST_SEQUENCE_INDEX = _env_int("QSP_OBSERVABLES_TEST_SEQUENCE_INDEX", 0)
CLAMP_AUDIT_PRINT = _env_bool("QSP_CLAMP_AUDIT_PRINT", True)
CLAMP_AUDIT_MAX_SEQUENCES = _env_int("QSP_CLAMP_AUDIT_MAX_SEQUENCES", 3)
CLAMP_AUDIT_MAX_STATES = _env_int("QSP_CLAMP_AUDIT_MAX_STATES", min(6, int(NUM_STATES)))
CLAMP_AUDIT_PRINT_BITSTRINGS = _env_bool("QSP_CLAMP_AUDIT_PRINT_BITSTRINGS", True)
CLAMP_AUDIT_PRINT_COEFFS = _env_bool("QSP_CLAMP_AUDIT_PRINT_COEFFS", False)
if CLAMP_AUDIT_MAX_SEQUENCES < 0 or CLAMP_AUDIT_MAX_STATES < 0:
    raise ValueError("CLAMP_AUDIT_MAX_SEQUENCES e CLAMP_AUDIT_MAX_STATES devono essere >= 0.")
if CLAMP_AUDIT_MAX_STATES > NUM_STATES:
    raise ValueError(
        f"CLAMP_AUDIT_MAX_STATES non puo' superare NUM_STATES={NUM_STATES}, "
        f"ricevuto: {CLAMP_AUDIT_MAX_STATES}"
    )
if OBSERVABLES_TEST_SEQUENCE_INDEX < 0:
    raise ValueError(
        "OBSERVABLES_TEST_SEQUENCE_INDEX deve essere >= 0, "
        f"ricevuto: {OBSERVABLES_TEST_SEQUENCE_INDEX}"
    )


NUM_WORKERS = _env_int("QSP_NUM_WORKERS", 0)
PIN_MEMORY = _env_bool("QSP_PIN_MEMORY", False)
SAVE_MODEL = _env_bool("QSP_SAVE_MODEL", True)
# Se attivo, salta il training e ricalcola metriche/plot da best_model.pt.
EVAL_ONLY = _env_bool("QSP_EVAL_ONLY", False)
# Disattivo resume per la run "paper-like" (ripartiamo puliti).
AUTO_RESUME = _env_bool("QSP_AUTO_RESUME", False)
CHECKPOINT_EVERY_EPOCH = _env_int("QSP_CHECKPOINT_EVERY_EPOCH", 1)
if CHECKPOINT_EVERY_EPOCH < 1:
    raise ValueError(
        f"CHECKPOINT_EVERY_EPOCH deve essere >= 1, ricevuto: {CHECKPOINT_EVERY_EPOCH}"
    )
CHECKPOINT_EVERY_BATCH = _env_int("QSP_CHECKPOINT_EVERY_BATCH", 0)
if CHECKPOINT_EVERY_BATCH < 0:
    raise ValueError(
        f"CHECKPOINT_EVERY_BATCH deve essere >= 0, ricevuto: {CHECKPOINT_EVERY_BATCH}"
    )

EMPTY_CACHE_EVERY_EPOCH = _env_bool("QSP_EMPTY_CACHE_EVERY_EPOCH", True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parametrizzazione dell'output:
# - "direct_complex": la rete produce Re/Im direttamente, poi normalizziamo.
# - "logamp_phase": la rete produce (log ampiezza, fase) e costruiamo psi = exp(log_amp + i*phase), poi normalizziamo.
OUTPUT_PARAMETRIZATION = _env_str("QSP_OUTPUT_PARAMETRIZATION", "logamp_phase")
if OUTPUT_PARAMETRIZATION not in {"direct_complex", "logamp_phase"}:
    raise ValueError(
        f"OUTPUT_PARAMETRIZATION={OUTPUT_PARAMETRIZATION!r} non valida. "
        "Valori ammessi: direct_complex, logamp_phase."
    )
