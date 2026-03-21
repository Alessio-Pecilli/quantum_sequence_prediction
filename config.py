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
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_PATH = RESULTS_DIR / "best_model.pt"
SUMMARY_PATH = RESULTS_DIR / "run_summary.json"
FIDELITY_PLOT_PATH = RESULTS_DIR / "fidelity_vs_time.png"
TRAINING_CURVES_PATH = RESULTS_DIR / "training_curves.png"


SEED = _env_int("QSP_SEED", 7)


N_QUBITS = _env_int("QSP_N_QUBITS", 8)
if N_QUBITS < 1:
    raise ValueError(f"N_QUBITS deve essere >= 1, ricevuto: {N_QUBITS}")

DIM_2N = 2 ** N_QUBITS

# Numero totale di stati in ogni traiettoria, incluso psi(0).
NUM_STATES = _env_int("QSP_NUM_STATES", 12)
if NUM_STATES < 2:
    raise ValueError(f"NUM_STATES deve essere >= 2, ricevuto: {NUM_STATES}")

# Alias retrocompatibile: numero di transizioni supervisionate.
SEQ_LEN = NUM_STATES - 1


TRAIN_SEQUENCES = _env_int("QSP_TRAIN_SEQUENCES", 2048)
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


INITIAL_STATE_FAMILY = _env_str("QSP_INITIAL_STATE_FAMILY", "local_clifford")
if INITIAL_STATE_FAMILY not in {"auto", "basis", "pauli_basis", "local_clifford"}:
    raise ValueError(
        f"INITIAL_STATE_FAMILY={INITIAL_STATE_FAMILY!r} non valida. "
        "Valori ammessi: auto, basis, pauli_basis, local_clifford."
    )
BASIS_SUPPORT_FRACTION_LIMIT = _env_float("QSP_BASIS_SUPPORT_FRACTION_LIMIT", 0.125)
if not (0.0 < BASIS_SUPPORT_FRACTION_LIMIT <= 1.0):
    raise ValueError(
        "BASIS_SUPPORT_FRACTION_LIMIT deve stare in (0,1], "
        f"ricevuto: {BASIS_SUPPORT_FRACTION_LIMIT}"
    )


D_MODEL = _env_int("QSP_D_MODEL", 512)
NUM_HEADS = _env_int("QSP_NUM_HEADS", 8)
NUM_LAYERS = _env_int("QSP_NUM_LAYERS", 4)
DIM_FEEDFORWARD = _env_int("QSP_DIM_FEEDFORWARD", 2048)
DROPOUT = _env_float("QSP_DROPOUT", 0.0)
if D_MODEL <= 0 or NUM_HEADS <= 0 or NUM_LAYERS <= 0 or DIM_FEEDFORWARD <= 0:
    raise ValueError("D_MODEL, NUM_HEADS, NUM_LAYERS e DIM_FEEDFORWARD devono essere > 0.")
if D_MODEL % NUM_HEADS != 0:
    raise ValueError("D_MODEL deve essere divisibile per NUM_HEADS.")


BATCH_SIZE = _env_int("QSP_BATCH_SIZE", 32)
EPOCHS = _env_int("QSP_EPOCHS", 200)
LEARNING_RATE = _env_float("QSP_LEARNING_RATE", 1e-4)
WEIGHT_DECAY = _env_float("QSP_WEIGHT_DECAY", 1e-4)
GRAD_CLIP_MAX_NORM = _env_float("QSP_GRAD_CLIP_MAX_NORM", 1.0)
LOG_FIDELITY_EPS = _env_float("QSP_LOG_FIDELITY_EPS", 1e-8)
SCHEDULED_SAMPLING_MAX_PROB = _env_float("QSP_SCHEDULED_SAMPLING_MAX_PROB", 0.30)
SCHEDULED_SAMPLING_RAMP_EPOCHS = _env_int("QSP_SCHEDULED_SAMPLING_RAMP_EPOCHS", 120)
ROLLOUT_AUX_WEIGHT = _env_float("QSP_ROLLOUT_AUX_WEIGHT", 0.50)
ROLLOUT_CURRICULUM_EPOCHS = _env_int("QSP_ROLLOUT_CURRICULUM_EPOCHS", 120)
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


EXPOSURE_BIAS_GAP_THRESHOLD = _env_float("QSP_EXPOSURE_BIAS_GAP_THRESHOLD", 0.10)
EXPOSURE_BIAS_DROP_THRESHOLD = _env_float("QSP_EXPOSURE_BIAS_DROP_THRESHOLD", 0.10)
PARTIAL_WARMUP_STEPS = _env_str("QSP_PARTIAL_WARMUP_STEPS", "auto")
PLOT_DPI = _env_int("QSP_PLOT_DPI", 160)


NUM_WORKERS = _env_int("QSP_NUM_WORKERS", 0)
PIN_MEMORY = _env_bool("QSP_PIN_MEMORY", True)
SAVE_MODEL = _env_bool("QSP_SAVE_MODEL", True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
