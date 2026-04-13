from __future__ import annotations

import builtins
import functools
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# FakeMPI fallback: sul laptop locale, se mpi4py non e' disponibile, il codice
# gira come singolo processo rank 0.
try:
    from mpi4py import MPI
except Exception:
    class _FakeComm:
        def Get_rank(self) -> int:
            return 0

        def Get_size(self) -> int:
            return 1

        def bcast(self, value, root: int = 0):
            return value

        def Barrier(self):
            return None

    class _FakeMPI:
        COMM_WORLD = _FakeComm()

    MPI = _FakeMPI()


comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
world_size = comm.Get_size()
local_rank = global_rank % 4 if torch.cuda.is_available() else 0

if world_size > 1 and torch.cuda.is_available():
    # Setup variabili DDP usando MPI
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if global_rank == 0:
        import socket

        master_addr = socket.gethostbyname(socket.gethostname())
    else:
        master_addr = None
    master_addr = comm.bcast(master_addr, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))

import config
from input import generate_fixed_tfim_dataset
import trainer as trainer_lib
from trainer import (
    AdaptiveTrainingTrace,
    AutoencoderTrainingTrace,
    ModelSelectionTrace,
    TrainingHistory,
    benchmark_latent_pipeline,
    build_model,
    compute_observable_curves,
    evaluate_autoencoder_reconstruction,
    evaluate_autoregressive,
    evaluate_multistep,
    evaluate_teacher_forced,
    exposure_bias_detected,
    plot_autoencoder_training_curves,
    plot_observable_curves,
    plot_training_curves,
    resolve_partial_warmup_steps,
    set_seed,
    train_model,
    try_resume_from_last_checkpoint,
)


DDP_ACTIVE = bool(world_size > 1 and device.type == "cuda" and dist.is_initialized())
IS_ROOT = global_rank == 0
ROOT_PRINT = builtins.print


def _rank0_print(*args, **kwargs):
    if IS_ROOT:
        ROOT_PRINT(*args, **kwargs)


builtins.print = _rank0_print

# `config.DEVICE == "cuda"` e' usato in trainer.py per autocast/GradScaler.
# Il binding al device corretto avviene con `torch.cuda.set_device(local_rank)`.
config.DEVICE = "cuda" if device.type == "cuda" else "cpu"
config.SAVE_MODEL = bool(config.SAVE_MODEL and IS_ROOT)
config.CLAMP_AUDIT_PRINT = bool(config.CLAMP_AUDIT_PRINT and IS_ROOT)


def _dist_ready() -> bool:
    return bool(DDP_ACTIVE and dist.is_available() and dist.is_initialized())


def _barrier():
    if _dist_ready():
        dist.barrier()
    elif world_size > 1:
        comm.Barrier()


def _cleanup_dist():
    if _dist_ready():
        dist.destroy_process_group()


def _shard_size(total: int, rank: int, size: int) -> int:
    base = int(total) // int(size)
    remainder = int(total) % int(size)
    return base + (1 if rank < remainder else 0)


def _all_shard_sizes(total: int, size: int) -> list[int]:
    return [_shard_size(total, rank, size) for rank in range(size)]


GLOBAL_TRAIN_SEQUENCES = int(config.TRAIN_SEQUENCES)
GLOBAL_TEST_SEQUENCES = int(config.TEST_SEQUENCES)
TRAIN_SHARDS = _all_shard_sizes(GLOBAL_TRAIN_SEQUENCES, world_size)
TEST_SHARDS = _all_shard_sizes(GLOBAL_TEST_SEQUENCES, world_size)
LOCAL_TRAIN_SEQUENCES = TRAIN_SHARDS[global_rank]
LOCAL_TEST_SEQUENCES = TEST_SHARDS[global_rank]

if LOCAL_TRAIN_SEQUENCES < 1:
    raise ValueError(
        "Shard train vuoto: aumenta QSP_TRAIN_SEQUENCES o riduci world_size. "
        f"train={GLOBAL_TRAIN_SEQUENCES}, world_size={world_size}"
    )
if LOCAL_TEST_SEQUENCES < 1:
    raise ValueError(
        "Shard validation/test vuoto: aumenta QSP_TEST_SEQUENCES o riduci world_size. "
        f"test={GLOBAL_TEST_SEQUENCES}, world_size={world_size}"
    )

# Requisito 3: runtime sharding per evitare che ogni rank materializzi tutto il dataset.
config.TRAIN_SEQUENCES = int(LOCAL_TRAIN_SEQUENCES)
config.TEST_SEQUENCES = int(LOCAL_TEST_SEQUENCES)
config.S_TRAIN = int(LOCAL_TRAIN_SEQUENCES)
config.S_TEST = int(LOCAL_TEST_SEQUENCES)

# Seed shard-aware per evitare che tutti i rank generino la stessa porzione.
# Nota: per `fixed_tfim_basis` questo cambia anche il campionamento dell'Hamiltoniana.
DATASET_SEED = int(config.SEED if world_size == 1 else config.SEED + global_rank)

_ORIGINAL_CHECKPOINT_CONFIG_SNAPSHOT = trainer_lib._checkpoint_config_snapshot


def _checkpoint_config_snapshot_hpc() -> dict[str, object]:
    snapshot = _ORIGINAL_CHECKPOINT_CONFIG_SNAPSHOT()
    snapshot["TRAIN_SEQUENCES"] = int(GLOBAL_TRAIN_SEQUENCES)
    snapshot["TEST_SEQUENCES"] = int(GLOBAL_TEST_SEQUENCES)
    return snapshot


trainer_lib._checkpoint_config_snapshot = _checkpoint_config_snapshot_hpc


class DDPModelAdapter(torch.nn.Module):
    """
    Adapter minimale per usare il modello wrappato in DDP con il trainer esistente.

    - `forward()` e `predict_latent()` passano da DDP.
    - I metodi helper continuano a delegare al modulo sottostante per compatibilita'
      con il codice attuale (encode/decode/reconstruct/state_dict/load_state_dict).
    """

    def __init__(self, ddp_model: DistributedDataParallel):
        super().__init__()
        self.ddp = ddp_model

    @property
    def module(self):
        return self.ddp.module

    @property
    def autoencoder(self):
        return self.ddp.module.autoencoder

    @property
    def predictor(self):
        return self.ddp.module.predictor

    def forward(
        self,
        latent_context: torch.Tensor,
        phys_params: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.ddp(latent_context, phys_params, padding_mask=padding_mask)

    def predict_latent(
        self,
        latent_context: torch.Tensor,
        phys_params: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.ddp(latent_context, phys_params, padding_mask=padding_mask)

    def encode_states(self, states: torch.Tensor) -> torch.Tensor:
        return self.ddp.module.encode_states(states)

    def decode_latents(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.ddp.module.decode_latents(latent_states)

    def reconstruct_states(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ddp.module.reconstruct_states(states)

    def state_dict(self, *args, **kwargs):
        return self.ddp.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.ddp.module.load_state_dict(state_dict, strict=strict)


def _reduce_weighted_scalar(value: float, weight: float) -> float:
    if not _dist_ready():
        return float(value)

    if weight <= 0 or not np.isfinite(value):
        payload = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    else:
        payload = torch.tensor([float(value) * float(weight), float(weight)], dtype=torch.float64, device=device)
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    if float(payload[1].item()) <= 0.0:
        return float("nan")
    return float((payload[0] / payload[1]).item())


def _reduce_weighted_array(values, weight: float, valid_mask=None):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not _dist_ready():
        return array.copy()

    value_tensor = torch.as_tensor(array, dtype=torch.float64, device=device)
    if valid_mask is None:
        valid_tensor = torch.isfinite(value_tensor)
    else:
        valid_tensor = torch.as_tensor(valid_mask, dtype=torch.bool, device=device) & torch.isfinite(value_tensor)

    numerator = torch.where(valid_tensor, value_tensor * float(weight), torch.zeros_like(value_tensor))
    denominator = torch.where(valid_tensor, torch.full_like(value_tensor, float(weight)), torch.zeros_like(value_tensor))
    packed = torch.stack([numerator, denominator], dim=0)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    reduced = torch.full_like(value_tensor, float("nan"))
    valid = packed[1] > 0
    reduced[valid] = packed[0][valid] / packed[1][valid]
    return reduced.cpu().numpy()


def _reduce_evaluation_result(result: trainer_lib.EvaluationResult, *, local_weight: float) -> trainer_lib.EvaluationResult:
    if not _dist_ready():
        return result

    coverage = np.asarray(result.coverage_curve, dtype=np.float64)
    fidelity = np.asarray(result.fidelity_curve, dtype=np.float64)
    valid_steps = coverage > 0.0

    return trainer_lib.EvaluationResult(
        loss=_reduce_weighted_scalar(float(result.loss), local_weight),
        mean_fidelity=_reduce_weighted_scalar(float(result.mean_fidelity), local_weight),
        fidelity_curve=_reduce_weighted_array(fidelity, local_weight, valid_mask=valid_steps).tolist(),
        coverage_curve=_reduce_weighted_array(coverage, local_weight, valid_mask=valid_steps).tolist(),
    )


def _reduce_observable_curves(
    curves: trainer_lib.ObservableComparisonCurves,
    *,
    local_weight: float,
) -> trainer_lib.ObservableComparisonCurves:
    if not _dist_ready():
        return curves

    return trainer_lib.ObservableComparisonCurves(
        time_indices=np.asarray(curves.time_indices, dtype=np.float64),
        physical_time=np.asarray(curves.physical_time, dtype=np.float64),
        mz_exact=_reduce_weighted_array(curves.mz_exact, local_weight),
        mz_multistep=_reduce_weighted_array(curves.mz_multistep, local_weight),
        mz_rollout=_reduce_weighted_array(curves.mz_rollout, local_weight),
        mx_exact=_reduce_weighted_array(curves.mx_exact, local_weight),
        mx_multistep=_reduce_weighted_array(curves.mx_multistep, local_weight),
        mx_rollout=_reduce_weighted_array(curves.mx_rollout, local_weight),
        cz_exact=_reduce_weighted_array(curves.cz_exact, local_weight),
        cz_multistep=_reduce_weighted_array(curves.cz_multistep, local_weight),
        cz_rollout=_reduce_weighted_array(curves.cz_rollout, local_weight),
        entropy_exact=_reduce_weighted_array(curves.entropy_exact, local_weight),
        entropy_multistep=_reduce_weighted_array(curves.entropy_multistep, local_weight),
        entropy_rollout=_reduce_weighted_array(curves.entropy_rollout, local_weight),
    )


def _reduce_float_list(values: list[float], *, local_weight: float) -> list[float]:
    reduced = _reduce_weighted_array(values, local_weight)
    return [float(v) for v in reduced.tolist()]


def _synchronize_history(history: TrainingHistory) -> TrainingHistory:
    if not _dist_ready():
        return history
    return TrainingHistory(
        epochs=list(history.epochs),
        train_loss=[float(v) for v in _reduce_float_list(history.train_loss, local_weight=LOCAL_TRAIN_SEQUENCES)],
        train_fidelity=[float(v) for v in _reduce_float_list(history.train_fidelity, local_weight=LOCAL_TRAIN_SEQUENCES)],
    )


def _synchronize_autoencoder_trace(trace: AutoencoderTrainingTrace) -> AutoencoderTrainingTrace:
    if not _dist_ready():
        return trace
    return AutoencoderTrainingTrace(
        epochs=list(trace.epochs),
        train_loss=[float(v) for v in _reduce_float_list(trace.train_loss, local_weight=LOCAL_TRAIN_SEQUENCES)],
        train_fidelity=[float(v) for v in _reduce_float_list(trace.train_fidelity, local_weight=LOCAL_TRAIN_SEQUENCES)],
        validation_loss=[float(v) for v in _reduce_float_list(trace.validation_loss, local_weight=LOCAL_TEST_SEQUENCES)],
        validation_fidelity=[float(v) for v in _reduce_float_list(trace.validation_fidelity, local_weight=LOCAL_TEST_SEQUENCES)],
        best_epoch=int(trace.best_epoch),
        best_validation_fidelity=_reduce_weighted_scalar(float(trace.best_validation_fidelity), LOCAL_TEST_SEQUENCES),
        training_seconds=float(trace.training_seconds),
    )


def _synchronize_adaptive_trace(trace: AdaptiveTrainingTrace) -> AdaptiveTrainingTrace:
    if not _dist_ready():
        return trace

    summaries: list[trainer_lib.AdaptiveEpochSummary] = []
    for summary in trace.epoch_summaries:
        summaries.append(
            trainer_lib.AdaptiveEpochSummary(
                epoch=int(summary.epoch),
                horizon=int(summary.horizon),
                teacher_steps=int(summary.teacher_steps),
                head_loss=_reduce_weighted_scalar(float(summary.head_loss), LOCAL_TRAIN_SEQUENCES),
                tail_loss=_reduce_weighted_scalar(float(summary.tail_loss), LOCAL_TRAIN_SEQUENCES),
                head_fidelity=_reduce_weighted_scalar(float(summary.head_fidelity), LOCAL_TRAIN_SEQUENCES),
                tail_fidelity=_reduce_weighted_scalar(float(summary.tail_fidelity), LOCAL_TRAIN_SEQUENCES),
                mean_offset_losses=[
                    float(v)
                    for v in _reduce_float_list(summary.mean_offset_losses, local_weight=LOCAL_TRAIN_SEQUENCES)
                ],
                mean_offset_fidelities=[
                    float(v)
                    for v in _reduce_float_list(summary.mean_offset_fidelities, local_weight=LOCAL_TRAIN_SEQUENCES)
                ],
                mean_offset_weights=[
                    float(v)
                    for v in _reduce_float_list(summary.mean_offset_weights, local_weight=LOCAL_TRAIN_SEQUENCES)
                ],
            )
        )

    return AdaptiveTrainingTrace(
        enabled=bool(trace.enabled),
        initial_horizon=int(trace.initial_horizon),
        initial_teacher_steps=int(trace.initial_teacher_steps),
        final_horizon=int(trace.final_horizon),
        final_teacher_steps=int(trace.final_teacher_steps),
        epoch_summaries=summaries,
    )


def _install_ddp_metric_wrappers():
    original_autoencoder_eval = trainer_lib.evaluate_autoencoder_reconstruction
    original_teacher_eval = trainer_lib.evaluate_teacher_forced
    original_multistep_eval = trainer_lib.evaluate_multistep
    original_autoregressive_eval = trainer_lib.evaluate_autoregressive
    original_observable_eval = trainer_lib.compute_observable_curves

    @functools.wraps(original_autoencoder_eval)
    def wrapped_autoencoder_eval(model, states, *args, **kwargs):
        local_result = original_autoencoder_eval(model, states, *args, **kwargs)
        local_weight = float(states.shape[0] * states.shape[1])
        return _reduce_evaluation_result(local_result, local_weight=local_weight)

    @functools.wraps(original_teacher_eval)
    def wrapped_teacher_eval(model, states, *args, **kwargs):
        local_result = original_teacher_eval(model, states, *args, **kwargs)
        return _reduce_evaluation_result(local_result, local_weight=float(states.shape[0]))

    @functools.wraps(original_multistep_eval)
    def wrapped_multistep_eval(model, states, *args, **kwargs):
        local_result = original_multistep_eval(model, states, *args, **kwargs)
        return _reduce_evaluation_result(local_result, local_weight=float(states.shape[0]))

    @functools.wraps(original_autoregressive_eval)
    def wrapped_autoregressive_eval(model, states, *args, **kwargs):
        local_result = original_autoregressive_eval(model, states, *args, **kwargs)
        return _reduce_evaluation_result(local_result, local_weight=float(states.shape[0]))

    @functools.wraps(original_observable_eval)
    def wrapped_observable_eval(model, states, *args, **kwargs):
        local_curves = original_observable_eval(model, states, *args, **kwargs)
        return _reduce_observable_curves(local_curves, local_weight=float(states.shape[0]))

    trainer_lib.evaluate_autoencoder_reconstruction = wrapped_autoencoder_eval
    trainer_lib.evaluate_teacher_forced = wrapped_teacher_eval
    trainer_lib.evaluate_multistep = wrapped_multistep_eval
    trainer_lib.evaluate_autoregressive = wrapped_autoregressive_eval
    trainer_lib.compute_observable_curves = wrapped_observable_eval


_install_ddp_metric_wrappers()

# Riallinea gli alias locali alle funzioni monkey-patchate nel modulo trainer.
evaluate_autoencoder_reconstruction = trainer_lib.evaluate_autoencoder_reconstruction
evaluate_teacher_forced = trainer_lib.evaluate_teacher_forced
evaluate_multistep = trainer_lib.evaluate_multistep
evaluate_autoregressive = trainer_lib.evaluate_autoregressive


def _as_serializable(result) -> dict[str, object]:
    return {
        "loss": float(result.loss),
        "mean_fidelity": float(result.mean_fidelity),
        "fidelity_curve": [None if np.isnan(v) else float(v) for v in result.fidelity_curve],
        "coverage_curve": [float(v) for v in result.coverage_curve],
    }


def _history_as_serializable(history: TrainingHistory) -> dict[str, object]:
    return {
        "epochs": [int(epoch) for epoch in history.epochs],
        "train_loss": [float(value) for value in history.train_loss],
        "train_fidelity": [float(value) for value in history.train_fidelity],
    }


def _observable_curves_as_serializable(curves) -> dict[str, object]:
    return {
        "time_indices": [int(v) for v in curves.time_indices.tolist()],
        "physical_time": [float(v) for v in curves.physical_time.tolist()],
        "mz_exact": [float(v) for v in curves.mz_exact.tolist()],
        "mz_multistep": [float(v) for v in curves.mz_multistep.tolist()],
        "mz_rollout": [float(v) for v in curves.mz_rollout.tolist()],
        "mx_exact": [float(v) for v in curves.mx_exact.tolist()],
        "mx_multistep": [float(v) for v in curves.mx_multistep.tolist()],
        "mx_rollout": [float(v) for v in curves.mx_rollout.tolist()],
        "cz_exact": [float(v) for v in curves.cz_exact.tolist()],
        "cz_multistep": [float(v) for v in curves.cz_multistep.tolist()],
        "cz_rollout": [float(v) for v in curves.cz_rollout.tolist()],
        "entropy_exact": [float(v) for v in curves.entropy_exact.tolist()],
        "entropy_multistep": [float(v) for v in curves.entropy_multistep.tolist()],
        "entropy_rollout": [float(v) for v in curves.entropy_rollout.tolist()],
    }


def _autoencoder_trace_as_serializable(trace: AutoencoderTrainingTrace) -> dict[str, object]:
    return {
        "epochs": [int(v) for v in trace.epochs],
        "train_loss": [float(v) for v in trace.train_loss],
        "train_fidelity": [float(v) for v in trace.train_fidelity],
        "validation_loss": [float(v) for v in trace.validation_loss],
        "validation_fidelity": [float(v) for v in trace.validation_fidelity],
        "best_epoch": int(trace.best_epoch),
        "best_validation_fidelity": float(trace.best_validation_fidelity),
        "training_seconds": float(trace.training_seconds),
    }


def _benchmark_as_serializable(benchmark) -> dict[str, object]:
    return {
        "input_feature_dim": int(benchmark.input_feature_dim),
        "embedding_dim": int(benchmark.embedding_dim),
        "compression_ratio": float(benchmark.compression_ratio),
        "autoencoder_training_seconds": float(benchmark.autoencoder_training_seconds),
        "train_cache_seconds": float(benchmark.train_cache_seconds),
        "test_cache_seconds": float(benchmark.test_cache_seconds),
        "teacher_eval_uncached_seconds": float(benchmark.teacher_eval_uncached_seconds),
        "teacher_eval_cached_seconds": float(benchmark.teacher_eval_cached_seconds),
    }


def _adaptive_training_as_serializable(trace: AdaptiveTrainingTrace) -> dict[str, object]:
    return {
        "enabled": bool(trace.enabled),
        "initial_horizon": int(trace.initial_horizon),
        "initial_teacher_steps": int(trace.initial_teacher_steps),
        "final_horizon": int(trace.final_horizon),
        "final_teacher_steps": int(trace.final_teacher_steps),
        "epoch_summaries": [
            {
                "epoch": int(summary.epoch),
                "horizon": int(summary.horizon),
                "teacher_steps": int(summary.teacher_steps),
                "head_loss": float(summary.head_loss),
                "tail_loss": float(summary.tail_loss),
                "head_fidelity": float(summary.head_fidelity),
                "tail_fidelity": float(summary.tail_fidelity),
                "mean_offset_losses": [float(v) for v in summary.mean_offset_losses],
                "mean_offset_fidelities": [float(v) for v in summary.mean_offset_fidelities],
                "mean_offset_weights": [float(v) for v in summary.mean_offset_weights],
            }
            for summary in trace.epoch_summaries
        ],
    }


def _model_selection_as_serializable(trace: ModelSelectionTrace) -> dict[str, object]:
    return {
        "criterion": str(trace.criterion),
        "best_epoch": int(trace.best_epoch),
        "best_objective": float(trace.best_objective),
        "best_teacher_forced_fidelity": float(trace.best_teacher_forced_fidelity),
        "best_multistep_fidelity": float(trace.best_multistep_fidelity),
        "best_rollout_fidelity": float(trace.best_rollout_fidelity),
        "rollout_weight": float(trace.rollout_weight),
        "multistep_weight": float(trace.multistep_weight),
        "teacher_forced_weight": float(trace.teacher_forced_weight),
    }


def _load_history_from_last_checkpoint() -> TrainingHistory:
    if not config.LAST_CHECKPOINT_PATH.exists():
        return TrainingHistory(epochs=[], train_loss=[], train_fidelity=[])

    payload = torch.load(config.LAST_CHECKPOINT_PATH, map_location="cpu")
    history = payload.get("history", {})
    return TrainingHistory(
        epochs=[int(epoch) for epoch in history.get("epochs", [])],
        train_loss=[float(value) for value in history.get("train_loss", [])],
        train_fidelity=[float(value) for value in history.get("train_fidelity", [])],
    )


def _load_trained_model(model):
    if not config.CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint non trovato: {config.CHECKPOINT_PATH}. "
            "Disattiva QSP_EVAL_ONLY oppure genera prima best_model.pt."
        )
    state_dict = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return _load_history_from_last_checkpoint()


def _plot_split_curves(ax, title: str, multistep, autoregressive, partial_results: dict[int, object]):
    x = np.arange(1, len(multistep.fidelity_curve) + 1)
    ax.plot(x, multistep.fidelity_curve, label="Metodo 1: multi-step", linewidth=2.3, color="#117a65")
    ax.plot(x, autoregressive.fidelity_curve, label="Metodo 2: rollout libero", linewidth=2.3, color="#b03a2e")
    palette = ["#117a65", "#7d6608", "#6c3483", "#566573"]
    for color, (warmup_n1, result) in zip(palette, sorted(partial_results.items())):
        ax.plot(
            x,
            result.fidelity_curve,
            label=f"Metodo 3: warmup N1={warmup_n1}",
            linewidth=2.0,
            linestyle="--",
            color=color,
        )
    ax.set_title(title)
    ax.set_xlabel("Indice stato predetto")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)


def main():
    # Reset del seed per tenere inizializzazione modello e training coerenti su tutti i rank.
    set_seed(config.SEED)

    if IS_ROOT:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = generate_fixed_tfim_dataset(
        train_sequences=config.TRAIN_SEQUENCES,
        test_sequences=config.TEST_SEQUENCES,
        n_qubits=config.N_QUBITS,
        num_states=config.NUM_STATES,
        seed=DATASET_SEED,
    )
    resume_status = {
        "enabled": bool(config.AUTO_RESUME),
        "resumed": False,
        "reason": "eval-only attivo" if config.EVAL_ONLY else "auto-resume disattivato",
    }

    print("=" * 88)
    print("Quantum Sequence Prediction | HPC MPI + DDP pipeline")
    print("=" * 88)
    print(f"DDP attivo:             {DDP_ACTIVE}")
    print(f"World size:             {world_size}")
    print(f"Nodo/rank locale:       global_rank={global_rank} local_rank={local_rank}")
    print(f"Device:                 {device}")
    print(f"Qubits:                 {config.N_QUBITS} (dim={config.DIM_2N})")
    print(f"Stati per traiettoria:  {config.NUM_STATES} (predizioni={config.SEQ_LEN})")
    print(
        "Dataset shard-aware:    "
        f"train_local={LOCAL_TRAIN_SEQUENCES} / train_global={GLOBAL_TRAIN_SEQUENCES}, "
        f"test_local={LOCAL_TEST_SEQUENCES} / test_global={GLOBAL_TEST_SEQUENCES}"
    )
    print(f"Dataset seed rank0:     {config.SEED} | seed runtime rank-aware: {DATASET_SEED} su rank {global_rank}")
    print(
        f"Transformer:            d_model={config.D_MODEL}, layers={config.NUM_LAYERS}, "
        f"heads={config.NUM_HEADS}, ff={config.DIM_FEEDFORWARD}, dropout={config.DROPOUT}"
    )
    print(
        f"Latent pipeline:        embedding_dim={config.EMBEDDING_DIM}, "
        f"ae_hidden={config.AUTOENCODER_HIDDEN_DIM}, ae_epochs={config.AUTOENCODER_EPOCHS}"
    )
    print(
        f"Training:               batch_size={config.BATCH_SIZE}, lr={config.LEARNING_RATE:.2e}, "
        f"epochs={config.EPOCHS}, tf_only={config.HYBRID_TEACHER_FORCING_EPOCHS}, "
        f"early_stop_patience={config.EARLY_STOPPING_PATIENCE}"
    )
    print(f"Stati iniziali:         {dataset.train.initial_state_family}")
    print(f"Motivo famiglia:        {dataset.initial_state_family_reason}")
    print(f"Checkpoint best:        {'trovato' if config.CHECKPOINT_PATH.exists() else 'assente'} | {config.CHECKPOINT_PATH}")
    print(f"Checkpoint last:        {'trovato' if config.LAST_CHECKPOINT_PATH.exists() else 'assente'} | {config.LAST_CHECKPOINT_PATH}")
    print("=" * 88)

    model = build_model()
    if DDP_ACTIVE:
        model = DDPModelAdapter(DistributedDataParallel(model, device_ids=[local_rank]))

    if config.EVAL_ONLY:
        history = _load_trained_model(model)
        autoencoder_trace = AutoencoderTrainingTrace(
            epochs=[],
            train_loss=[],
            train_fidelity=[],
            validation_loss=[],
            validation_fidelity=[],
            best_epoch=0,
            best_validation_fidelity=float("nan"),
            training_seconds=0.0,
        )
        adaptive_trace = AdaptiveTrainingTrace(
            enabled=False,
            initial_horizon=int(config.MULTISTEP_H),
            initial_teacher_steps=int(config.MULTISTEP_EFFECTIVE_TEACHER_FORCING_STEPS),
            final_horizon=int(config.MULTISTEP_H),
            final_teacher_steps=int(config.MULTISTEP_EFFECTIVE_TEACHER_FORCING_STEPS),
            epoch_summaries=[],
        )
        selection_trace = ModelSelectionTrace(
            criterion="existing_checkpoint",
            best_epoch=int(history.epochs[-1]) if history.epochs else 0,
            best_objective=float("nan"),
            best_teacher_forced_fidelity=float("nan"),
            best_multistep_fidelity=float("nan"),
            best_rollout_fidelity=float("nan"),
            rollout_weight=0.0,
            multistep_weight=1.0,
            teacher_forced_weight=0.0,
        )
        train_latent_states = None
        test_latent_states = None
    else:
        resume_state = try_resume_from_last_checkpoint(model)
        if config.AUTO_RESUME:
            resume_status = {
                "enabled": True,
                "resumed": bool(resume_state.resumed),
                "reason": str(resume_state.reason),
            }
            if resume_state.resumed:
                print(f"Auto-resume:            {resume_state.reason}")
            else:
                print(f"Auto-resume saltato:    {resume_state.reason}")

        history, adaptive_trace, selection_trace, autoencoder_trace, latent_cache = train_model(
            model,
            dataset.train.states,
            dataset.train.params,
            validation_states=dataset.test.states,
            validation_params=dataset.test.params,
            start_epoch=resume_state.start_epoch,
            history=resume_state.history,
            optimizer_state_dict=resume_state.optimizer_state_dict,
            scheduler_state_dict=resume_state.scheduler_state_dict,
            best_objective=resume_state.best_objective,
            best_state=resume_state.best_state,
        )
        history = _synchronize_history(history)
        adaptive_trace = _synchronize_adaptive_trace(adaptive_trace)
        autoencoder_trace = _synchronize_autoencoder_trace(autoencoder_trace)
        train_latent_states = latent_cache.train_latent_states
        test_latent_states = latent_cache.test_latent_states

    if train_latent_states is None:
        train_latent_states = model.encode_states(dataset.train.states.to(config.DEVICE)).detach().cpu()
    if test_latent_states is None:
        test_latent_states = model.encode_states(dataset.test.states.to(config.DEVICE)).detach().cpu()

    # Tutte le valutazioni collettive vanno eseguite nello stesso ordine su ogni rank.
    train_teacher = evaluate_teacher_forced(
        model,
        dataset.train.states,
        dataset.train.params,
        latent_states=train_latent_states,
    )
    test_teacher = evaluate_teacher_forced(
        model,
        dataset.test.states,
        dataset.test.params,
        latent_states=test_latent_states,
    )
    train_multistep = evaluate_multistep(
        model,
        dataset.train.states,
        dataset.train.params,
        latent_states=train_latent_states,
    )
    test_multistep = evaluate_multistep(
        model,
        dataset.test.states,
        dataset.test.params,
        latent_states=test_latent_states,
    )
    rollout_warmup = int(config.ROLLOUT_WARMUP_STATES)
    train_rollout = evaluate_autoregressive(
        model,
        dataset.train.states,
        dataset.train.params,
        latent_states=train_latent_states,
        warmup_states=rollout_warmup,
    )
    test_rollout = evaluate_autoregressive(
        model,
        dataset.test.states,
        dataset.test.params,
        latent_states=test_latent_states,
        warmup_states=rollout_warmup,
    )
    train_autoencoder_eval = evaluate_autoencoder_reconstruction(model, dataset.train.states)
    test_autoencoder_eval = evaluate_autoencoder_reconstruction(model, dataset.test.states)

    add_partial_curves = exposure_bias_detected(train_multistep.fidelity_curve, train_rollout.fidelity_curve) or (
        exposure_bias_detected(test_multistep.fidelity_curve, test_rollout.fidelity_curve)
    )
    partial_results_train: dict[int, object] = {}
    partial_results_test: dict[int, object] = {}
    warmup_n1_values: list[int] = []
    if add_partial_curves:
        warmup_n1_values = resolve_partial_warmup_steps(config.SEQ_LEN)
        print("\nExposure bias rilevato: aggiungo curve metodo 3 per N1=" + ", ".join(str(v) for v in warmup_n1_values))
        for warmup_n1 in warmup_n1_values:
            warmup_states = warmup_n1 + 1
            partial_results_train[warmup_n1] = evaluate_autoregressive(
                model,
                dataset.train.states,
                dataset.train.params,
                latent_states=train_latent_states,
                warmup_states=warmup_states,
            )
            partial_results_test[warmup_n1] = evaluate_autoregressive(
                model,
                dataset.test.states,
                dataset.test.params,
                latent_states=test_latent_states,
                warmup_states=warmup_states,
            )
    else:
        print("\nNessun exposure bias marcato: mantengo solo metodo 1 e 2.")

    test_seq_idx = min(int(config.OBSERVABLES_TEST_SEQUENCE_INDEX), int(dataset.test.num_sequences) - 1)
    test_single_sequence = dataset.test.states[test_seq_idx : test_seq_idx + 1]
    test_single_params = dataset.test.params[test_seq_idx : test_seq_idx + 1]
    test_observables = compute_observable_curves(
        model,
        test_single_sequence,
        test_single_params,
        latent_states=test_latent_states[test_seq_idx : test_seq_idx + 1],
        warmup_states=rollout_warmup,
    )

    benchmark = benchmark_latent_pipeline(
        model,
        dataset.test.states,
        dataset.test.params,
        test_latent_states,
        autoencoder_trace=autoencoder_trace,
    )

    if not config.EVAL_ONLY:
        benchmark.train_cache_seconds = float(latent_cache.train_cache_seconds)
        benchmark.test_cache_seconds = float(latent_cache.test_cache_seconds)

    _barrier()

    if IS_ROOT:
        if history.epochs:
            plot_training_curves(history)
        if autoencoder_trace.epochs:
            plot_autoencoder_training_curves(autoencoder_trace)

        fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.3), sharey=True)
        _plot_split_curves(axes[0], "Train Set", train_multistep, train_rollout, partial_results_train)
        _plot_split_curves(axes[1], "Test Set", test_multistep, test_rollout, partial_results_test)
        fig.suptitle("Fidelity multi-step vs rollout nel tempo", fontsize=14)
        fig.tight_layout()
        fig.savefig(config.FIDELITY_PLOT_PATH, dpi=config.PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

        plot_observable_curves(
            curves=test_observables,
            warmup_states=rollout_warmup,
            output_path=config.OBSERVABLES_TEST_PLOT_PATH,
            title=(
                f"Osservabili | test sequence idx={test_seq_idx} | "
                f"{test_observables.time_indices.size} stati per traiettoria, warmup={rollout_warmup}"
            ),
        )

        summary = {
            "seed": int(config.SEED),
            "device": str(device),
            "hpc": {
                "ddp_active": bool(DDP_ACTIVE),
                "world_size": int(world_size),
                "global_rank_logging": int(global_rank),
                "local_rank_logging": int(local_rank),
                "train_sequences_global": int(GLOBAL_TRAIN_SEQUENCES),
                "test_sequences_global": int(GLOBAL_TEST_SEQUENCES),
                "train_sequences_local_rank0": int(TRAIN_SHARDS[0]),
                "test_sequences_local_rank0": int(TEST_SHARDS[0]),
                "train_shards": [int(v) for v in TRAIN_SHARDS],
                "test_shards": [int(v) for v in TEST_SHARDS],
                "dataset_seed_rule": "QSP_SEED + global_rank (solo in world_size>1)",
            },
            "config": {
                "DATASET_SOURCE": config.DATASET_SOURCE,
                "N_QUBITS": int(config.N_QUBITS),
                "DIM_2N": int(config.DIM_2N),
                "NUM_STATES": int(config.NUM_STATES),
                "SEQ_LEN": int(config.SEQ_LEN),
                "MULTISTEP_H": int(config.MULTISTEP_H),
                "MULTISTEP_H_START": int(config.MULTISTEP_H_START),
                "MULTISTEP_H_MAX": int(config.MULTISTEP_H_MAX),
                "MULTISTEP_EFFECTIVE_TEACHER_FORCING_STEPS": int(config.MULTISTEP_EFFECTIVE_TEACHER_FORCING_STEPS),
                "MULTISTEP_TEACHER_FORCING_STEPS": int(config.MULTISTEP_TEACHER_FORCING_STEPS),
                "HYBRID_TEACHER_FORCING_EPOCHS": int(config.HYBRID_TEACHER_FORCING_EPOCHS),
                "MULTISTEP_H_PLATEAU_PATIENCE": int(config.MULTISTEP_H_PLATEAU_PATIENCE),
                "MULTISTEP_H_PLATEAU_MIN_DELTA": float(config.MULTISTEP_H_PLATEAU_MIN_DELTA),
                "EARLY_STOPPING_PATIENCE": int(config.EARLY_STOPPING_PATIENCE),
                "EARLY_STOPPING_MIN_EPOCHS": int(config.EARLY_STOPPING_MIN_EPOCHS),
                "ADAPTIVE_MULTISTEP_ENABLED": bool(config.ADAPTIVE_MULTISTEP_ENABLED),
                "ADAPTIVE_STATS_EMA": float(config.ADAPTIVE_STATS_EMA),
                "ADAPTIVE_WEIGHT_ALPHA": float(config.ADAPTIVE_WEIGHT_ALPHA),
                "ADAPTIVE_WEIGHT_MIN": float(config.ADAPTIVE_WEIGHT_MIN),
                "ADAPTIVE_WEIGHT_MAX": float(config.ADAPTIVE_WEIGHT_MAX),
                "ADAPTIVE_H_MIN": int(config.ADAPTIVE_H_MIN),
                "ADAPTIVE_H_MAX": int(config.ADAPTIVE_H_MAX),
                "ADAPTIVE_TEACHER_MIN": int(config.ADAPTIVE_TEACHER_MIN),
                "ADAPTIVE_TEACHER_MAX": int(config.ADAPTIVE_TEACHER_MAX),
                "ADAPTIVE_H_LOSS_THRESHOLD": float(config.ADAPTIVE_H_LOSS_THRESHOLD),
                "ADAPTIVE_H_FIDELITY_THRESHOLD": float(config.ADAPTIVE_H_FIDELITY_THRESHOLD),
                "ADAPTIVE_TEACHER_LOSS_THRESHOLD": float(config.ADAPTIVE_TEACHER_LOSS_THRESHOLD),
                "ADAPTIVE_TEACHER_FIDELITY_THRESHOLD": float(config.ADAPTIVE_TEACHER_FIDELITY_THRESHOLD),
                "TRAIN_SEQUENCES": int(GLOBAL_TRAIN_SEQUENCES),
                "TEST_SEQUENCES": int(GLOBAL_TEST_SEQUENCES),
                "INITIAL_STATE_FAMILY": config.INITIAL_STATE_FAMILY,
                "FORCE_X_BASIS_ONLY": bool(config.FORCE_X_BASIS_ONLY),
                "DROPOUT": float(config.DROPOUT),
                "EMBEDDING_DIM": int(config.EMBEDDING_DIM),
                "AUTOENCODER_HIDDEN_DIM": int(config.AUTOENCODER_HIDDEN_DIM),
                "AUTOENCODER_EPOCHS": int(config.AUTOENCODER_EPOCHS),
                "AUTOENCODER_BATCH_SIZE": int(config.AUTOENCODER_BATCH_SIZE),
                "AUTOENCODER_LEARNING_RATE": float(config.AUTOENCODER_LEARNING_RATE),
                "AUTOENCODER_WEIGHT_DECAY": float(config.AUTOENCODER_WEIGHT_DECAY),
                "ROLLOUT_WARMUP_STATES": int(config.ROLLOUT_WARMUP_STATES),
                "EVAL_ONLY": bool(config.EVAL_ONLY),
                "AUTO_RESUME": bool(config.AUTO_RESUME),
                "PARTIAL_WARMUP_STEPS": config.PARTIAL_WARMUP_STEPS,
                "OBSERVABLES_TEST_SEQUENCE_INDEX": int(config.OBSERVABLES_TEST_SEQUENCE_INDEX),
                "CLAMP_AUDIT_PRINT": bool(config.CLAMP_AUDIT_PRINT),
                "CLAMP_AUDIT_MAX_SEQUENCES": int(config.CLAMP_AUDIT_MAX_SEQUENCES),
                "CLAMP_AUDIT_MAX_STATES": int(config.CLAMP_AUDIT_MAX_STATES),
                "active_env_overrides": config.get_active_env_overrides(),
            },
            "dataset": {
                "source": config.DATASET_SOURCE,
                "initial_state_family": dataset.train.initial_state_family,
                "initial_state_family_reason": dataset.initial_state_family_reason,
                "rank0_train_initial_state_codes": dataset.train.initial_state_codes,
                "rank0_test_initial_state_codes": dataset.test.initial_state_codes,
            },
            "resume": resume_status,
            "training_scheme": {
                "mode": "teacher_forced_then_hybrid_50_50_vectorized_multistep",
                "teacher_forcing_epochs": int(config.HYBRID_TEACHER_FORCING_EPOCHS),
                "multistep_epochs": max(0, int(config.EPOCHS) - int(config.HYBRID_TEACHER_FORCING_EPOCHS)),
                "multistep_horizon_start": int(config.MULTISTEP_H_START),
                "multistep_horizon_max": int(config.MULTISTEP_H_MAX),
                "multistep_horizon_eval": int(config.MULTISTEP_H),
                "multistep_teacher_steps": int(config.MULTISTEP_EFFECTIVE_TEACHER_FORCING_STEPS),
                "hybrid_teacher_forced_weight": 0.5,
                "hybrid_multistep_weight": 0.5,
                "multistep_step_weighting": "descending_linear_mean_normalized",
                "rollout_evaluation_warmup_states": int(config.ROLLOUT_WARMUP_STATES),
            },
            "training_history": _history_as_serializable(history),
            "autoencoder": {
                "training_trace": _autoencoder_trace_as_serializable(autoencoder_trace),
                "train_reconstruction": _as_serializable(train_autoencoder_eval),
                "test_reconstruction": _as_serializable(test_autoencoder_eval),
            },
            "adaptive_training": _adaptive_training_as_serializable(adaptive_trace),
            "model_selection": _model_selection_as_serializable(selection_trace),
            "benchmark": _benchmark_as_serializable(benchmark),
            "evaluation": {
                "train_teacher_forced": _as_serializable(train_teacher),
                "train_multistep": _as_serializable(train_multistep),
                "train_autoregressive": _as_serializable(train_rollout),
                "test_teacher_forced": _as_serializable(test_teacher),
                "test_multistep": _as_serializable(test_multistep),
                "test_autoregressive": _as_serializable(test_rollout),
                "test_observables_sequence_index_rank0_shard": int(test_seq_idx),
                "test_observables": _observable_curves_as_serializable(test_observables),
                "partial_warmup_n1_values": warmup_n1_values,
                "train_partial_warmups": {str(k): _as_serializable(v) for k, v in partial_results_train.items()},
                "test_partial_warmups": {str(k): _as_serializable(v) for k, v in partial_results_test.items()},
            },
            "artifacts": {
                "autoencoder_checkpoint": str(config.AUTOENCODER_CHECKPOINT_PATH),
                "autoencoder_training_plot": str(config.AUTOENCODER_TRAINING_CURVES_PATH),
                "latent_train_cache": str(config.LATENT_TRAIN_CACHE_PATH),
                "latent_test_cache": str(config.LATENT_TEST_CACHE_PATH),
                "fidelity_plot": str(config.FIDELITY_PLOT_PATH),
                "training_curves_plot": str(config.TRAINING_CURVES_PATH),
                "observables_train_plot": str(config.OBSERVABLES_TRAIN_PLOT_PATH),
                "observables_test_plot": str(config.OBSERVABLES_TEST_PLOT_PATH),
                "summary_json": str(config.SUMMARY_PATH),
            },
        }
        with config.SUMMARY_PATH.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        print("\nMetriche aggregate cluster-wide")
        print(
            f"  Train | teacher={train_teacher.mean_fidelity:.6f} | "
            f"multistep={train_multistep.mean_fidelity:.6f} | rollout={train_rollout.mean_fidelity:.6f}"
        )
        print(
            f"  Test  | teacher={test_teacher.mean_fidelity:.6f} | "
            f"multistep={test_multistep.mean_fidelity:.6f} | rollout={test_rollout.mean_fidelity:.6f}"
        )
        print(
            f"  AE    | train={train_autoencoder_eval.mean_fidelity:.6f} | "
            f"test={test_autoencoder_eval.mean_fidelity:.6f} | "
            f"compression={benchmark.compression_ratio:.2f}x"
        )
        print(
            f"  Best  | epoch={selection_trace.best_epoch} | "
            f"score={selection_trace.best_objective:.6f} | "
            f"tf/ms=({selection_trace.best_teacher_forced_fidelity:.6f}/"
            f"{selection_trace.best_multistep_fidelity:.6f})"
        )
        print(f"\nPlot fidelity:  {config.FIDELITY_PLOT_PATH}")
        print(f"Plot training:  {config.TRAINING_CURVES_PATH}")
        print(f"Obs test plot:  {config.OBSERVABLES_TEST_PLOT_PATH}")
        print(f"Summary JSON:   {config.SUMMARY_PATH}")


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup_dist()
