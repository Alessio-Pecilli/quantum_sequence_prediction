from __future__ import annotations

import copy
import gc
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from input import LatentSequenceDataset, QuantumSequenceDataset, cache_latent_trajectories
from observables import (
    batch_average_entanglement_entropy,
    batch_observables_tfim,
    precompute_observables,
)
from predictor import NegativeLogFidelityLoss, QuantumLatentSequenceModel, QuantumSequencePredictor, quantum_fidelity


@dataclass
class TrainingHistory:
    epochs: list[int]
    train_loss: list[float]
    train_fidelity: list[float]


@dataclass
class EvaluationResult:
    loss: float
    mean_fidelity: float
    fidelity_curve: list[float]
    coverage_curve: list[float]


@dataclass
class ObservableComparisonCurves:
    """Confronto tra traiettorie esatte, multi-step e rollout libero."""

    time_indices: np.ndarray
    physical_time: np.ndarray
    mz_exact: np.ndarray
    mz_multistep: np.ndarray
    mz_rollout: np.ndarray
    mx_exact: np.ndarray
    mx_multistep: np.ndarray
    mx_rollout: np.ndarray
    cz_exact: np.ndarray
    cz_multistep: np.ndarray
    cz_rollout: np.ndarray
    entropy_exact: np.ndarray
    entropy_multistep: np.ndarray
    entropy_rollout: np.ndarray


@dataclass
class ResumeCheckpointState:
    start_epoch: int
    history: TrainingHistory
    optimizer_state_dict: dict | None
    scheduler_state_dict: dict | None
    best_objective: float | None
    best_state: dict | None
    resumed: bool
    reason: str


@dataclass
class BatchAdaptiveStats:
    horizon: int
    teacher_steps: int
    mean_offset_losses: list[float]
    mean_offset_fidelities: list[float]
    mean_offset_weights: list[float]


@dataclass
class AdaptiveEpochSummary:
    epoch: int
    horizon: int
    teacher_steps: int
    head_loss: float
    tail_loss: float
    head_fidelity: float
    tail_fidelity: float
    mean_offset_losses: list[float]
    mean_offset_fidelities: list[float]
    mean_offset_weights: list[float]


@dataclass
class AdaptiveTrainingTrace:
    enabled: bool
    initial_horizon: int
    initial_teacher_steps: int
    final_horizon: int
    final_teacher_steps: int
    epoch_summaries: list[AdaptiveEpochSummary]


@dataclass
class AutoencoderTrainingTrace:
    epochs: list[int]
    train_loss: list[float]
    train_fidelity: list[float]
    validation_loss: list[float]
    validation_fidelity: list[float]
    best_epoch: int
    best_validation_fidelity: float
    training_seconds: float


@dataclass
class LatentCacheArtifacts:
    train_latent_states: torch.Tensor
    test_latent_states: torch.Tensor
    train_cache_seconds: float
    test_cache_seconds: float


@dataclass
class PipelineBenchmark:
    input_feature_dim: int
    embedding_dim: int
    compression_ratio: float
    autoencoder_training_seconds: float
    train_cache_seconds: float
    test_cache_seconds: float
    teacher_eval_uncached_seconds: float
    teacher_eval_cached_seconds: float


@dataclass
class ModelSelectionTrace:
    criterion: str
    best_epoch: int
    best_objective: float
    best_teacher_forced_fidelity: float
    best_multistep_fidelity: float
    best_rollout_fidelity: float
    rollout_weight: float
    multistep_weight: float
    teacher_forced_weight: float


@dataclass
class AdaptiveControllerState:
    current_horizon: int
    current_teacher_steps: int
    ema_head_loss: float | None = None
    ema_tail_loss: float | None = None
    ema_head_fidelity: float | None = None
    ema_tail_fidelity: float | None = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(states: torch.Tensor, params: torch.Tensor, shuffle: bool) -> DataLoader:
    dataset = QuantumSequenceDataset(states, params)
    safe_batch_size = max(1, int(config.BATCH_SIZE) // 2)
    return DataLoader(
        dataset,
        batch_size=safe_batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def build_latent_loader(
    latent_states: torch.Tensor,
    target_states: torch.Tensor,
    params: torch.Tensor,
    shuffle: bool,
) -> DataLoader:
    dataset = LatentSequenceDataset(latent_states, target_states, params)
    safe_batch_size = max(1, int(config.BATCH_SIZE) // 2)
    return DataLoader(
        dataset,
        batch_size=safe_batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def build_model() -> QuantumLatentSequenceModel:
    return QuantumLatentSequenceModel().to(config.DEVICE)


def _build_autoencoder_state_loader(states: torch.Tensor, shuffle: bool) -> DataLoader:
    flattened_states = states.reshape(-1, states.shape[-1])
    dataset = torch.utils.data.TensorDataset(flattened_states)
    return DataLoader(
        dataset,
        batch_size=max(1, int(config.AUTOENCODER_BATCH_SIZE)),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


@torch.no_grad()
def evaluate_autoencoder_reconstruction(
    model: QuantumLatentSequenceModel,
    states: torch.Tensor,
) -> EvaluationResult:
    model.eval()
    criterion = NegativeLogFidelityLoss()
    loader = _build_autoencoder_state_loader(states, shuffle=False)
    total_loss = 0.0
    total_fidelity = 0.0
    total_samples = 0
    for (batch_states,) in loader:
        batch_states = batch_states.to(config.DEVICE)
        _, reconstructed = model.reconstruct_states(batch_states)
        loss, mean_fidelity, _ = criterion(reconstructed, batch_states)
        batch_size = int(batch_states.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_fidelity += float(mean_fidelity.item()) * batch_size
        total_samples += batch_size
    return EvaluationResult(
        loss=total_loss / max(1, total_samples),
        mean_fidelity=total_fidelity / max(1, total_samples),
        fidelity_curve=[],
        coverage_curve=[],
    )


def train_autoencoder(
    model: QuantumLatentSequenceModel,
    train_states: torch.Tensor,
    validation_states: torch.Tensor | None = None,
) -> AutoencoderTrainingTrace:
    if config.AUTO_RESUME and config.AUTOENCODER_CHECKPOINT_PATH.exists():
        payload = torch.load(config.AUTOENCODER_CHECKPOINT_PATH, map_location=config.DEVICE)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            model.autoencoder.load_state_dict(payload["model_state_dict"])
            trace_payload = payload.get("trace", {})
            return AutoencoderTrainingTrace(
                epochs=[int(v) for v in trace_payload.get("epochs", [])],
                train_loss=[float(v) for v in trace_payload.get("train_loss", [])],
                train_fidelity=[float(v) for v in trace_payload.get("train_fidelity", [])],
                validation_loss=[float(v) for v in trace_payload.get("validation_loss", [])],
                validation_fidelity=[float(v) for v in trace_payload.get("validation_fidelity", [])],
                best_epoch=int(trace_payload.get("best_epoch", 0)),
                best_validation_fidelity=float(trace_payload.get("best_validation_fidelity", float("nan"))),
                training_seconds=float(trace_payload.get("training_seconds", 0.0)),
            )

    train_loader = _build_autoencoder_state_loader(train_states, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.autoencoder.parameters(),
        lr=config.AUTOENCODER_LEARNING_RATE,
        weight_decay=config.AUTOENCODER_WEIGHT_DECAY,
    )
    criterion = NegativeLogFidelityLoss()
    use_amp = config.DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    trace = AutoencoderTrainingTrace(
        epochs=[],
        train_loss=[],
        train_fidelity=[],
        validation_loss=[],
        validation_fidelity=[],
        best_epoch=0,
        best_validation_fidelity=float("-inf"),
        training_seconds=0.0,
    )
    best_state = copy.deepcopy(model.autoencoder.state_dict())
    started_at = time.perf_counter()
    config.AUTOENCODER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(config.AUTOENCODER_EPOCHS) + 1):
        model.autoencoder.train()
        loss_sum = 0.0
        fidelity_sum = 0.0
        sample_count = 0
        for (batch_states,) in train_loader:
            batch_states = batch_states.to(config.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _, reconstructed = model.reconstruct_states(batch_states)
                loss, mean_fidelity, _ = criterion(reconstructed, batch_states)
            scaler.scale(loss).backward()
            if config.GRAD_CLIP_MAX_NORM > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.autoencoder.parameters(), config.GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(batch_states.shape[0])
            loss_sum += float(loss.item()) * batch_size
            fidelity_sum += float(mean_fidelity.item()) * batch_size
            sample_count += batch_size

        trace.epochs.append(epoch)
        trace.train_loss.append(loss_sum / max(1, sample_count))
        trace.train_fidelity.append(fidelity_sum / max(1, sample_count))

        if validation_states is not None:
            validation = evaluate_autoencoder_reconstruction(model, validation_states)
            val_loss = float(validation.loss)
            val_fidelity = float(validation.mean_fidelity)
        else:
            val_loss = float("nan")
            val_fidelity = float(trace.train_fidelity[-1])
        trace.validation_loss.append(val_loss)
        trace.validation_fidelity.append(val_fidelity)

        if val_fidelity > trace.best_validation_fidelity:
            trace.best_validation_fidelity = val_fidelity
            trace.best_epoch = epoch
            best_state = copy.deepcopy(model.autoencoder.state_dict())
            _safe_atomic_torch_save(
                {
                    "model_state_dict": best_state,
                    "trace": {
                        "epochs": trace.epochs,
                        "train_loss": trace.train_loss,
                        "train_fidelity": trace.train_fidelity,
                        "validation_loss": trace.validation_loss,
                        "validation_fidelity": trace.validation_fidelity,
                        "best_epoch": trace.best_epoch,
                        "best_validation_fidelity": trace.best_validation_fidelity,
                        "training_seconds": 0.0,
                    },
                },
                config.AUTOENCODER_CHECKPOINT_PATH,
                label="best autoencoder checkpoint",
            )

        if epoch == int(config.AUTOENCODER_EPOCHS) or epoch <= 3 or epoch % max(1, config.AUTOENCODER_EPOCHS // 10) == 0:
            print(
                f"  [autoencoder] epoca {epoch:4d}/{config.AUTOENCODER_EPOCHS} | "
                f"train(fid/loss)=({trace.train_fidelity[-1]:.4f}/{trace.train_loss[-1]:.4f}) | "
                f"val(fid/loss)=({val_fidelity:.4f}/{val_loss:.4f})"
            )

    trace.training_seconds = float(time.perf_counter() - started_at)
    model.autoencoder.load_state_dict(best_state)
    _safe_atomic_torch_save(
        {
            "model_state_dict": model.autoencoder.state_dict(),
            "trace": {
                "epochs": trace.epochs,
                "train_loss": trace.train_loss,
                "train_fidelity": trace.train_fidelity,
                "validation_loss": trace.validation_loss,
                "validation_fidelity": trace.validation_fidelity,
                "best_epoch": trace.best_epoch,
                "best_validation_fidelity": trace.best_validation_fidelity,
                "training_seconds": trace.training_seconds,
            },
        },
        config.AUTOENCODER_LAST_CHECKPOINT_PATH,
        label="last autoencoder checkpoint",
    )
    _safe_atomic_torch_save(
        {
            "model_state_dict": model.autoencoder.state_dict(),
            "trace": {
                "epochs": trace.epochs,
                "train_loss": trace.train_loss,
                "train_fidelity": trace.train_fidelity,
                "validation_loss": trace.validation_loss,
                "validation_fidelity": trace.validation_fidelity,
                "best_epoch": trace.best_epoch,
                "best_validation_fidelity": trace.best_validation_fidelity,
                "training_seconds": trace.training_seconds,
            },
        },
        config.AUTOENCODER_CHECKPOINT_PATH,
        label="final autoencoder checkpoint",
    )
    return trace


def freeze_encoder(model: QuantumLatentSequenceModel):
    for parameter in model.autoencoder.encoder.parameters():
        parameter.requires_grad_(False)
    model.autoencoder.encoder.eval()


def build_latent_caches(
    model: QuantumLatentSequenceModel,
    train_states: torch.Tensor,
    train_params: torch.Tensor,
    test_states: torch.Tensor,
    test_params: torch.Tensor,
) -> LatentCacheArtifacts:
    freeze_encoder(model)
    started = time.perf_counter()
    train_latent_states = cache_latent_trajectories(
        train_states,
        train_params,
        encoder_fn=model.encode_states,
        batch_size=config.AUTOENCODER_BATCH_SIZE,
        device=config.DEVICE,
    )
    train_cache_seconds = float(time.perf_counter() - started)

    started = time.perf_counter()
    test_latent_states = cache_latent_trajectories(
        test_states,
        test_params,
        encoder_fn=model.encode_states,
        batch_size=config.AUTOENCODER_BATCH_SIZE,
        device=config.DEVICE,
    )
    test_cache_seconds = float(time.perf_counter() - started)

    if config.SAVE_MODEL:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _safe_atomic_torch_save(
            {
                "latent_states": train_latent_states,
                "params": train_params.to(torch.float32),
            },
            config.LATENT_TRAIN_CACHE_PATH,
            label="latent train cache",
        )
        _safe_atomic_torch_save(
            {
                "latent_states": test_latent_states,
                "params": test_params.to(torch.float32),
            },
            config.LATENT_TEST_CACHE_PATH,
            label="latent test cache",
        )

    return LatentCacheArtifacts(
        train_latent_states=train_latent_states,
        test_latent_states=test_latent_states,
        train_cache_seconds=train_cache_seconds,
        test_cache_seconds=test_cache_seconds,
    )


def _empty_resume_state(reason: str) -> ResumeCheckpointState:
    return ResumeCheckpointState(
        start_epoch=1,
        history=TrainingHistory(epochs=[], train_loss=[], train_fidelity=[]),
        optimizer_state_dict=None,
        scheduler_state_dict=None,
        best_objective=None,
        best_state=None,
        resumed=False,
        reason=reason,
    )


def _checkpoint_config_snapshot() -> dict[str, object]:
    return {
        "DATASET_SOURCE": str(config.DATASET_SOURCE),
        "N_QUBITS": int(config.N_QUBITS),
        "DIM_2N": int(config.DIM_2N),
        "NUM_STATES": int(config.NUM_STATES),
        "TRAIN_SEQUENCES": int(config.TRAIN_SEQUENCES),
        "TEST_SEQUENCES": int(config.TEST_SEQUENCES),
        "TIME_STEP": float(config.TIME_STEP),
        "COUPLING_MEAN": float(config.COUPLING_MEAN),
        "FIELD_STRENGTH": float(config.FIELD_STRENGTH),
        "BATCH_SIZE": int(config.BATCH_SIZE),
        "EPOCHS": int(config.EPOCHS),
        "LEARNING_RATE": float(config.LEARNING_RATE),
        "WEIGHT_DECAY": float(config.WEIGHT_DECAY),
        "EMBEDDING_DIM": int(config.EMBEDDING_DIM),
        "AUTOENCODER_HIDDEN_DIM": int(config.AUTOENCODER_HIDDEN_DIM),
        "AUTOENCODER_EPOCHS": int(config.AUTOENCODER_EPOCHS),
        "AUTOENCODER_BATCH_SIZE": int(config.AUTOENCODER_BATCH_SIZE),
        "AUTOENCODER_LEARNING_RATE": float(config.AUTOENCODER_LEARNING_RATE),
        "AUTOENCODER_WEIGHT_DECAY": float(config.AUTOENCODER_WEIGHT_DECAY),
        "D_MODEL": int(config.D_MODEL),
        "NUM_HEADS": int(config.NUM_HEADS),
        "NUM_LAYERS": int(config.NUM_LAYERS),
        "DIM_FEEDFORWARD": int(config.DIM_FEEDFORWARD),
        "DROPOUT": float(config.DROPOUT),
        "OUTPUT_PARAMETRIZATION": str(config.OUTPUT_PARAMETRIZATION),
        "FORCE_X_BASIS_ONLY": bool(config.FORCE_X_BASIS_ONLY),
        "MULTISTEP_H_START": int(config.MULTISTEP_H_START),
        "MULTISTEP_H_MAX": int(config.MULTISTEP_H_MAX),
        "MULTISTEP_H": int(config.MULTISTEP_H),
        "MULTISTEP_TEACHER_FORCING_STEPS": int(config.MULTISTEP_TEACHER_FORCING_STEPS),
        "MULTISTEP_EFFECTIVE_TEACHER_FORCING_STEPS": int(config.MULTISTEP_EFFECTIVE_TEACHER_FORCING_STEPS),
        "HYBRID_TEACHER_FORCING_EPOCHS": int(config.HYBRID_TEACHER_FORCING_EPOCHS),
        "MULTISTEP_H_PLATEAU_PATIENCE": int(config.MULTISTEP_H_PLATEAU_PATIENCE),
        "MULTISTEP_H_PLATEAU_MIN_DELTA": float(config.MULTISTEP_H_PLATEAU_MIN_DELTA),
        "EARLY_STOPPING_PATIENCE": int(config.EARLY_STOPPING_PATIENCE),
        "EARLY_STOPPING_MIN_EPOCHS": int(config.EARLY_STOPPING_MIN_EPOCHS),
        "ADAPTIVE_MULTISTEP_ENABLED": bool(config.ADAPTIVE_MULTISTEP_ENABLED),
        "ADAPTIVE_H_MIN": int(config.ADAPTIVE_H_MIN),
        "ADAPTIVE_H_MAX": int(config.ADAPTIVE_H_MAX),
        "ADAPTIVE_TEACHER_MIN": int(config.ADAPTIVE_TEACHER_MIN),
        "ADAPTIVE_TEACHER_MAX": int(config.ADAPTIVE_TEACHER_MAX),
        "ADAPTIVE_WEIGHT_ALPHA": float(config.ADAPTIVE_WEIGHT_ALPHA),
        "ADAPTIVE_WEIGHT_MIN": float(config.ADAPTIVE_WEIGHT_MIN),
        "ADAPTIVE_WEIGHT_MAX": float(config.ADAPTIVE_WEIGHT_MAX),
        "ADAPTIVE_STATS_EMA": float(config.ADAPTIVE_STATS_EMA),
    }


def _checkpoint_config_mismatches(saved_config: dict[str, object]) -> list[str]:
    current_config = _checkpoint_config_snapshot()
    mismatches: list[str] = []
    for key, current_value in current_config.items():
        if key == "EPOCHS":
            continue
        if key not in saved_config:
            continue
        saved_value = saved_config[key]
        if isinstance(current_value, float):
            if not np.isclose(float(saved_value), current_value, rtol=0.0, atol=1e-12):
                mismatches.append(f"{key}: checkpoint={saved_value} current={current_value}")
        elif saved_value != current_value:
            mismatches.append(f"{key}: checkpoint={saved_value} current={current_value}")
    return mismatches


def try_resume_from_last_checkpoint(model: QuantumLatentSequenceModel) -> ResumeCheckpointState:
    if not config.AUTO_RESUME:
        return _empty_resume_state("QSP_AUTO_RESUME=0")
    if not config.LAST_CHECKPOINT_PATH.exists():
        return _empty_resume_state(f"checkpoint assente: {config.LAST_CHECKPOINT_PATH}")

    try:
        payload = torch.load(config.LAST_CHECKPOINT_PATH, map_location=config.DEVICE)
    except Exception as exc:
        return _empty_resume_state(f"caricamento fallito: {exc}")

    saved_config = payload.get("config", {})
    if not isinstance(saved_config, dict):
        return _empty_resume_state("config checkpoint mancante o non valida")

    mismatches = _checkpoint_config_mismatches(saved_config)
    if mismatches:
        return _empty_resume_state("config incompatibile; " + "; ".join(mismatches))

    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        return _empty_resume_state("model_state_dict mancante")

    try:
        model.load_state_dict(model_state_dict)
    except Exception as exc:
        return _empty_resume_state(f"state_dict incompatibile: {exc}")

    history_payload = payload.get("history", {})
    if not isinstance(history_payload, dict):
        history_payload = {}
    history = TrainingHistory(
        epochs=[int(epoch) for epoch in history_payload.get("epochs", [])],
        train_loss=[float(value) for value in history_payload.get("train_loss", [])],
        train_fidelity=[float(value) for value in history_payload.get("train_fidelity", [])],
    )
    last_completed_epoch = int(payload.get("epoch", 0))
    if history.epochs:
        last_completed_epoch = max(last_completed_epoch, int(history.epochs[-1]))

    best_objective = payload.get("best_objective", payload.get("best_loss"))
    return ResumeCheckpointState(
        start_epoch=max(1, last_completed_epoch + 1),
        history=history,
        optimizer_state_dict=payload.get("optimizer_state_dict"),
        scheduler_state_dict=payload.get("scheduler_state_dict"),
        best_objective=None if best_objective is None else float(best_objective),
        best_state=payload.get("best_state_dict"),
        resumed=True,
        reason=f"resume da {config.LAST_CHECKPOINT_PATH} (ultima epoca completa: {last_completed_epoch})",
    )


def _teacher_forcing_epochs() -> int:
    return max(1, min(int(config.HYBRID_TEACHER_FORCING_EPOCHS), int(config.EPOCHS)))


def _training_phase_for_epoch(epoch: int) -> str:
    return "teacher_forced" if int(epoch) <= _teacher_forcing_epochs() else "hybrid"


def _scheduler_total_steps(num_batches: int) -> int:
    return max(1, int(config.EPOCHS) * int(num_batches))


def _effective_multistep_teacher_steps(horizon: int, requested_steps: int | None = None) -> int:
    if horizon < 1:
        return 0
    if requested_steps is None:
        requested_steps = max(1, int(np.ceil(float(horizon) / 2.0)))
    return max(0, min(int(requested_steps), int(horizon)))


def _describe_multistep_transition(start_index: int, target_index: int, use_teacher: bool) -> str:
    target_state = target_index + 1
    if target_index == start_index:
        return f"uso il contesto vero fino a t{start_index} per predire t{target_state}"
    source_label = "vero" if use_teacher else "predetto"
    return (
        f"uso il nuovo stato {source_label} t{target_index} insieme al contesto fino a t{target_index} "
        f"per predire t{target_state}"
    )


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _ema_update(previous: float | None, value: float) -> float:
    if previous is None or not np.isfinite(previous):
        return float(value)
    return float(config.ADAPTIVE_STATS_EMA * previous + (1.0 - config.ADAPTIVE_STATS_EMA) * value)


def _make_adaptive_controller() -> AdaptiveControllerState:
    if not config.ADAPTIVE_MULTISTEP_ENABLED:
        horizon = int(config.MULTISTEP_H)
        teacher_steps = _effective_multistep_teacher_steps(horizon, int(config.MULTISTEP_TEACHER_FORCING_STEPS))
        return AdaptiveControllerState(
            current_horizon=horizon,
            current_teacher_steps=teacher_steps,
        )

    horizon = max(1, min(int(config.ADAPTIVE_H_MIN), int(config.ADAPTIVE_H_MAX), int(config.SEQ_LEN)))
    teacher_upper = min(int(config.ADAPTIVE_TEACHER_MAX), horizon)
    teacher_seed = min(int(config.MULTISTEP_TEACHER_FORCING_STEPS), teacher_upper)
    teacher_steps = max(int(config.ADAPTIVE_TEACHER_MIN), teacher_seed)
    return AdaptiveControllerState(
        current_horizon=horizon,
        current_teacher_steps=teacher_steps,
    )


def _build_step_weights(
    horizon: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if horizon < 1:
        raise ValueError(f"horizon deve essere >= 1, ricevuto {horizon}")
    raw_weights = torch.arange(int(horizon), 0, -1, device=device, dtype=dtype)
    return raw_weights / raw_weights.mean().clamp(min=1e-8)


def _summarize_epoch_adaptive_stats(
    *,
    epoch: int,
    horizon: int,
    teacher_steps: int,
    loss_sums: np.ndarray,
    fidelity_sums: np.ndarray,
    weight_sums: np.ndarray,
    counts: np.ndarray,
) -> AdaptiveEpochSummary:
    mean_losses: list[float] = []
    mean_fidelities: list[float] = []
    mean_weights: list[float] = []
    for offset in range(horizon):
        if counts[offset] <= 0.0:
            mean_losses.append(float("nan"))
            mean_fidelities.append(float("nan"))
            mean_weights.append(float("nan"))
            continue
        mean_losses.append(float(loss_sums[offset] / counts[offset]))
        mean_fidelities.append(float(fidelity_sums[offset] / counts[offset]))
        mean_weights.append(float(weight_sums[offset] / counts[offset]))

    finite_losses = [value for value in mean_losses if np.isfinite(value)]
    finite_fidelities = [value for value in mean_fidelities if np.isfinite(value)]
    head_loss = mean_losses[0] if mean_losses else float("nan")
    tail_loss = finite_losses[-1] if finite_losses else float("nan")
    head_fidelity = mean_fidelities[0] if mean_fidelities else float("nan")
    tail_fidelity = finite_fidelities[-1] if finite_fidelities else float("nan")
    return AdaptiveEpochSummary(
        epoch=epoch,
        horizon=horizon,
        teacher_steps=teacher_steps,
        head_loss=head_loss,
        tail_loss=tail_loss,
        head_fidelity=head_fidelity,
        tail_fidelity=tail_fidelity,
        mean_offset_losses=mean_losses,
        mean_offset_fidelities=mean_fidelities,
        mean_offset_weights=mean_weights,
    )


def _update_adaptive_controller(
    controller: AdaptiveControllerState,
    summary: AdaptiveEpochSummary,
):
    controller.ema_head_loss = _ema_update(controller.ema_head_loss, summary.head_loss)
    controller.ema_tail_loss = _ema_update(controller.ema_tail_loss, summary.tail_loss)
    controller.ema_head_fidelity = _ema_update(controller.ema_head_fidelity, summary.head_fidelity)
    controller.ema_tail_fidelity = _ema_update(controller.ema_tail_fidelity, summary.tail_fidelity)

    if not config.ADAPTIVE_MULTISTEP_ENABLED:
        return

    tail_loss = float(controller.ema_tail_loss)
    tail_fidelity = float(controller.ema_tail_fidelity)
    head_loss = float(controller.ema_head_loss)
    head_fidelity = float(controller.ema_head_fidelity)

    if (
        tail_loss <= float(config.ADAPTIVE_H_LOSS_THRESHOLD)
        and tail_fidelity >= float(config.ADAPTIVE_H_FIDELITY_THRESHOLD)
        and controller.current_horizon < int(config.ADAPTIVE_H_MAX)
    ):
        controller.current_horizon += 1
    elif (
        (
            tail_loss > 1.25 * float(config.ADAPTIVE_H_LOSS_THRESHOLD)
            or tail_fidelity < float(config.ADAPTIVE_H_FIDELITY_THRESHOLD) - 0.04
        )
        and controller.current_horizon > int(config.ADAPTIVE_H_MIN)
    ):
        controller.current_horizon -= 1

    teacher_cap = min(int(config.ADAPTIVE_TEACHER_MAX), int(controller.current_horizon))
    if (
        (
            head_loss > float(config.ADAPTIVE_TEACHER_LOSS_THRESHOLD)
            or head_fidelity < float(config.ADAPTIVE_TEACHER_FIDELITY_THRESHOLD)
        )
        and controller.current_teacher_steps < teacher_cap
    ):
        controller.current_teacher_steps += 1
    elif (
        head_loss <= 0.90 * float(config.ADAPTIVE_TEACHER_LOSS_THRESHOLD)
        and head_fidelity >= float(config.ADAPTIVE_TEACHER_FIDELITY_THRESHOLD)
        and controller.current_teacher_steps > int(config.ADAPTIVE_TEACHER_MIN)
    ):
        controller.current_teacher_steps -= 1

    controller.current_teacher_steps = max(
        int(config.ADAPTIVE_TEACHER_MIN),
        min(int(controller.current_teacher_steps), teacher_cap),
    )


def _teacher_forced_training_loss(
    model: QuantumLatentSequenceModel,
    criterion: NegativeLogFidelityLoss,
    inputs: torch.Tensor,
    target_states: torch.Tensor,
    params: torch.Tensor,
):
    predicted_latent = model.predict_latent(inputs, params)
    predicted = model.decode_latents(predicted_latent.reshape(-1, predicted_latent.shape[-1])).reshape(
        target_states.shape
    )
    loss, mean_fidelity, fidelity_matrix = criterion(predicted, target_states)
    per_step_losses = -torch.log(fidelity_matrix.clamp(min=config.LOG_FIDELITY_EPS))
    mean_losses = per_step_losses.mean(dim=0)
    mean_fidelities = fidelity_matrix.mean(dim=0)
    unit_weights = torch.ones_like(mean_losses)
    stats = BatchAdaptiveStats(
        horizon=int(inputs.shape[1]),
        teacher_steps=int(inputs.shape[1]),
        mean_offset_losses=[float(value) for value in mean_losses.detach().cpu().tolist()],
        mean_offset_fidelities=[float(value) for value in mean_fidelities.detach().cpu().tolist()],
        mean_offset_weights=[float(value) for value in unit_weights.detach().cpu().tolist()],
    )
    return loss, mean_fidelity, stats


def compute_multistep_loss(
    model: QuantumLatentSequenceModel,
    x: torch.Tensor,
    y_latent: torch.Tensor,
    y_states: torch.Tensor,
    params: torch.Tensor,
    current_h: int,
    loss_fn: NegativeLogFidelityLoss,
    teacher_steps_override: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, BatchAdaptiveStats]:
    batch_size, seq_len, _ = x.shape
    if batch_size < 1:
        raise ValueError("Il batch multi-step deve contenere almeno una sequenza.")

    current_h = max(1, min(int(current_h), int(seq_len)))
    max_start_exclusive = seq_len - current_h + 1
    if max_start_exclusive <= 1:
        t_start = 1
    else:
        t_start = int(torch.randint(1, max_start_exclusive, (1,), device=x.device).item())

    # Preserva l'intera storia vera fino a t_start, mantenendo coerenti causalita' e posizioni.
    current_context = x[:, :t_start, :].clone()
    predictions: list[torch.Tensor] = []
    step_losses: list[float] = []
    step_fidelities: list[float] = []

    # Nel repository y e' gia' shiftato di +1 rispetto a x:
    # il primo target dopo il contesto x[:, :t_start] e' y[:, t_start - 1].
    target_start = t_start - 1
    target_latents = y_latent[:, target_start : target_start + current_h, :]
    target_states = y_states[:, target_start : target_start + current_h, :]
    teacher_steps = current_h if teacher_steps_override is None else max(0, min(int(teacher_steps_override), current_h))

    for step_idx in range(current_h):
        out = model.predict_latent(current_context, params)
        next_pred = out[:, -1:, :]
        predictions.append(next_pred)

        step_target = target_states[:, step_idx : step_idx + 1, :]
        decoded_step = model.decode_latents(next_pred[:, 0, :]).unsqueeze(1)
        step_loss, step_mean_fidelity, _ = loss_fn(decoded_step, step_target)
        step_losses.append(float(step_loss.detach().item()))
        step_fidelities.append(float(step_mean_fidelity.detach().item()))

        if step_idx + 1 >= current_h:
            continue

        if step_idx + 1 < teacher_steps:
            next_context = target_latents[:, step_idx : step_idx + 1, :]
        else:
            next_context = next_pred
        current_context = torch.cat([current_context, next_context], dim=1)

    predictions_tensor = torch.cat(predictions, dim=1)
    decoded_predictions = model.decode_latents(predictions_tensor.reshape(-1, predictions_tensor.shape[-1])).reshape(
        target_states.shape
    )
    total_loss, total_fidelity, _ = loss_fn(decoded_predictions, target_states)
    stats = BatchAdaptiveStats(
        horizon=current_h,
        teacher_steps=teacher_steps,
        mean_offset_losses=step_losses,
        mean_offset_fidelities=step_fidelities,
        mean_offset_weights=[1.0 for _ in range(current_h)],
    )
    return total_loss, total_fidelity, stats


def _multistep_training_loss(
    model: QuantumLatentSequenceModel,
    criterion: NegativeLogFidelityLoss,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    target_states: torch.Tensor,
    params: torch.Tensor,
    horizon_limit: int,
    teacher_steps_override: int,
    epoch: int | None = None,
    batch_idx: int | None = None,
):
    seq_len = int(inputs.shape[1])
    horizon = max(1, min(int(horizon_limit), seq_len))
    verbose = bool(config.MULTISTEP_TRAIN_VERBOSE)

    if verbose and epoch is not None and batch_idx is not None:
        print(
            f"[train multistep] epoca={epoch} batch={batch_idx} | "
            f"SEQ_LEN={seq_len} H={horizon} | pure autoregressive unroll after t_start"
        )

    multistep_loss, multistep_fidelity, stats = compute_multistep_loss(
        model=model,
        x=inputs,
        y_latent=targets,
        y_states=target_states,
        params=params,
        current_h=horizon,
        loss_fn=criterion,
        teacher_steps_override=teacher_steps_override,
    )
    return multistep_loss, multistep_fidelity, stats, 1


def _atomic_torch_save(payload: dict, destination: os.PathLike):
    destination_path = os.fspath(destination)
    destination_dir = os.path.dirname(destination_path) or "."
    os.makedirs(destination_dir, exist_ok=True)
    backoff = 0.05
    last_error: Exception | None = None

    for attempt in range(4):
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(destination_path) + ".",
            suffix=".tmp",
            dir=destination_dir,
        )
        os.close(fd)
        try:
            torch.save(payload, tmp_path)
            try:
                os.replace(tmp_path, destination_path)
                return
            except (PermissionError, OSError):
                # Windows: destination may be read-locked; copy can still succeed.
                shutil.copyfile(tmp_path, destination_path)
                return
        except (PermissionError, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(backoff)
                backoff *= 2.0
            else:
                break
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if last_error is not None:
        raise last_error


def _safe_atomic_torch_save(payload: dict, destination: os.PathLike, *, label: str) -> bool:
    try:
        _atomic_torch_save(payload, destination)
        return True
    except (PermissionError, OSError, RuntimeError) as exc:
        print(f"[checkpoint warning] save fallito per {label}: {exc}")
        return False


def _save_last_checkpoint(
    model: QuantumSequencePredictor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    history: TrainingHistory,
    epoch: int,
    best_objective: float,
    best_state: dict | None,
):
    if not config.SAVE_MODEL:
        return
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": {
            "epochs": list(history.epochs),
            "train_loss": list(history.train_loss),
            "train_fidelity": list(history.train_fidelity),
        },
        "best_objective": float(best_objective),
        "best_loss": float(best_objective),
        "best_state_dict": best_state,
        "config": {
            **_checkpoint_config_snapshot(),
        },
    }
    _safe_atomic_torch_save(
        checkpoint_payload,
        config.LAST_CHECKPOINT_PATH,
        label="last checkpoint",
    )


def train_model(
    model: QuantumLatentSequenceModel,
    train_states: torch.Tensor,
    train_params: torch.Tensor,
    validation_states: torch.Tensor | None = None,
    validation_params: torch.Tensor | None = None,
    start_epoch: int = 1,
    history: TrainingHistory | None = None,
    optimizer_state_dict: dict | None = None,
    scheduler_state_dict: dict | None = None,
    best_objective: float | None = None,
    best_state: dict | None = None,
) -> tuple[TrainingHistory, AdaptiveTrainingTrace, ModelSelectionTrace, AutoencoderTrainingTrace, LatentCacheArtifacts]:
    history = history or TrainingHistory(epochs=[], train_loss=[], train_fidelity=[])
    autoencoder_trace = train_autoencoder(model, train_states, validation_states=validation_states)
    if validation_states is None or validation_params is None:
        validation_states = train_states
        validation_params = train_params
    latent_cache = build_latent_caches(
        model,
        train_states=train_states,
        train_params=train_params,
        test_states=validation_states,
        test_params=validation_params,
    )
    optimizer = torch.optim.AdamW(
        model.predictor.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    loader = build_latent_loader(
        latent_cache.train_latent_states,
        train_states,
        train_params,
        shuffle=True,
    )
    steps_per_epoch = len(loader)
    use_amp = config.DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_scheduler_steps = _scheduler_total_steps(steps_per_epoch)

    # OneCycleLR sul numero reale di update del training ibrido.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        total_steps=total_scheduler_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy="cos",
    )
    criterion = NegativeLogFidelityLoss()

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    best_objective = float("-inf") if best_objective is None else float(best_objective)
    current_horizon = int(config.MULTISTEP_H_START)
    eval_horizon = int(config.MULTISTEP_H_MAX)
    plateau_best_tf_loss = float("inf")
    plateau_epochs = 0
    early_stop_counter = 0
    best_metric_delta = max(1e-8, float(config.MULTISTEP_H_PLATEAU_MIN_DELTA))
    initial_horizon = int(current_horizon)
    initial_teacher_steps = int(_effective_multistep_teacher_steps(current_horizon))
    adaptive_epoch_summaries: list[AdaptiveEpochSummary] = []
    selection_trace = ModelSelectionTrace(
        criterion=f"maximize test_multistep mean_fidelity @ H={eval_horizon}",
        best_epoch=max(0, start_epoch - 1),
        best_objective=best_objective,
        best_teacher_forced_fidelity=float("nan"),
        best_multistep_fidelity=float("nan"),
        best_rollout_fidelity=float("nan"),
        rollout_weight=0.0,
        multistep_weight=1.0,
        teacher_forced_weight=0.0,
    )

    last_completed_epoch = history.epochs[-1] if history.epochs else max(0, start_epoch - 1)
    interrupted = False
    try:
        for epoch in range(max(1, start_epoch), config.EPOCHS + 1):
            model.train()
            loss_sum = 0.0
            fidelity_sum = 0.0
            sample_weight = 0
            phase = _training_phase_for_epoch(epoch)
            hybrid_teacher_weight = 1.0 if phase == "teacher_forced" else 0.5
            hybrid_multistep_weight = 0.0 if phase == "teacher_forced" else 0.5
            epoch_horizon = int(config.SEQ_LEN if phase == "teacher_forced" else current_horizon)
            epoch_teacher_steps = int(
                epoch_horizon
                if phase == "teacher_forced"
                else _effective_multistep_teacher_steps(epoch_horizon)
            )
            offset_loss_sums = np.zeros(epoch_horizon, dtype=np.float64)
            offset_fidelity_sums = np.zeros(epoch_horizon, dtype=np.float64)
            offset_weight_sums = np.zeros(epoch_horizon, dtype=np.float64)
            offset_counts = np.zeros(epoch_horizon, dtype=np.float64)

            for batch_idx, (inputs, targets, target_states, params) in enumerate(loader, start=1):
                inputs = inputs.to(config.DEVICE)
                targets = targets.to(config.DEVICE)
                target_states = target_states.to(config.DEVICE)
                params = params.to(config.DEVICE)

                batch_size = int(inputs.shape[0])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    teacher_loss, teacher_fidelity, teacher_stats = _teacher_forced_training_loss(
                        model=model,
                        criterion=criterion,
                        inputs=inputs,
                        target_states=target_states,
                        params=params,
                    )
                    if phase == "teacher_forced":
                        multistep_loss = torch.zeros_like(teacher_loss)
                        multistep_fidelity = torch.zeros_like(teacher_fidelity)
                        batch_stats = teacher_stats
                    else:
                        multistep_loss, multistep_fidelity, batch_stats, _ = _multistep_training_loss(
                            model=model,
                            criterion=criterion,
                            inputs=inputs,
                            targets=targets,
                            target_states=target_states,
                            params=params,
                            horizon_limit=epoch_horizon,
                            teacher_steps_override=epoch_teacher_steps,
                            epoch=epoch,
                            batch_idx=batch_idx,
                        )
                    total_loss = (
                        hybrid_teacher_weight * teacher_loss
                        + hybrid_multistep_weight * multistep_loss
                    )
                    total_fidelity = (
                        hybrid_teacher_weight * teacher_fidelity
                        + hybrid_multistep_weight * multistep_fidelity
                    )

                scaler.scale(total_loss).backward()

                if config.GRAD_CLIP_MAX_NORM > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)

                scaler.step(optimizer)
                scaler.update()
                if scheduler.last_epoch < scheduler.total_steps:
                    scheduler.step()

                weighted_batch_count = batch_size
                loss_sum += float(total_loss.item()) * weighted_batch_count
                fidelity_sum += float(total_fidelity.item()) * weighted_batch_count
                sample_weight += weighted_batch_count
                offset_loss_sums += np.asarray(batch_stats.mean_offset_losses, dtype=np.float64) * batch_size
                offset_fidelity_sums += np.asarray(batch_stats.mean_offset_fidelities, dtype=np.float64) * batch_size
                offset_weight_sums += np.asarray(batch_stats.mean_offset_weights, dtype=np.float64) * batch_size
                offset_counts += float(batch_size)

                if (
                    config.CHECKPOINT_EVERY_BATCH > 0
                    and batch_idx % config.CHECKPOINT_EVERY_BATCH == 0
                ):
                    _save_last_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        history=history,
                        epoch=last_completed_epoch,
                        best_objective=best_objective,
                        best_state=best_state,
                    )

            epoch_loss = loss_sum / max(1, sample_weight)
            epoch_fidelity = fidelity_sum / max(1, sample_weight)
            adaptive_summary = _summarize_epoch_adaptive_stats(
                epoch=epoch,
                horizon=epoch_horizon,
                teacher_steps=epoch_teacher_steps,
                loss_sums=offset_loss_sums,
                fidelity_sums=offset_fidelity_sums,
                weight_sums=offset_weight_sums,
                counts=offset_counts,
            )
            adaptive_epoch_summaries.append(adaptive_summary)
            history.epochs.append(epoch)
            history.train_loss.append(epoch_loss)
            history.train_fidelity.append(epoch_fidelity)
            last_completed_epoch = epoch
            teacher_metric = None
            multistep_metric = None
            if validation_states is not None:
                if validation_params is None:
                    raise ValueError("validation_params e' richiesto quando validation_states non e' None.")
                teacher_metric = evaluate_teacher_forced(
                    model,
                    validation_states,
                    validation_params,
                    latent_states=latent_cache.test_latent_states,
                )
                multistep_metric = evaluate_multistep(
                    model,
                    validation_states,
                    validation_params,
                    latent_states=latent_cache.test_latent_states,
                    horizon_limit=eval_horizon,
                    teacher_steps_override=_effective_multistep_teacher_steps(eval_horizon),
                )
                epoch_objective = float(multistep_metric.mean_fidelity)
                if teacher_metric.loss + float(config.MULTISTEP_H_PLATEAU_MIN_DELTA) < plateau_best_tf_loss:
                    plateau_best_tf_loss = float(teacher_metric.loss)
                    plateau_epochs = 0
                else:
                    plateau_epochs += 1
            else:
                epoch_objective = -float(epoch_loss)

            if epoch_objective > best_objective + best_metric_delta:
                best_objective = epoch_objective
                best_state = copy.deepcopy(model.state_dict())
                selection_trace = ModelSelectionTrace(
                    criterion=f"maximize test_multistep mean_fidelity @ H={eval_horizon}"
                    if validation_states is not None
                    else "minimize train loss",
                    best_epoch=int(epoch),
                    best_objective=float(best_objective),
                    best_teacher_forced_fidelity=float("nan")
                    if teacher_metric is None
                    else float(teacher_metric.mean_fidelity),
                    best_multistep_fidelity=float("nan")
                    if multistep_metric is None
                    else float(multistep_metric.mean_fidelity),
                    best_rollout_fidelity=float("nan"),
                    rollout_weight=0.0,
                    multistep_weight=1.0,
                    teacher_forced_weight=0.0,
                )
                early_stop_counter = 0
                if config.SAVE_MODEL:
                    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                    _safe_atomic_torch_save(
                        model.state_dict(),
                        config.CHECKPOINT_PATH,
                        label="best model checkpoint",
                    )
            elif validation_states is not None and epoch >= int(config.EARLY_STOPPING_MIN_EPOCHS):
                early_stop_counter += 1

            log_every = max(1, min(10, config.EPOCHS // 20 if config.EPOCHS > 20 else 1))
            if epoch <= 3 or epoch == config.EPOCHS or epoch % log_every == 0:
                phase_label = "teacher-forced" if phase == "teacher_forced" else "hybrid-50/50"
                if teacher_metric is not None and multistep_metric is not None:
                    print(
                        f"  Epoca {epoch:4d}/{config.EPOCHS} | "
                        f"fase={phase_label:14s} | "
                        f"loss={epoch_loss:.6f} | fidelity={epoch_fidelity:.6f} | "
                        f"H_train={epoch_horizon:2d} | H_eval={eval_horizon:2d} | "
                        f"teacher_steps={epoch_teacher_steps:2d} | "
                        f"val(tf/ms)=({teacher_metric.mean_fidelity:.3f}/{multistep_metric.mean_fidelity:.3f}) | "
                        f"score={epoch_objective:.3f} | plateau={plateau_epochs:4d} | "
                        f"es={early_stop_counter:4d} | lr={optimizer.param_groups[0]['lr']:.2e}"
                    )
                else:
                    print(
                        f"  Epoca {epoch:4d}/{config.EPOCHS} | "
                        f"fase={phase_label:14s} | "
                        f"loss={epoch_loss:.6f} | fidelity={epoch_fidelity:.6f} | "
                        f"H={epoch_horizon:2d} | teacher_steps={epoch_teacher_steps:2d} | "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}"
                    )

            if (
                phase == "hybrid"
                and validation_states is not None
                and current_horizon < int(config.MULTISTEP_H_MAX)
                and plateau_epochs >= int(config.MULTISTEP_H_PLATEAU_PATIENCE)
            ):
                current_horizon += 1
                plateau_epochs = 0
                print(
                    f"  Curriculum H: aumento orizzonte a {current_horizon} "
                    f"(plateau teacher-forced sul validation split)."
                )

            if epoch % config.CHECKPOINT_EVERY_EPOCH == 0:
                _save_last_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    history=history,
                    epoch=epoch,
                    best_objective=best_objective,
                    best_state=best_state,
                )
            if (
                validation_states is not None
                and epoch >= int(config.EARLY_STOPPING_MIN_EPOCHS)
                and early_stop_counter >= int(config.EARLY_STOPPING_PATIENCE)
            ):
                print(
                    "Early stopping: nessun miglioramento della metrica "
                    "`test_multistep` entro la patience configurata."
                )
                break
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterruzione manuale rilevata (Ctrl+C): salvo checkpoint e genero i risultati correnti...")
        _save_last_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            epoch=last_completed_epoch,
            best_objective=best_objective,
            best_state=best_state,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    if config.SAVE_MODEL:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _safe_atomic_torch_save(
            model.state_dict(),
            config.CHECKPOINT_PATH,
            label="final best model checkpoint",
        )
        _save_last_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            epoch=last_completed_epoch,
            best_objective=best_objective,
            best_state=best_state,
        )

    adaptive_trace = AdaptiveTrainingTrace(
        enabled=int(config.MULTISTEP_H_START) < int(config.MULTISTEP_H_MAX),
        initial_horizon=initial_horizon,
        initial_teacher_steps=initial_teacher_steps,
        final_horizon=int(current_horizon),
        final_teacher_steps=int(_effective_multistep_teacher_steps(int(current_horizon))),
        epoch_summaries=adaptive_epoch_summaries,
    )
    return history, adaptive_trace, selection_trace, autoencoder_trace, latent_cache


@torch.no_grad()
def evaluate_teacher_forced(
    model: QuantumLatentSequenceModel,
    states: torch.Tensor,
    params: torch.Tensor,
    latent_states: torch.Tensor | None = None,
) -> EvaluationResult:
    model.eval()
    criterion = NegativeLogFidelityLoss()
    if latent_states is None:
        latent_states = cache_latent_trajectories(
            states,
            params,
            encoder_fn=model.encode_states,
            batch_size=config.AUTOENCODER_BATCH_SIZE,
            device=config.DEVICE,
        )
    loader = build_latent_loader(latent_states, states, params, shuffle=False)

    total_loss = 0.0
    total_fidelity = 0.0
    total_sequences = 0
    per_step_sum = torch.zeros(states.shape[1] - 1, dtype=torch.float64)

    for inputs, _, targets, batch_params in loader:
        inputs = inputs.to(config.DEVICE)
        targets = targets.to(config.DEVICE)
        batch_params = batch_params.to(config.DEVICE)
        predicted_latent = model.predict_latent(inputs, batch_params)
        predicted = model.decode_latents(predicted_latent.reshape(-1, predicted_latent.shape[-1])).reshape(targets.shape)
        loss, mean_fidelity, fidelity_matrix = criterion(predicted, targets)

        batch_size = int(inputs.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_fidelity += float(mean_fidelity.item()) * batch_size
        total_sequences += batch_size
        per_step_sum += fidelity_matrix.mean(dim=0).cpu().double() * batch_size

    curve = (per_step_sum / max(1, total_sequences)).tolist()
    return EvaluationResult(
        loss=total_loss / max(1, total_sequences),
        mean_fidelity=total_fidelity / max(1, total_sequences),
        fidelity_curve=curve,
        coverage_curve=[1.0 for _ in curve],
    )


@torch.no_grad()
def evaluate_multistep(
    model: QuantumLatentSequenceModel,
    states: torch.Tensor,
    params: torch.Tensor,
    latent_states: torch.Tensor | None = None,
    horizon_limit: int | None = None,
    teacher_steps_override: int | None = None,
) -> EvaluationResult:
    model.eval()
    pred_steps = int(states.shape[1]) - 1
    horizon = max(1, min(int(config.MULTISTEP_H if horizon_limit is None else horizon_limit), pred_steps))
    if latent_states is None:
        latent_states = cache_latent_trajectories(
            states,
            params,
            encoder_fn=model.encode_states,
            batch_size=config.AUTOENCODER_BATCH_SIZE,
            device=config.DEVICE,
        )
    loader = build_latent_loader(latent_states, states, params, shuffle=False)

    loss_sum = torch.zeros(pred_steps, dtype=torch.float64)
    fidelity_sum = torch.zeros(pred_steps, dtype=torch.float64)
    counts = torch.zeros(pred_steps, dtype=torch.float64)

    for inputs, targets, target_states, batch_params in loader:
        inputs = inputs.to(config.DEVICE)
        targets = targets.to(config.DEVICE)
        target_states = target_states.to(config.DEVICE)
        batch_params = batch_params.to(config.DEVICE)

        for t_start in range(1, pred_steps + 1):
            current_h = min(horizon, pred_steps - (t_start - 1))
            current_context = inputs[:, :t_start, :]
            target_start = t_start - 1
            teacher_steps = (
                current_h
                if teacher_steps_override is None
                else max(0, min(int(teacher_steps_override), current_h))
            )

            for step_idx in range(current_h):
                out = model.predict_latent(current_context, batch_params)
                next_pred = out[:, -1:, :]
                step_target = target_states[:, target_start + step_idx : target_start + step_idx + 1, :]
                decoded_step = model.decode_latents(next_pred[:, 0, :])
                step_fidelity = quantum_fidelity(decoded_step, step_target[:, 0, :]).cpu().double()
                curve_index = target_start + step_idx
                fidelity_sum[curve_index] += step_fidelity.sum()
                loss_sum[curve_index] += (
                    -torch.log(step_fidelity.clamp(min=config.LOG_FIDELITY_EPS))
                ).sum()
                counts[curve_index] += float(step_fidelity.shape[0])

                if step_idx + 1 >= current_h:
                    continue

                if step_idx + 1 < teacher_steps:
                    next_context = targets[:, target_start + step_idx : target_start + step_idx + 1, :]
                else:
                    next_context = next_pred
                current_context = torch.cat([current_context, next_context], dim=1)

    fidelity_curve: list[float] = []
    coverage_curve: list[float] = []
    for index in range(pred_steps):
        if counts[index] == 0:
            fidelity_curve.append(float("nan"))
            coverage_curve.append(0.0)
        else:
            fidelity_curve.append(float(fidelity_sum[index] / counts[index]))
            coverage_curve.append(1.0)

    valid_mask = counts > 0
    if valid_mask.any():
        mean_loss = float(loss_sum[valid_mask].sum().item() / counts[valid_mask].sum().item())
        mean_fidelity = float(fidelity_sum[valid_mask].sum().item() / counts[valid_mask].sum().item())
    else:
        mean_loss = float("nan")
        mean_fidelity = float("nan")

    return EvaluationResult(
        loss=mean_loss,
        mean_fidelity=mean_fidelity,
        fidelity_curve=fidelity_curve,
        coverage_curve=coverage_curve,
    )


@torch.no_grad()
def evaluate_autoregressive(
    model: QuantumLatentSequenceModel,
    states: torch.Tensor,
    params: torch.Tensor,
    latent_states: torch.Tensor | None = None,
    warmup_states: int = 1,
) -> EvaluationResult:
    if warmup_states < 1:
        raise ValueError(f"warmup_states deve essere >= 1, ricevuto {warmup_states}")
    if warmup_states >= states.shape[1]:
        raise ValueError(
            f"warmup_states={warmup_states} deve essere < num_states={states.shape[1]}"
        )

    model.eval()
    pred_steps = states.shape[1] - 1
    batch_size = max(1, int(config.BATCH_SIZE) // 2)
    if latent_states is None:
        latent_states = cache_latent_trajectories(
            states,
            params,
            encoder_fn=model.encode_states,
            batch_size=config.AUTOENCODER_BATCH_SIZE,
            device=config.DEVICE,
        )

    loss_sum = torch.zeros(pred_steps, dtype=torch.float64)
    fidelity_sum = torch.zeros(pred_steps, dtype=torch.float64)
    counts = torch.zeros(pred_steps, dtype=torch.float64)

    for start in range(0, int(states.shape[0]), batch_size):
        stop = min(int(states.shape[0]), start + batch_size)
        true_states = states[start:stop].to(config.DEVICE)
        true_latents = latent_states[start:stop].to(config.DEVICE)
        batch_params = params[start:stop].to(config.DEVICE)
        context = true_latents[:, :warmup_states, :]

        for target_index in range(warmup_states, true_states.shape[1]):
            predicted_next = model.predict_latent(context, batch_params)[:, -1, :]
            decoded_next = model.decode_latents(predicted_next)
            fidelity = quantum_fidelity(decoded_next, true_states[:, target_index, :]).cpu().double()
            curve_index = target_index - 1
            fidelity_sum[curve_index] += fidelity.sum()
            loss_sum[curve_index] += (-torch.log(fidelity.clamp(min=config.LOG_FIDELITY_EPS))).sum()
            counts[curve_index] += float(fidelity.shape[0])
            context = torch.cat([context, predicted_next.unsqueeze(1)], dim=1)

    fidelity_curve: list[float] = []
    coverage_curve: list[float] = []
    for index in range(pred_steps):
        if counts[index] == 0:
            fidelity_curve.append(float("nan"))
            coverage_curve.append(0.0)
        else:
            fidelity_curve.append(float(fidelity_sum[index] / counts[index]))
            coverage_curve.append(1.0)

    valid_mask = counts > 0
    if valid_mask.any():
        mean_loss = float((loss_sum[valid_mask] / counts[valid_mask]).mean().item())
        mean_fidelity = float((fidelity_sum[valid_mask] / counts[valid_mask]).mean().item())
    else:
        mean_loss = float("nan")
        mean_fidelity = float("nan")

    return EvaluationResult(
        loss=mean_loss,
        mean_fidelity=mean_fidelity,
        fidelity_curve=fidelity_curve,
        coverage_curve=coverage_curve,
    )


@torch.no_grad()
def compute_train_observable_curves(
    model: QuantumSequencePredictor,
    states: torch.Tensor,
    params: torch.Tensor,
    warmup_states: int = 1,
) -> ObservableComparisonCurves:
    return compute_observable_curves(
        model=model,
        states=states,
        params=params,
        warmup_states=warmup_states,
        time_step=float(config.TIME_STEP),
    )


def _average_observable_curve(
    totals: torch.Tensor,
    counts: torch.Tensor,
) -> np.ndarray:
    averaged = torch.full_like(totals, float("nan"), dtype=torch.float64)
    valid = counts > 0
    averaged[valid] = totals[valid] / counts[valid]
    return averaged.cpu().numpy()


@torch.no_grad()
def compute_observable_curves(
    model: QuantumLatentSequenceModel,
    states: torch.Tensor,
    params: torch.Tensor,
    latent_states: torch.Tensor | None = None,
    warmup_states: int = 1,
    time_step: float = float(config.TIME_STEP),
) -> ObservableComparisonCurves:
    """
    Confronto tra traiettorie esatte, predizioni multi-step e rollout libero.
    """
    if warmup_states < 1:
        raise ValueError(f"warmup_states deve essere >= 1, ricevuto {warmup_states}")
    if warmup_states >= states.shape[1]:
        raise ValueError(
            f"warmup_states={warmup_states} deve essere < num_states={states.shape[1]}"
        )

    device = torch.device(config.DEVICE)
    model.eval()
    z_eigs, zz_nn_eigs, _, x_flip_idx = precompute_observables(config.N_QUBITS, device)
    z_eigs = z_eigs.to(device)
    zz_nn_eigs = zz_nn_eigs.to(device)
    x_flip_idx = [idx.to(device) for idx in x_flip_idx]
    if latent_states is None:
        latent_states = cache_latent_trajectories(
            states,
            params,
            encoder_fn=model.encode_states,
            batch_size=config.AUTOENCODER_BATCH_SIZE,
            device=config.DEVICE,
        )

    num_states = int(states.shape[1])
    total_sequences = int(states.shape[0])
    pred_steps = num_states - 1
    batch_size = max(1, int(config.BATCH_SIZE) // 2)

    exact_mz = torch.zeros(num_states, dtype=torch.float64, device=device)
    exact_mx = torch.zeros(num_states, dtype=torch.float64, device=device)
    exact_cz = torch.zeros(num_states, dtype=torch.float64, device=device)
    exact_entropy = torch.zeros(num_states, dtype=torch.float64, device=device)
    multistep_mz = torch.zeros(num_states, dtype=torch.float64, device=device)
    multistep_mx = torch.zeros(num_states, dtype=torch.float64, device=device)
    multistep_cz = torch.zeros(num_states, dtype=torch.float64, device=device)
    multistep_entropy = torch.zeros(num_states, dtype=torch.float64, device=device)
    multistep_counts = torch.zeros(num_states, dtype=torch.float64, device=device)
    rollout_mz = torch.zeros(num_states, dtype=torch.float64, device=device)
    rollout_mx = torch.zeros(num_states, dtype=torch.float64, device=device)
    rollout_cz = torch.zeros(num_states, dtype=torch.float64, device=device)
    rollout_entropy = torch.zeros(num_states, dtype=torch.float64, device=device)
    rollout_counts = torch.zeros(num_states, dtype=torch.float64, device=device)

    for start in range(0, total_sequences, batch_size):
        stop = min(total_sequences, start + batch_size)
        true_states = states[start:stop].to(device)
        true_latents = latent_states[start:stop].to(device)
        batch_params = params[start:stop].to(device)

        for t in range(num_states):
            mz, mx, cz = batch_observables_tfim(
                true_states[:, t, :],
                z_eigs,
                zz_nn_eigs,
                x_flip_idx,
            )
            exact_mz[t] += mz.double().sum()
            exact_mx[t] += mx.double().sum()
            exact_cz[t] += cz.double().sum()
            entropy = batch_average_entanglement_entropy(true_states[:, t, :], n_qubits=config.N_QUBITS)
            exact_entropy[t] += entropy.double().sum()

        mz0, mx0, cz0 = batch_observables_tfim(
            true_states[:, 0, :],
            z_eigs,
            zz_nn_eigs,
            x_flip_idx,
        )
        entropy0 = batch_average_entanglement_entropy(true_states[:, 0, :], n_qubits=config.N_QUBITS)
        multistep_mz[0] += mz0.double().sum()
        multistep_mx[0] += mx0.double().sum()
        multistep_cz[0] += cz0.double().sum()
        multistep_entropy[0] += entropy0.double().sum()
        multistep_counts[0] += float(true_states.shape[0])

        for start_index in range(pred_steps):
            horizon = min(int(config.MULTISTEP_H), pred_steps - start_index)
            teacher_steps = _effective_multistep_teacher_steps(horizon)
            context = true_latents[:, : start_index + 1, :]

            for step_offset in range(horizon):
                target_state_index = start_index + step_offset + 1
                predicted_next_latent = model.predict_latent(context, batch_params)[:, -1, :]
                predicted_next = model.decode_latents(predicted_next_latent)
                mz, mx, cz = batch_observables_tfim(
                    predicted_next,
                    z_eigs,
                    zz_nn_eigs,
                    x_flip_idx,
                )
                entropy = batch_average_entanglement_entropy(predicted_next, n_qubits=config.N_QUBITS)
                multistep_mz[target_state_index] += mz.double().sum()
                multistep_mx[target_state_index] += mx.double().sum()
                multistep_cz[target_state_index] += cz.double().sum()
                multistep_entropy[target_state_index] += entropy.double().sum()
                multistep_counts[target_state_index] += float(true_states.shape[0])

                if step_offset + 1 >= horizon:
                    continue

                if step_offset + 1 < teacher_steps:
                    next_context = true_latents[:, target_state_index : target_state_index + 1, :]
                else:
                    next_context = predicted_next_latent.unsqueeze(1)
                context = torch.cat([context, next_context], dim=1)

        context = true_latents[:, :warmup_states, :]
        for t in range(warmup_states):
            mz, mx, cz = batch_observables_tfim(
                true_states[:, t, :],
                z_eigs,
                zz_nn_eigs,
                x_flip_idx,
            )
            entropy = batch_average_entanglement_entropy(true_states[:, t, :], n_qubits=config.N_QUBITS)
            rollout_mz[t] += mz.double().sum()
            rollout_mx[t] += mx.double().sum()
            rollout_cz[t] += cz.double().sum()
            rollout_entropy[t] += entropy.double().sum()
            rollout_counts[t] += float(true_states.shape[0])

        for t in range(warmup_states, num_states):
            predicted_next_latent = model.predict_latent(context, batch_params)[:, -1, :]
            predicted_next = model.decode_latents(predicted_next_latent)
            mz, mx, cz = batch_observables_tfim(
                predicted_next,
                z_eigs,
                zz_nn_eigs,
                x_flip_idx,
            )
            entropy = batch_average_entanglement_entropy(predicted_next, n_qubits=config.N_QUBITS)
            rollout_mz[t] += mz.double().sum()
            rollout_mx[t] += mx.double().sum()
            rollout_cz[t] += cz.double().sum()
            rollout_entropy[t] += entropy.double().sum()
            rollout_counts[t] += float(true_states.shape[0])
            context = torch.cat([context, predicted_next_latent.unsqueeze(1)], dim=1)

    scale = float(max(1, total_sequences))
    time_indices = np.arange(num_states, dtype=np.float64)
    physical_time = time_indices * float(time_step)

    inv_scale = 1.0 / scale
    return ObservableComparisonCurves(
        time_indices=time_indices,
        physical_time=physical_time,
        mz_exact=(exact_mz * inv_scale).cpu().numpy(),
        mz_multistep=_average_observable_curve(multistep_mz, multistep_counts),
        mz_rollout=_average_observable_curve(rollout_mz, rollout_counts),
        mx_exact=(exact_mx * inv_scale).cpu().numpy(),
        mx_multistep=_average_observable_curve(multistep_mx, multistep_counts),
        mx_rollout=_average_observable_curve(rollout_mx, rollout_counts),
        cz_exact=(exact_cz * inv_scale).cpu().numpy(),
        cz_multistep=_average_observable_curve(multistep_cz, multistep_counts),
        cz_rollout=_average_observable_curve(rollout_cz, rollout_counts),
        entropy_exact=(exact_entropy * inv_scale).cpu().numpy(),
        entropy_multistep=_average_observable_curve(multistep_entropy, multistep_counts),
        entropy_rollout=_average_observable_curve(rollout_entropy, rollout_counts),
    )


def plot_train_observables(curves: ObservableComparisonCurves, warmup_states: int):
    plot_observable_curves(
        curves=curves,
        warmup_states=warmup_states,
        output_path=config.OBSERVABLES_PLOT_PATH,
        title=(
            f"Osservabili | training set | "
            f"{curves.time_indices.size} stati per traiettoria, warmup rollout={warmup_states}"
        ),
    )


def plot_observable_curves(
    curves: ObservableComparisonCurves,
    warmup_states: int,
    output_path,
    title: str,
):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.0), sharex=True)
    axes = axes.reshape(-1)

    t = curves.physical_time
    axes[0].plot(t, curves.mz_exact, label="Esatto (Hamiltoniana)", color="#1f618d", linewidth=2.2)
    axes[0].plot(t, curves.mz_multistep, label="Predetto (multi-step)", color="#117a65", linewidth=2.2)
    axes[0].plot(t, curves.mz_rollout, label="Predetto (rollout)", color="#b03a2e", linewidth=2.2)
    axes[0].set_title(r"Magnetizzazione $m^z = \frac{1}{N}\sum_i \langle Z_i \rangle$")
    axes[0].set_ylabel(r"$m^z$")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(t, curves.mx_exact, label="Esatto (Hamiltoniana)", color="#1f618d", linewidth=2.2)
    axes[1].plot(t, curves.mx_multistep, label="Predetto (multi-step)", color="#117a65", linewidth=2.2)
    axes[1].plot(t, curves.mx_rollout, label="Predetto (rollout)", color="#b03a2e", linewidth=2.2)
    axes[1].set_title(r"Magnetizzazione $m^x = \frac{1}{N}\sum_i \langle X_i \rangle$")
    axes[1].set_ylabel(r"$m^x$")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)

    axes[2].plot(t, curves.cz_exact, label="Esatto (Hamiltoniana)", color="#1f618d", linewidth=2.2)
    axes[2].plot(t, curves.cz_multistep, label="Predetto (multi-step)", color="#117a65", linewidth=2.2)
    axes[2].plot(t, curves.cz_rollout, label="Predetto (rollout)", color="#b03a2e", linewidth=2.2)
    axes[2].set_title(
        r"Correlazione NN $c^z = \frac{2}{N(N-1)}\sum_{\langle i,j\rangle}\langle Z_i Z_j\rangle$"
    )
    axes[2].set_ylabel(r"$c^z$")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=9)

    axes[3].plot(t, curves.entropy_exact, label="Esatto (Hamiltoniana)", color="#1f618d", linewidth=2.2)
    axes[3].plot(t, curves.entropy_multistep, label="Predetto (multi-step)", color="#117a65", linewidth=2.2)
    axes[3].plot(t, curves.entropy_rollout, label="Predetto (rollout)", color="#b03a2e", linewidth=2.2)
    axes[3].set_title(r"Entanglement entropy media $\bar{S}_A$")
    axes[3].set_ylabel(r"$\bar{S}_A$")
    axes[3].grid(alpha=0.25)
    axes[3].legend(frameon=False, fontsize=9)

    axes[2].set_xlabel(r"Tempo $t = k\,\Delta t$ (indice stato $k$)")
    axes[3].set_xlabel(r"Tempo $t = k\,\Delta t$ (indice stato $k$)")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def exposure_bias_detected(
    teacher_forced_curve: list[float],
    autoregressive_curve: list[float],
) -> bool:
    tf = torch.tensor(teacher_forced_curve, dtype=torch.float32)
    ar = torch.tensor(autoregressive_curve, dtype=torch.float32)
    valid = torch.isfinite(ar)
    if valid.sum() < 4:
        return False

    tf_valid = tf[valid]
    ar_valid = ar[valid]
    tail = max(1, ar_valid.numel() // 4)
    gap = float((tf_valid[-tail:] - ar_valid[-tail:]).mean().item())
    drop = float((ar_valid[:tail].mean() - ar_valid[-tail:].mean()).item())
    return gap >= config.EXPOSURE_BIAS_GAP_THRESHOLD and drop >= config.EXPOSURE_BIAS_DROP_THRESHOLD


def resolve_partial_warmup_steps(prediction_steps: int) -> list[int]:
    raw = config.PARTIAL_WARMUP_STEPS.strip()
    if raw == "none":
        return []

    if raw != "auto":
        values: list[int] = []
        for token in raw.replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            value = int(token)
            if 0 <= value < prediction_steps:
                values.append(value)
        return sorted(set(values))

    candidates = {
        max(1, prediction_steps // 4),
        max(1, prediction_steps // 2),
        max(1, (3 * prediction_steps) // 4),
    }
    return sorted(value for value in candidates if value < prediction_steps)


def plot_training_curves(history: TrainingHistory):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history.epochs, history.train_loss, color="#b03a2e", linewidth=2.0)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Neg. log fidelity loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(history.epochs, history.train_fidelity, color="#1f618d", linewidth=2.0)
    axes[1].set_title("Training Fidelity")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Fidelity media")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(config.TRAINING_CURVES_PATH, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_autoencoder_training_curves(trace: AutoencoderTrainingTrace):
    config.AUTOENCODER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(trace.epochs, trace.train_loss, label="train", color="#b03a2e", linewidth=2.0)
    axes[0].plot(trace.epochs, trace.validation_loss, label="validation", color="#7d6608", linewidth=2.0)
    axes[0].set_title("Autoencoder Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Neg. log fidelity loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(trace.epochs, trace.train_fidelity, label="train", color="#1f618d", linewidth=2.0)
    axes[1].plot(trace.epochs, trace.validation_fidelity, label="validation", color="#117a65", linewidth=2.0)
    axes[1].set_title("Autoencoder Fidelity")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Fidelity media")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(config.AUTOENCODER_TRAINING_CURVES_PATH, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def benchmark_latent_pipeline(
    model: QuantumLatentSequenceModel,
    states: torch.Tensor,
    params: torch.Tensor,
    latent_states: torch.Tensor,
    *,
    autoencoder_trace: AutoencoderTrainingTrace,
) -> PipelineBenchmark:
    started = time.perf_counter()
    evaluate_teacher_forced(model, states, params)
    teacher_eval_uncached_seconds = float(time.perf_counter() - started)

    started = time.perf_counter()
    evaluate_teacher_forced(model, states, params, latent_states=latent_states)
    teacher_eval_cached_seconds = float(time.perf_counter() - started)

    input_feature_dim = 2 * int(config.DIM_2N) - 1
    embedding_dim = int(latent_states.shape[-1])
    return PipelineBenchmark(
        input_feature_dim=input_feature_dim,
        embedding_dim=embedding_dim,
        compression_ratio=float(input_feature_dim / max(1, embedding_dim)),
        autoencoder_training_seconds=float(autoencoder_trace.training_seconds),
        train_cache_seconds=0.0,
        test_cache_seconds=0.0,
        teacher_eval_uncached_seconds=teacher_eval_uncached_seconds,
        teacher_eval_cached_seconds=teacher_eval_cached_seconds,
    )
