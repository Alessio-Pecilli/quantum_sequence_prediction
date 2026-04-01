from __future__ import annotations

import copy
import gc
import os
import random
import shutil
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from input import QuantumSequenceDataset
from observables import batch_observables_tfim, precompute_observables
from predictor import NegativeLogFidelityLoss, QuantumSequencePredictor, quantum_fidelity


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


@dataclass
class ResumeCheckpointState:
    start_epoch: int
    history: TrainingHistory
    optimizer_state_dict: dict | None
    scheduler_state_dict: dict | None
    best_loss: float | None
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


def build_loader(states: torch.Tensor, shuffle: bool) -> DataLoader:
    dataset = QuantumSequenceDataset(states)
    safe_batch_size = max(1, int(config.BATCH_SIZE) // 2)
    return DataLoader(
        dataset,
        batch_size=safe_batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def build_model() -> QuantumSequencePredictor:
    return QuantumSequencePredictor().to(config.DEVICE)


def _empty_resume_state(reason: str) -> ResumeCheckpointState:
    return ResumeCheckpointState(
        start_epoch=1,
        history=TrainingHistory(epochs=[], train_loss=[], train_fidelity=[]),
        optimizer_state_dict=None,
        scheduler_state_dict=None,
        best_loss=None,
        best_state=None,
        resumed=False,
        reason=reason,
    )


def _checkpoint_config_snapshot() -> dict[str, object]:
    return {
        "N_QUBITS": int(config.N_QUBITS),
        "DIM_2N": int(config.DIM_2N),
        "NUM_STATES": int(config.NUM_STATES),
        "TRAIN_SEQUENCES": int(config.TRAIN_SEQUENCES),
        "BATCH_SIZE": int(config.BATCH_SIZE),
        "EPOCHS": int(config.EPOCHS),
        "LEARNING_RATE": float(config.LEARNING_RATE),
        "WEIGHT_DECAY": float(config.WEIGHT_DECAY),
        "D_MODEL": int(config.D_MODEL),
        "NUM_HEADS": int(config.NUM_HEADS),
        "NUM_LAYERS": int(config.NUM_LAYERS),
        "DIM_FEEDFORWARD": int(config.DIM_FEEDFORWARD),
        "OUTPUT_PARAMETRIZATION": str(config.OUTPUT_PARAMETRIZATION),
        "MULTISTEP_H": int(config.MULTISTEP_H),
        "MULTISTEP_TEACHER_FORCING_STEPS": int(config.MULTISTEP_TEACHER_FORCING_STEPS),
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


def try_resume_from_last_checkpoint(model: QuantumSequencePredictor) -> ResumeCheckpointState:
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

    best_loss = payload.get("best_loss")
    return ResumeCheckpointState(
        start_epoch=max(1, last_completed_epoch + 1),
        history=history,
        optimizer_state_dict=payload.get("optimizer_state_dict"),
        scheduler_state_dict=payload.get("scheduler_state_dict"),
        best_loss=None if best_loss is None else float(best_loss),
        best_state=payload.get("best_state_dict"),
        resumed=True,
        reason=f"resume da {config.LAST_CHECKPOINT_PATH} (ultima epoca completa: {last_completed_epoch})",
    )


def _effective_multistep_teacher_steps(horizon: int, requested_steps: int | None = None) -> int:
    if horizon < 1:
        return 0
    if requested_steps is None:
        requested_steps = int(config.MULTISTEP_TEACHER_FORCING_STEPS)
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
    mean_losses: torch.Tensor,
    mean_fidelities: torch.Tensor,
) -> torch.Tensor:
    weights = torch.ones_like(mean_losses)
    if not config.ADAPTIVE_MULTISTEP_ENABLED or mean_losses.numel() <= 1:
        return weights

    with torch.no_grad():
        detached_losses = mean_losses.detach()
        detached_fidelities = mean_fidelities.detach()
        loss_jumps = torch.zeros_like(detached_losses)
        fidelity_drops = torch.zeros_like(detached_fidelities)
        loss_jumps[1:] = torch.relu(detached_losses[1:] - detached_losses[:-1])
        fidelity_drops[1:] = torch.relu(detached_fidelities[:-1] - detached_fidelities[1:])
        signal = loss_jumps + fidelity_drops
        signal_mean = signal.mean().clamp(min=1e-8)
        normalized_signal = signal / signal_mean
        weights = 1.0 + float(config.ADAPTIVE_WEIGHT_ALPHA) * normalized_signal
        weights = weights.clamp(
            min=float(config.ADAPTIVE_WEIGHT_MIN),
            max=float(config.ADAPTIVE_WEIGHT_MAX),
        )
    return weights.to(mean_losses.device, dtype=mean_losses.dtype)


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


def _multistep_training_loss(
    model: QuantumSequencePredictor,
    criterion: NegativeLogFidelityLoss,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    horizon_limit: int,
    teacher_steps_override: int,
    epoch: int | None = None,
    batch_idx: int | None = None,
):
    per_offset_losses: list[list[torch.Tensor]] = [[] for _ in range(max(1, int(horizon_limit)))]
    per_offset_fidelities: list[list[torch.Tensor]] = [[] for _ in range(max(1, int(horizon_limit)))]
    seq_len = int(inputs.shape[1])
    verbose = bool(config.MULTISTEP_TRAIN_VERBOSE)

    if verbose and epoch is not None and batch_idx is not None:
        print(
            f"[train multistep] epoca={epoch} batch={batch_idx} | "
            f"SEQ_LEN={seq_len} H={horizon_limit} teacher_steps={teacher_steps_override}"
        )

    for start_index in range(seq_len):
        horizon = min(int(horizon_limit), seq_len - start_index)
        teacher_steps = _effective_multistep_teacher_steps(horizon, teacher_steps_override)
        context = inputs[:, : start_index + 1, :]

        if verbose:
            print(
                f"  [start t{start_index}] contesto iniziale: t0..t{start_index} | "
                f"orizzonte effettivo={horizon} | teacher_steps={teacher_steps}"
            )

        for step_offset in range(horizon):
            target_index = start_index + step_offset
            if verbose:
                use_teacher = step_offset + 1 < teacher_steps
                print(f"    - {_describe_multistep_transition(start_index, target_index, use_teacher)}")
            predicted_next = model(context)[:, -1, :]
            step_target = targets[:, target_index : target_index + 1, :]
            step_loss, step_mean_fidelity, _ = criterion(predicted_next.unsqueeze(1), step_target)
            per_offset_losses[step_offset].append(step_loss)
            per_offset_fidelities[step_offset].append(step_mean_fidelity)

            if step_offset + 1 >= horizon:
                continue

            if step_offset + 1 < teacher_steps:
                next_context = step_target
            else:
                next_context = predicted_next.unsqueeze(1)
            context = torch.cat([context, next_context], dim=1)

        if verbose:
            print(f"  [end t{start_index}] completati {horizon} step supervisonati")

    mean_losses = torch.stack([torch.stack(offset_losses).mean() for offset_losses in per_offset_losses])
    mean_fidelities = torch.stack([torch.stack(offset_fidelities).mean() for offset_fidelities in per_offset_fidelities])
    offset_weights = _build_step_weights(mean_losses, mean_fidelities)
    multistep_loss = (offset_weights * mean_losses).sum() / offset_weights.sum().clamp(min=1e-8)
    multistep_fidelity = mean_fidelities.mean()
    stats = BatchAdaptiveStats(
        horizon=int(horizon_limit),
        teacher_steps=int(_effective_multistep_teacher_steps(horizon_limit, teacher_steps_override)),
        mean_offset_losses=[float(value) for value in mean_losses.detach().cpu().tolist()],
        mean_offset_fidelities=[float(value) for value in mean_fidelities.detach().cpu().tolist()],
        mean_offset_weights=[float(value) for value in offset_weights.detach().cpu().tolist()],
    )
    return multistep_loss, multistep_fidelity, stats


def _atomic_torch_save(payload: dict, destination: os.PathLike):
    destination_path = os.fspath(destination)
    tmp_path = f"{destination_path}.tmp"
    torch.save(payload, tmp_path)
    backoff = 0.05
    for attempt in range(4):
        try:
            os.replace(tmp_path, destination_path)
            return
        except (PermissionError, OSError):
            if attempt < 3:
                time.sleep(backoff)
                backoff *= 2.0
                continue
            break
    # Windows: destination may be read-locked (IDE, indexer); replace() can fail while overwrite works.
    try:
        shutil.copyfile(tmp_path, destination_path)
    except OSError:
        torch.save(payload, destination_path)
    try:
        os.unlink(tmp_path)
    except OSError:
        pass


def _save_last_checkpoint(
    model: QuantumSequencePredictor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    history: TrainingHistory,
    epoch: int,
    best_loss: float,
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
        "best_loss": float(best_loss),
        "best_state_dict": best_state,
        "config": {
            **_checkpoint_config_snapshot(),
        },
    }
    _atomic_torch_save(checkpoint_payload, config.LAST_CHECKPOINT_PATH)


def train_model(
    model: QuantumSequencePredictor,
    train_states: torch.Tensor,
    start_epoch: int = 1,
    history: TrainingHistory | None = None,
    optimizer_state_dict: dict | None = None,
    scheduler_state_dict: dict | None = None,
    best_loss: float | None = None,
    best_state: dict | None = None,
) -> tuple[TrainingHistory, AdaptiveTrainingTrace]:
    history = history or TrainingHistory(epochs=[], train_loss=[], train_fidelity=[])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    loader = build_loader(train_states, shuffle=True)
    steps_per_epoch = len(loader)
    use_amp = config.DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # OneCycleLR: warmup sul learning rate per accelerare la convergenza del Transformer.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # 10% warmup
        anneal_strategy="cos",
    )
    criterion = NegativeLogFidelityLoss()

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)
        # last_epoch = batch index; checkpoint "epoch" can still be last_completed_epoch-1
        # if we saved right after the last batch of an epoch. Avoid replaying a full epoch
        # and stepping OneCycleLR past total_steps.
        completed_epochs = int(scheduler.last_epoch) // int(steps_per_epoch)
        start_epoch = max(int(start_epoch), completed_epochs + 1)

    best_loss = float("inf") if best_loss is None else float(best_loss)
    adaptive_controller = _make_adaptive_controller()
    initial_horizon = int(adaptive_controller.current_horizon)
    initial_teacher_steps = int(adaptive_controller.current_teacher_steps)
    adaptive_epoch_summaries: list[AdaptiveEpochSummary] = []

    last_completed_epoch = history.epochs[-1] if history.epochs else max(0, start_epoch - 1)
    interrupted = False
    try:
        for epoch in range(max(1, start_epoch), config.EPOCHS + 1):
            model.train()
            loss_sum = 0.0
            fidelity_sum = 0.0
            sample_count = 0
            epoch_horizon = int(adaptive_controller.current_horizon)
            epoch_teacher_steps = int(adaptive_controller.current_teacher_steps)
            offset_loss_sums = np.zeros(epoch_horizon, dtype=np.float64)
            offset_fidelity_sums = np.zeros(epoch_horizon, dtype=np.float64)
            offset_weight_sums = np.zeros(epoch_horizon, dtype=np.float64)
            offset_counts = np.zeros(epoch_horizon, dtype=np.float64)

            for batch_idx, (inputs, targets) in enumerate(loader, start=1):
                inputs = inputs.to(config.DEVICE)
                targets = targets.to(config.DEVICE)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    total_loss, total_fidelity, batch_stats = _multistep_training_loss(
                        model=model,
                        criterion=criterion,
                        inputs=inputs,
                        targets=targets,
                        horizon_limit=epoch_horizon,
                        teacher_steps_override=epoch_teacher_steps,
                        epoch=epoch,
                        batch_idx=batch_idx,
                    )

                scaler.scale(total_loss).backward()

                if config.GRAD_CLIP_MAX_NORM > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)

                scaler.step(optimizer)
                scaler.update()
                if scheduler.last_epoch < scheduler.total_steps:
                    scheduler.step()

                batch_size = int(inputs.shape[0])
                loss_sum += float(total_loss.item()) * batch_size
                fidelity_sum += float(total_fidelity.item()) * batch_size
                sample_count += batch_size
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
                        best_loss=best_loss,
                        best_state=best_state,
                    )

            epoch_loss = loss_sum / max(1, sample_count)
            epoch_fidelity = fidelity_sum / max(1, sample_count)
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
            _update_adaptive_controller(adaptive_controller, adaptive_summary)
            history.epochs.append(epoch)
            history.train_loss.append(epoch_loss)
            history.train_fidelity.append(epoch_fidelity)
            last_completed_epoch = epoch

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = copy.deepcopy(model.state_dict())

            if epoch == 1 or epoch == config.EPOCHS or epoch % max(1, config.EPOCHS // 10) == 0:
                print(
                    f"  Epoca {epoch:4d}/{config.EPOCHS} | "
                    f"loss={epoch_loss:.6f} | fidelity={epoch_fidelity:.6f} | "
                    f"H={epoch_horizon:2d} | teacher_steps={epoch_teacher_steps:2d} | "
                    f"next_H={adaptive_controller.current_horizon:2d} | "
                    f"next_teacher={adaptive_controller.current_teacher_steps:2d} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if epoch % config.CHECKPOINT_EVERY_EPOCH == 0:
                _save_last_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    history=history,
                    epoch=epoch,
                    best_loss=best_loss,
                    best_state=best_state,
                )
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
            best_loss=best_loss,
            best_state=best_state,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    if config.SAVE_MODEL:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), config.CHECKPOINT_PATH)
        _save_last_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            epoch=last_completed_epoch if interrupted else config.EPOCHS,
            best_loss=best_loss,
            best_state=best_state,
        )

    adaptive_trace = AdaptiveTrainingTrace(
        enabled=bool(config.ADAPTIVE_MULTISTEP_ENABLED),
        initial_horizon=initial_horizon,
        initial_teacher_steps=initial_teacher_steps,
        final_horizon=int(adaptive_controller.current_horizon),
        final_teacher_steps=int(adaptive_controller.current_teacher_steps),
        epoch_summaries=adaptive_epoch_summaries,
    )
    return history, adaptive_trace


@torch.no_grad()
def evaluate_teacher_forced(model: QuantumSequencePredictor, states: torch.Tensor) -> EvaluationResult:
    model.eval()
    criterion = NegativeLogFidelityLoss()
    loader = build_loader(states, shuffle=False)

    total_loss = 0.0
    total_fidelity = 0.0
    total_sequences = 0
    per_step_sum = torch.zeros(states.shape[1] - 1, dtype=torch.float64)

    for inputs, targets in loader:
        inputs = inputs.to(config.DEVICE)
        targets = targets.to(config.DEVICE)
        predicted = model(inputs)
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
    model: QuantumSequencePredictor,
    states: torch.Tensor,
) -> EvaluationResult:
    model.eval()
    pred_steps = int(states.shape[1]) - 1
    loader = DataLoader(
        states,
        batch_size=max(1, int(config.BATCH_SIZE) // 2),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    loss_sum = torch.zeros(pred_steps, dtype=torch.float64)
    fidelity_sum = torch.zeros(pred_steps, dtype=torch.float64)
    counts = torch.zeros(pred_steps, dtype=torch.float64)

    for batch_states in loader:
        true_states = batch_states.to(config.DEVICE)

        for start_index in range(pred_steps):
            horizon = min(int(config.MULTISTEP_H), pred_steps - start_index)
            teacher_steps = _effective_multistep_teacher_steps(horizon)
            context = true_states[:, : start_index + 1, :]

            for step_offset in range(horizon):
                target_state_index = start_index + step_offset + 1
                curve_index = target_state_index - 1
                predicted_next = model(context)[:, -1, :]
                fidelity = quantum_fidelity(
                    predicted_next,
                    true_states[:, target_state_index, :],
                ).cpu().double()
                fidelity_sum[curve_index] += fidelity.sum()
                loss_sum[curve_index] += (
                    -torch.log(fidelity.clamp(min=config.LOG_FIDELITY_EPS))
                ).sum()
                counts[curve_index] += float(fidelity.shape[0])

                if step_offset + 1 >= horizon:
                    continue

                if step_offset + 1 < teacher_steps:
                    next_context = true_states[:, target_state_index : target_state_index + 1, :]
                else:
                    next_context = predicted_next.unsqueeze(1)
                context = torch.cat([context, next_context], dim=1)

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
    model: QuantumSequencePredictor,
    states: torch.Tensor,
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
    loader = DataLoader(
        states,
        batch_size=max(1, int(config.BATCH_SIZE) // 2),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    loss_sum = torch.zeros(pred_steps, dtype=torch.float64)
    fidelity_sum = torch.zeros(pred_steps, dtype=torch.float64)
    counts = torch.zeros(pred_steps, dtype=torch.float64)

    for batch_states in loader:
        true_states = batch_states.to(config.DEVICE)
        context = true_states[:, :warmup_states, :]

        for target_index in range(warmup_states, true_states.shape[1]):
            predicted_next = model(context)[:, -1, :]
            fidelity = quantum_fidelity(predicted_next, true_states[:, target_index, :]).cpu().double()
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
    warmup_states: int = 1,
) -> ObservableComparisonCurves:
    return compute_observable_curves(
        model=model,
        states=states,
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
    model: QuantumSequencePredictor,
    states: torch.Tensor,
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

    num_states = int(states.shape[1])
    total_sequences = int(states.shape[0])
    pred_steps = num_states - 1

    exact_mz = torch.zeros(num_states, dtype=torch.float64, device=device)
    exact_mx = torch.zeros(num_states, dtype=torch.float64, device=device)
    exact_cz = torch.zeros(num_states, dtype=torch.float64, device=device)
    multistep_mz = torch.zeros(num_states, dtype=torch.float64, device=device)
    multistep_mx = torch.zeros(num_states, dtype=torch.float64, device=device)
    multistep_cz = torch.zeros(num_states, dtype=torch.float64, device=device)
    multistep_counts = torch.zeros(num_states, dtype=torch.float64, device=device)
    rollout_mz = torch.zeros(num_states, dtype=torch.float64, device=device)
    rollout_mx = torch.zeros(num_states, dtype=torch.float64, device=device)
    rollout_cz = torch.zeros(num_states, dtype=torch.float64, device=device)
    rollout_counts = torch.zeros(num_states, dtype=torch.float64, device=device)

    loader = DataLoader(
        states,
        batch_size=max(1, int(config.BATCH_SIZE) // 2),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    for batch_states in loader:
        true_states = batch_states.to(device)

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

        mz0, mx0, cz0 = batch_observables_tfim(
            true_states[:, 0, :],
            z_eigs,
            zz_nn_eigs,
            x_flip_idx,
        )
        multistep_mz[0] += mz0.double().sum()
        multistep_mx[0] += mx0.double().sum()
        multistep_cz[0] += cz0.double().sum()
        multistep_counts[0] += float(true_states.shape[0])

        for start_index in range(pred_steps):
            horizon = min(int(config.MULTISTEP_H), pred_steps - start_index)
            teacher_steps = _effective_multistep_teacher_steps(horizon)
            context = true_states[:, : start_index + 1, :]

            for step_offset in range(horizon):
                target_state_index = start_index + step_offset + 1
                predicted_next = model(context)[:, -1, :]
                mz, mx, cz = batch_observables_tfim(
                    predicted_next,
                    z_eigs,
                    zz_nn_eigs,
                    x_flip_idx,
                )
                multistep_mz[target_state_index] += mz.double().sum()
                multistep_mx[target_state_index] += mx.double().sum()
                multistep_cz[target_state_index] += cz.double().sum()
                multistep_counts[target_state_index] += float(true_states.shape[0])

                if step_offset + 1 >= horizon:
                    continue

                if step_offset + 1 < teacher_steps:
                    next_context = true_states[:, target_state_index : target_state_index + 1, :]
                else:
                    next_context = predicted_next.unsqueeze(1)
                context = torch.cat([context, next_context], dim=1)

        context = true_states[:, :warmup_states, :]
        for t in range(warmup_states):
            mz, mx, cz = batch_observables_tfim(
                true_states[:, t, :],
                z_eigs,
                zz_nn_eigs,
                x_flip_idx,
            )
            rollout_mz[t] += mz.double().sum()
            rollout_mx[t] += mx.double().sum()
            rollout_cz[t] += cz.double().sum()
            rollout_counts[t] += float(true_states.shape[0])

        for t in range(warmup_states, num_states):
            predicted_next = model(context)[:, -1, :]
            mz, mx, cz = batch_observables_tfim(
                predicted_next,
                z_eigs,
                zz_nn_eigs,
                x_flip_idx,
            )
            rollout_mz[t] += mz.double().sum()
            rollout_mx[t] += mx.double().sum()
            rollout_cz[t] += cz.double().sum()
            rollout_counts[t] += float(true_states.shape[0])
            context = torch.cat([context, predicted_next.unsqueeze(1)], dim=1)

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
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)

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

    axes[2].set_xlabel(r"Tempo $t = k\,\Delta t$ (indice stato $k$)")

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
