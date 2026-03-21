from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from input import QuantumSequenceDataset
from predictor import ComplexMSELoss, QuantumSequencePredictor, quantum_fidelity


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(states: torch.Tensor, shuffle: bool) -> DataLoader:
    dataset = QuantumSequenceDataset(states)
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=bool(config.PIN_MEMORY and config.DEVICE == "cuda"),
    )


def build_model() -> QuantumSequencePredictor:
    return QuantumSequencePredictor().to(config.DEVICE)


def _scheduled_sampling_probability(epoch: int) -> float:
    progress = min(1.0, float(epoch) / float(config.SCHEDULED_SAMPLING_RAMP_EPOCHS))
    return float(config.SCHEDULED_SAMPLING_MAX_PROB) * progress


def _rollout_training_steps(seq_len: int, epoch: int) -> int:
    progress = min(1.0, float(epoch) / float(config.ROLLOUT_CURRICULUM_EPOCHS))
    return max(1, min(seq_len, int(round(progress * seq_len))))


def _autoregressive_unroll_loss(
    model: QuantumSequencePredictor,
    criterion: ComplexMSELoss,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    scheduled_sampling_prob: float,
):
    context = inputs[:, :1, :]
    per_step_losses: list[torch.Tensor] = []
    per_step_fidelities: list[torch.Tensor] = []

    max_steps = min(int(steps), int(targets.shape[1]))
    for step in range(max_steps):
        predicted_next = model(context)[:, -1, :]
        step_target = targets[:, step : step + 1, :]
        step_loss, step_mean_fidelity, _ = criterion(predicted_next.unsqueeze(1), step_target)
        per_step_losses.append(step_loss)
        per_step_fidelities.append(step_mean_fidelity)

        if step + 1 >= max_steps:
            continue

        gold_next_context = inputs[:, step + 1 : step + 2, :]
        if scheduled_sampling_prob <= 0.0:
            next_context = gold_next_context
        else:
            use_model_mask = (
                torch.rand((inputs.shape[0], 1, 1), device=inputs.device) < scheduled_sampling_prob
            )
            next_context = torch.where(
                use_model_mask,
                predicted_next.detach().unsqueeze(1),
                gold_next_context,
            )
        context = torch.cat([context, next_context], dim=1)

    rollout_loss = torch.stack(per_step_losses).mean()
    rollout_fidelity = torch.stack(per_step_fidelities).mean()
    return rollout_loss, rollout_fidelity


def train_model(model: QuantumSequencePredictor, train_states: torch.Tensor) -> TrainingHistory:
    history = TrainingHistory(epochs=[], train_loss=[], train_fidelity=[])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    loader = build_loader(train_states, shuffle=True)
    steps_per_epoch = len(loader)

    # OneCycleLR: warmup sul learning rate per accelerare la convergenza del Transformer.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # 10% warmup
        anneal_strategy="cos",
    )
    criterion = ComplexMSELoss()

    best_state = None
    best_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        fidelity_sum = 0.0
        sample_count = 0
        scheduled_sampling_prob = _scheduled_sampling_probability(epoch)
        rollout_steps = _rollout_training_steps(config.SEQ_LEN, epoch)

        for inputs, targets in loader:
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            optimizer.zero_grad(set_to_none=True)
            predicted = model(inputs)
            teacher_forced_loss, teacher_forced_fidelity, _ = criterion(predicted, targets)
            total_loss = teacher_forced_loss
            total_fidelity = teacher_forced_fidelity

            if config.ROLLOUT_AUX_WEIGHT > 0.0:
                rollout_loss, rollout_fidelity = _autoregressive_unroll_loss(
                    model=model,
                    criterion=criterion,
                    inputs=inputs,
                    targets=targets,
                    steps=rollout_steps,
                    scheduled_sampling_prob=scheduled_sampling_prob,
                )
                total_loss = teacher_forced_loss + config.ROLLOUT_AUX_WEIGHT * rollout_loss
                total_fidelity = 0.5 * (teacher_forced_fidelity + rollout_fidelity)

            total_loss.backward()

            if config.GRAD_CLIP_MAX_NORM > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)

            optimizer.step()
            scheduler.step()

            batch_size = int(inputs.shape[0])
            loss_sum += float(total_loss.item()) * batch_size
            fidelity_sum += float(total_fidelity.item()) * batch_size
            sample_count += batch_size

        epoch_loss = loss_sum / max(1, sample_count)
        epoch_fidelity = fidelity_sum / max(1, sample_count)
        history.epochs.append(epoch)
        history.train_loss.append(epoch_loss)
        history.train_fidelity.append(epoch_fidelity)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch == config.EPOCHS or epoch % max(1, config.EPOCHS // 10) == 0:
            print(
                f"  Epoca {epoch:4d}/{config.EPOCHS} | "
                f"loss={epoch_loss:.6f} | fidelity={epoch_fidelity:.6f} | "
                f"ss_p={scheduled_sampling_prob:.3f} | rollout_steps={rollout_steps:2d} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    if config.SAVE_MODEL:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), config.CHECKPOINT_PATH)

    return history


@torch.no_grad()
def evaluate_teacher_forced(model: QuantumSequencePredictor, states: torch.Tensor) -> EvaluationResult:
    model.eval()
    criterion = ComplexMSELoss()
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
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=bool(config.PIN_MEMORY and config.DEVICE == "cuda"),
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
    axes[0].set_ylabel("Complex MSE loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(history.epochs, history.train_fidelity, color="#1f618d", linewidth=2.0)
    axes[1].set_title("Training Fidelity")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Fidelity media")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(config.TRAINING_CURVES_PATH, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
