import math
import time
import os
import gc
import json
import sys

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # Backend non-interattivo per salvare PNG
import matplotlib.pyplot as plt

import config
from input import QuantumStateDataset, QuantumTrajectoryWindowDataset, generate_quantum_dynamics_dataset
from predictor import QuantumStatePredictor, QuantumFidelityLoss
from trainer import AdvancedTrainer
from observables import precompute_observables, batch_observables


# ===== Directory per i risultati =====
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
generated_plots = []


def _format_time_index(t_idx: int) -> str:
    return f"t={t_idx} ({t_idx * config.DT:.3f})"


def _print_temporal_learning_map(dataset):
    print("  Semantica temporale del training:")
    print(
        f"    - traiettoria grezza: psi(0), psi(1), ..., psi({config.SEQ_LEN}) "
        f"con dt={config.DT}"
    )
    print("    - split dataset: inputs[t]=psi(t), targets[t]=psi(t+1)")

    if config.TRAINING_MODE == "rollout_window" and isinstance(dataset, QuantumTrajectoryWindowDataset):
        n_examples = min(config.TEMPORAL_LOG_EXAMPLES, dataset.windows_per_trajectory)
        print(
            "    - il DataLoader mescola finestre provenienti da Hamiltoniane, stati iniziali "
            "e rollout index diversi"
        )
        print(f"    - esempi dalle prime finestre della traiettoria 0 (tot finestre per traiettoria={dataset.windows_per_trajectory}):")
        for idx in range(n_examples):
            info = dataset.describe_window(idx)
            print(
                f"      window {idx}: psi[{info['context_start_t']}..{info['context_end_t']}] "
                f"({_format_time_index(info['context_start_t'])} -> {_format_time_index(info['context_end_t'])}) "
                f"-> target psi[{info['target_t']}] ({_format_time_index(info['target_t'])})"
            )
    else:
        example_steps = sorted({0, min(1, config.SEQ_LEN - 1), min(2, config.SEQ_LEN - 1), config.SEQ_LEN - 1})
        print("    - full_sequence: grazie alla maschera causale, il token t vede solo 0..t e impara psi(t+1)")
        for t_idx in example_steps:
            print(
                f"      token {t_idx}: contesto psi[0..{t_idx}] "
                f"(fino a {_format_time_index(t_idx)}) -> target psi[{t_idx + 1}] "
                f"({_format_time_index(t_idx + 1)})"
            )


# ============================================================
#  1. HEADER + GENERAZIONE DATI
# ============================================================
print("=" * 70)
print("  QUANTUM SEQUENCE PREDICTION — TRAINING PIPELINE")
print("=" * 70)
print(f"  Hamiltoniana:    {config.HAMILTONIAN_TYPE} (fissata)")
print(f"  Qubit:           {config.N_QUBITS}  (dim. Hilbert: {config.DIM_2N})")
print(f"  Sequenza:        SEQ_LEN={config.SEQ_LEN} (stati totali={config.SEQ_LEN + 1})  (dt={config.DT})")
print(f"  Rollout:         T1={config.T1} (context)  T2={config.T2} (horizon)  (max={config.SEQ_LEN + 1})")
print(f"  Modello:         d={config.D_MODEL}, heads={config.NUM_HEADS}, layers={config.NUM_LAYERS}")
print(f"  Training data:   {config.B_TRAIN} H × {config.S_TRAIN} stati = {config.B_TRAIN * config.S_TRAIN}")
print(f"  Test data:       {config.B_TEST} H × {config.S_TEST} stati = {config.B_TEST * config.S_TEST}")
print(f"  Epoche:          {config.EPOCHS},  batch={config.BATCH_SIZE},  lr={config.LEARNING_RATE}")
print(f"  Scheduler:       {config.LR_SCHEDULER_TYPE} (warmup={config.LR_WARMUP_EPOCHS})")
print(f"  Training mode:   {config.TRAINING_MODE}")
print(f"  Early stopping:  {'ON' if config.EARLY_STOPPING_ENABLED else 'OFF'} (patience={config.EARLY_STOPPING_PATIENCE})")
print(f"  EMA:             {'ON' if config.EMA_ENABLED else 'OFF'} (decay={config.EMA_DECAY})")
print(f"  Grad clip:       {config.GRAD_CLIP_MAX_NORM},  weight decay: {config.WEIGHT_DECAY}")
print(f"  torch.compile:   {'ON' if config.TORCH_COMPILE else 'OFF'}")
print(f"  Resume:          {'ON' if config.RESUME_TRAINING else 'OFF'}")
print(f"  Device:          {config.DEVICE}")
micro_bs_print = config.MICRO_BATCH_SIZE if config.MICRO_BATCH_SIZE > 0 else config.BATCH_SIZE
print(f"  Memory safe:     {'ON' if config.MEMORY_SAFE_MODE else 'OFF'} (micro-batch={micro_bs_print})")
print(
    f"  Logging:         dataset/H={config.DATASET_LOG_EVERY_N_HAMILTONIANS}, "
    f"train-step={config.TRAIN_LOG_EVERY_N_STEPS}, eval-step={config.EVAL_LOG_EVERY_N_STEPS}"
)
print("=" * 70)

print("\n[1/6] Generando dataset...")
t0 = time.time()
train_inputs, train_targets = generate_quantum_dynamics_dataset(
    B=config.B_TRAIN, S=config.S_TRAIN, dataset_name="train"
)
test_inputs, test_targets = generate_quantum_dynamics_dataset(
    B=config.B_TEST, S=config.S_TEST, dataset_name="test"
)
gen_time = time.time() - t0
print(f"  Train: {train_inputs.shape}  |  Test: {test_inputs.shape}  ({gen_time:.1f}s)")

pin_memory = bool(config.PIN_MEMORY and config.DEVICE == "cuda")
num_workers = 0 if config.MEMORY_SAFE_MODE else config.NUM_WORKERS
loader_kwargs = dict(
    batch_size=config.BATCH_SIZE,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
if num_workers > 0:
    loader_kwargs["persistent_workers"] = True

train_full_dataset = QuantumStateDataset(train_inputs, train_targets)
test_full_dataset = QuantumStateDataset(test_inputs, test_targets)

if config.TRAINING_MODE == "rollout_window":
    train_dataset = QuantumTrajectoryWindowDataset(
        train_inputs, train_targets, context_len=config.T1, rollout_horizon=config.T2
    )
    test_dataset = QuantumTrajectoryWindowDataset(
        test_inputs, test_targets, context_len=config.T1, rollout_horizon=config.T2
    )
    print(
        f"  Training objective: finestre T1={config.T1} -> next-step "
        f"(campioni train={len(train_dataset):,}, test={len(test_dataset):,})"
    )
else:
    train_dataset = train_full_dataset
    test_dataset = test_full_dataset
    print(
        f"  Training objective: teacher forcing full-sequence "
        f"(campioni train={len(train_dataset):,}, test={len(test_dataset):,})"
    )

if config.VERBOSE_STARTUP_LOGS:
    _print_temporal_learning_map(train_dataset)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    **loader_kwargs,
)
test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    **loader_kwargs,
)
train_eval_loader = DataLoader(
    train_full_dataset,
    shuffle=False,
    **loader_kwargs,
)
test_eval_loader = DataLoader(
    test_full_dataset,
    shuffle=False,
    **loader_kwargs,
)

if config.VERBOSE_STARTUP_LOGS:
    print(
        f"  DataLoader train: batches={len(train_loader)}, shuffle=True, "
        f"workers={num_workers}, pin_memory={pin_memory}"
    )
    print(
        f"  DataLoader test:  batches={len(test_loader)}, shuffle=False, "
        f"workers={num_workers}, pin_memory={pin_memory}"
    )

# Liberiamo i tensori raw — i DataLoader hanno copie interne nei Dataset
del train_inputs, train_targets, test_inputs, test_targets
gc.collect()


# ============================================================
#  2. MODELLO + TRAINER AVANZATO
# ============================================================
print("\n[2/6] Inizializzando modello e trainer avanzato...")
model = QuantumStatePredictor().to(config.DEVICE)
criterion = QuantumFidelityLoss()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parametri: {n_params:,}  |  Device: {config.DEVICE}")

trainer = AdvancedTrainer(
    model=model,
    criterion=criterion,
    train_loader=train_loader,
    test_loader=test_loader,
    device=config.DEVICE,
)


# ============================================================
#  2b. RESUME DA CHECKPOINT (se abilitato)
# ============================================================
if config.RESUME_TRAINING:
    print(f"\n  [RESUME] Verifico checkpoint in '{config.LAST_CHECKPOINT_PATH}'...")
    compatible, msg, ckpt_data = AdvancedTrainer.check_checkpoint_compatibility(
        config.LAST_CHECKPOINT_PATH
    )
    if compatible and ckpt_data is not None:
        print(f"  [RESUME] {msg}")
        trainer.resume_from_checkpoint(ckpt_data)
    else:
        print(f"  [RESUME] {msg}")
        print(f"  [RESUME] Si parte da zero.")
else:
    print(f"\n  [RESUME] Disabilitato (RESUME_TRAINING=False). Training da zero.")


# ============================================================
#  3. TRAINING (con sistema avanzato)
# ============================================================
print(f"\n[3/6] Training avanzato (max {config.EPOCHS} epoche)...\n")

history = trainer.train(epochs=config.EPOCHS, verbose=True)
total_train_time = trainer.total_train_time

trainer.print_training_summary()


# ============================================================
#  4. VALUTAZIONE FINALE SUL TEST SET
# ============================================================
print(f"\n[4/6] Valutazione finale sul Test Set (teacher forcing full-sequence)...")

model = trainer.model
model.eval()

non_blocking = pin_memory
fid_sum_per_step = torch.zeros(config.SEQ_LEN, dtype=torch.float64)
fid_sq_sum_per_step = torch.zeros(config.SEQ_LEN, dtype=torch.float64)
fid_total_sum = 0.0
total_samples = 0
sample_fid_means = []
max_plot_samples = max(1, int(config.MAX_FIDELITY_PLOT_SAMPLES))
per_step_plot_values = [[] for _ in range(config.SEQ_LEN)]

with torch.no_grad():
    for step, (x_batch_cpu, y_batch_cpu) in enumerate(test_eval_loader, start=1):
        x_batch = x_batch_cpu.to(config.DEVICE, non_blocking=non_blocking)
        y_batch = y_batch_cpu.to(config.DEVICE, non_blocking=non_blocking)

        pred = model(x_batch)
        overlap = torch.sum(y_batch.conj() * pred, dim=-1)
        fid_per_step = torch.abs(overlap) ** 2
        fid_cpu = fid_per_step.cpu()

        fid_sum_per_step += fid_cpu.sum(dim=0, dtype=torch.float64)
        fid_sq_sum_per_step += (fid_cpu.double().pow(2)).sum(dim=0)
        fid_total_sum += fid_cpu.sum().item()
        total_samples += fid_cpu.shape[0]

        if len(sample_fid_means) < max_plot_samples:
            remaining = max_plot_samples - len(sample_fid_means)
            sample_fid_means.extend(fid_cpu.mean(dim=-1).tolist()[:remaining])

        for t in range(config.SEQ_LEN):
            if len(per_step_plot_values[t]) < max_plot_samples:
                remaining = max_plot_samples - len(per_step_plot_values[t])
                per_step_plot_values[t].extend(fid_cpu[:, t].tolist()[:remaining])

        del x_batch_cpu, y_batch_cpu, x_batch, y_batch, pred, overlap, fid_per_step, fid_cpu

        if config.GC_COLLECT_EVERY_N_STEPS > 0 and step % config.GC_COLLECT_EVERY_N_STEPS == 0:
            gc.collect()

gc.collect()

total_points = max(1, total_samples * config.SEQ_LEN)
mean_fid = fid_total_sum / total_points
mean_loss = 1.0 - mean_fid
test_ppl = math.exp(mean_loss)

print(f"  Loss: {mean_loss:.6f}  |  PPL: {test_ppl:.6f}")

# Dati per grafici
actual_epochs = len(history["train_loss"])
den = max(1, total_samples)
fid_per_step_mean = (fid_sum_per_step / den).float()
fid_var_per_step = (fid_sq_sum_per_step / den) - fid_per_step_mean.double().pow(2)
fid_per_step_std = torch.sqrt(torch.clamp(fid_var_per_step, min=0.0)).float()

final_tr_loss = history["train_loss"][-1]
final_te_loss = history["test_loss"][-1]
final_tr_fid = history["train_fidelity"][-1]
final_te_fid = history["test_fidelity"][-1]
best_test_fid = max(history["test_fidelity"])
best_test_epoch = history["test_fidelity"].index(best_test_fid) + 1


# ============================================================
#  4b. ROLLOUT AUTOREGRESSIVO (FIDELITY + OSSERVABILI) SU TRAIN/TEST
# ============================================================
obs_curves = None
test_rollout_curves = None
if config.OBSERVABLE_PLOTS_ENABLED:
    print(f"\n[4b/6] Rollout autoregressivo: fidelity + osservabili (Train/Test)...")

    t1 = int(config.T1)
    t2 = int(config.T2)
    device = torch.device(config.DEVICE)

    z_eigs, zz_nn_eigs, x_flip_idx = precompute_observables(config.N_QUBITS, device=device)

    max_samples = int(config.OBSERVABLE_PLOT_SAMPLES)
    if max_samples > 0:
        print(f"  Max traiettorie per split: {max_samples}")
    else:
        print(f"  Max traiettorie per split: ALL (puo' essere lento)")

    def _rollout_curves(data_loader, split_name: str):
        fid_sum = torch.zeros(t2, device=device, dtype=torch.float64)
        fid_sumsq = torch.zeros(t2, device=device, dtype=torch.float64)

        mz_pred_sum = torch.zeros(t2, device=device, dtype=torch.float64)
        mz_pred_sumsq = torch.zeros(t2, device=device, dtype=torch.float64)
        mz_true_sum = torch.zeros(t2, device=device, dtype=torch.float64)
        mz_true_sumsq = torch.zeros(t2, device=device, dtype=torch.float64)

        mx_pred_sum = torch.zeros(t2, device=device, dtype=torch.float64)
        mx_pred_sumsq = torch.zeros(t2, device=device, dtype=torch.float64)
        mx_true_sum = torch.zeros(t2, device=device, dtype=torch.float64)
        mx_true_sumsq = torch.zeros(t2, device=device, dtype=torch.float64)

        cz_pred_sum = torch.zeros(t2, device=device, dtype=torch.float64)
        cz_pred_sumsq = torch.zeros(t2, device=device, dtype=torch.float64)
        cz_true_sum = torch.zeros(t2, device=device, dtype=torch.float64)
        cz_true_sumsq = torch.zeros(t2, device=device, dtype=torch.float64)

        count = 0

        model.eval()
        with torch.no_grad():
            for step, (x_batch_cpu, y_batch_cpu) in enumerate(data_loader, start=1):
                if max_samples > 0 and count >= max_samples:
                    break

                if max_samples > 0:
                    remaining = max(0, max_samples - count)
                    if remaining == 0:
                        break
                    x_batch_cpu = x_batch_cpu[:remaining]
                    y_batch_cpu = y_batch_cpu[:remaining]

                batch_size = x_batch_cpu.shape[0]
                count += batch_size

                x_batch = x_batch_cpu.to(config.DEVICE, non_blocking=non_blocking)
                # target[t] corrisponde allo stato a tempo (t+1); vogliamo tempi T1..T1+T2-1
                y_future = y_batch_cpu[:, t1 - 1 : t1 - 1 + t2, :].to(
                    config.DEVICE, non_blocking=non_blocking
                )

                # Finestra iniziale (ground-truth): stati a tempi 0..T1-1
                window = x_batch[:, :t1, :].clone()
                window_buf = torch.empty_like(window)

                for k in range(t2):
                    pred_seq = model(window)  # (batch, T1, dim), pred_seq[:, t] ~ state(t+1)
                    next_pred = pred_seq[:, -1, :]  # (batch, dim)
                    next_true = y_future[:, k, :]  # (batch, dim)

                    overlap = torch.sum(next_true.conj() * next_pred, dim=-1)  # (batch,)
                    fid = (torch.abs(overlap) ** 2).double()  # (batch,)
                    fid_sum[k] += fid.sum()
                    fid_sumsq[k] += fid.pow(2).sum()

                    mz_p, mx_p, cz_p = batch_observables(next_pred, z_eigs, zz_nn_eigs, x_flip_idx)
                    mz_t, mx_t, cz_t = batch_observables(next_true, z_eigs, zz_nn_eigs, x_flip_idx)

                    mz_pred_sum[k] += mz_p.double().sum()
                    mz_pred_sumsq[k] += mz_p.double().pow(2).sum()
                    mz_true_sum[k] += mz_t.double().sum()
                    mz_true_sumsq[k] += mz_t.double().pow(2).sum()

                    mx_pred_sum[k] += mx_p.double().sum()
                    mx_pred_sumsq[k] += mx_p.double().pow(2).sum()
                    mx_true_sum[k] += mx_t.double().sum()
                    mx_true_sumsq[k] += mx_t.double().pow(2).sum()

                    cz_pred_sum[k] += cz_p.double().sum()
                    cz_pred_sumsq[k] += cz_p.double().pow(2).sum()
                    cz_true_sum[k] += cz_t.double().sum()
                    cz_true_sumsq[k] += cz_t.double().pow(2).sum()

                    # Shift a finestra fissa (T1): drop del primo, append del predetto
                    if t1 > 1:
                        window_buf[:, :-1, :] = window[:, 1:, :]
                    window_buf[:, -1, :] = next_pred
                    window, window_buf = window_buf, window

                    del pred_seq, next_pred, next_true, overlap, fid, mz_p, mx_p, cz_p, mz_t, mx_t, cz_t

                del x_batch_cpu, y_batch_cpu, x_batch, y_future, window, window_buf
                if config.GC_COLLECT_EVERY_N_STEPS > 0 and step % config.GC_COLLECT_EVERY_N_STEPS == 0:
                    gc.collect()

        den_roll = max(1, count)

        def _mean_std(sum_, sumsq_):
            mean_ = sum_ / den_roll
            var_ = (sumsq_ / den_roll) - mean_.pow(2)
            std_ = torch.sqrt(torch.clamp(var_, min=0.0))
            return mean_.cpu().float().numpy(), std_.cpu().float().numpy()

        curves = {
            "fid": _mean_std(fid_sum, fid_sumsq),
            "mz_pred": _mean_std(mz_pred_sum, mz_pred_sumsq),
            "mz_true": _mean_std(mz_true_sum, mz_true_sumsq),
            "mx_pred": _mean_std(mx_pred_sum, mx_pred_sumsq),
            "mx_true": _mean_std(mx_true_sum, mx_true_sumsq),
            "cz_pred": _mean_std(cz_pred_sum, cz_pred_sumsq),
            "cz_true": _mean_std(cz_true_sum, cz_true_sumsq),
            "n_samples": count,
        }

        fid_mean = curves["fid"][0]
        if fid_mean.size > 0:
            print(
                f"  [ROLL] {split_name:5s} n={count:4d}  "
                f"F_mean={float(fid_mean.mean()):.4f}  F_last={float(fid_mean[-1]):.4f}"
            )
        else:
            print(f"  [ROLL] {split_name:5s} n={count:4d}  (no data)")

        return curves

    obs_curves = _rollout_curves(train_eval_loader, "Train")
    test_rollout_curves = _rollout_curves(test_eval_loader, "Test")


# ============================================================
#  5. GRAFICI
# ============================================================
print(f"\n[5/6] Generando grafici...")

epochs_range = range(1, actual_epochs + 1)

# --- Fig 1: Loss (train + test) ---
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle("Quantum Sequence Prediction — Training Report (Advanced)", fontsize=14, fontweight="bold")

ax = axes[0, 0]
ax.plot(epochs_range, history["train_loss"], label="Train Loss", linewidth=2)
ax.plot(epochs_range, history["test_loss"], label="Test Loss", linewidth=2, linestyle="--")
if best_test_epoch <= actual_epochs:
    ax.axvline(best_test_epoch, color="green", linestyle=":", alpha=0.5, label=f"Best @ {best_test_epoch}")
ax.set_xlabel("Epoca")
ax.set_ylabel("Loss (1 - Fidelity)")
ax.set_title("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Fig 2: Fidelity (train + test) ---
ax = axes[0, 1]
ax.plot(epochs_range, history["train_fidelity"], label="Train Fidelity", linewidth=2)
ax.plot(epochs_range, history["test_fidelity"], label="Test Fidelity", linewidth=2, linestyle="--")
if best_test_epoch <= actual_epochs:
    ax.axvline(best_test_epoch, color="green", linestyle=":", alpha=0.5, label=f"Best @ {best_test_epoch}")
ax.set_xlabel("Epoca")
ax.set_ylabel("Fidelity")
ax.set_title("Fidelity")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Fig 3: Learning Rate Schedule ---
ax = axes[0, 2]
if "lr" in history:
    ax.plot(epochs_range, history["lr"], linewidth=2, color="darkorange")
    ax.set_yscale("log")
ax.set_xlabel("Epoca")
ax.set_ylabel("Learning Rate")
ax.set_title("Learning Rate Schedule")
ax.grid(True, alpha=0.3)

# --- Fig 4: Perplexity (train + test) ---
ax = axes[1, 0]
ax.plot(epochs_range, history["train_ppl"], label="Train PPL", linewidth=2)
ax.plot(epochs_range, history["test_ppl"], label="Test PPL", linewidth=2, linestyle="--")
ax.set_xlabel("Epoca")
ax.set_ylabel("Perplexity")
ax.set_title("Perplexity (exp(Loss))")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Fig 5: Fidelity per step temporale ---
ax = axes[1, 1]
steps = list(range(config.SEQ_LEN))
means = fid_per_step_mean.numpy()
stds = fid_per_step_std.numpy()
ax.bar(steps, means, yerr=stds, capsize=3, alpha=0.7, color="steelblue", edgecolor="navy")
ax.set_xlabel("Step temporale")
ax.set_ylabel("Fidelity")
ax.set_title("Fidelity per step temporale (Test)")
ax.set_xticks(steps)
ax.grid(True, alpha=0.3, axis="y")

# --- Fig 6: Generalization gap nel tempo ---
ax = axes[1, 2]
gap_history = [tr - te for tr, te in zip(history["train_fidelity"], history["test_fidelity"])]
ax.plot(epochs_range, gap_history, linewidth=2, color="red")
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.fill_between(epochs_range, 0, gap_history, alpha=0.2, color="red")
ax.set_xlabel("Epoca")
ax.set_ylabel("Gap (Train - Test Fidelity)")
ax.set_title("Generalization Gap")
ax.grid(True, alpha=0.3)

plt.tight_layout()
training_report_path = os.path.join(RESULTS_DIR, "training_report.png")
plt.savefig(training_report_path, dpi=150, bbox_inches="tight")
plt.close()
generated_plots.append(training_report_path)
print(f"  [PLOT] {training_report_path}")

# --- Fig OBS: Osservabili su rollout (train) ---
if obs_curves is not None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"Osservabili su rollout (Train Set)  -  n={obs_curves['n_samples']}",
        fontsize=13,
        fontweight="bold",
    )

    t_axis = (np.arange(config.T2, dtype=np.float32) + config.T1) * float(config.DT)

    # m^z
    ax = axes[0]
    mz_pred_mean, mz_pred_std = obs_curves["mz_pred"]
    mz_true_mean, mz_true_std = obs_curves["mz_true"]
    ax.plot(t_axis, mz_true_mean, label="Esatto", color="black", linewidth=2)
    ax.fill_between(t_axis, mz_true_mean - mz_true_std, mz_true_mean + mz_true_std, color="black", alpha=0.15)
    ax.plot(t_axis, mz_pred_mean, label="Rollout Modello", color="tab:blue", linewidth=2)
    ax.fill_between(
        t_axis,
        mz_pred_mean - mz_pred_std,
        mz_pred_mean + mz_pred_std,
        color="tab:blue",
        alpha=0.20,
    )
    ax.set_ylabel("m^z")
    ax.set_title(r"$m^z(t)=\frac{1}{N}\sum_i \langle Z_i\rangle$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # m^x
    ax = axes[1]
    mx_pred_mean, mx_pred_std = obs_curves["mx_pred"]
    mx_true_mean, mx_true_std = obs_curves["mx_true"]
    ax.plot(t_axis, mx_true_mean, label="Esatto", color="black", linewidth=2)
    ax.fill_between(t_axis, mx_true_mean - mx_true_std, mx_true_mean + mx_true_std, color="black", alpha=0.15)
    ax.plot(t_axis, mx_pred_mean, label="Rollout Modello", color="tab:orange", linewidth=2)
    ax.fill_between(
        t_axis,
        mx_pred_mean - mx_pred_std,
        mx_pred_mean + mx_pred_std,
        color="tab:orange",
        alpha=0.20,
    )
    ax.set_ylabel("m^x")
    ax.set_title(r"$m^x(t)=\frac{1}{N}\sum_i \langle X_i\rangle$")
    ax.grid(True, alpha=0.3)

    # c^z (nearest neighbor)
    ax = axes[2]
    cz_pred_mean, cz_pred_std = obs_curves["cz_pred"]
    cz_true_mean, cz_true_std = obs_curves["cz_true"]
    ax.plot(t_axis, cz_true_mean, label="Esatto", color="black", linewidth=2)
    ax.fill_between(t_axis, cz_true_mean - cz_true_std, cz_true_mean + cz_true_std, color="black", alpha=0.15)
    ax.plot(t_axis, cz_pred_mean, label="Rollout Modello", color="tab:green", linewidth=2)
    ax.fill_between(
        t_axis,
        cz_pred_mean - cz_pred_std,
        cz_pred_mean + cz_pred_std,
        color="tab:green",
        alpha=0.20,
    )
    ax.set_ylabel("c^z")
    ax.set_xlabel("Tempo (t)")
    ax.set_title(r"$c^z(t)=\frac{1}{N-1}\sum_i \langle Z_i Z_{i+1}\rangle$")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    observables_path = os.path.join(RESULTS_DIR, "observables_rollout_train.png")
    plt.savefig(observables_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots.append(observables_path)
    print(f"  [PLOT] {observables_path}")

# --- Fig ROLL: Fidelity + osservabili su rollout (test) ---
if test_rollout_curves is not None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    fig.suptitle(
        f"Rollout (Test Set)  -  n={test_rollout_curves['n_samples']}  (T1={config.T1}, T2={config.T2})",
        fontsize=13,
        fontweight="bold",
    )

    t_axis = (np.arange(config.T2, dtype=np.float32) + config.T1) * float(config.DT)

    # Fidelity
    ax = axes[0]
    fid_mean, fid_std = test_rollout_curves["fid"]
    ax.plot(t_axis, fid_mean, label="Fidelity rollout", color="tab:purple", linewidth=2)
    ax.fill_between(t_axis, fid_mean - fid_std, fid_mean + fid_std, color="tab:purple", alpha=0.20)
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(r"$F(t)=|\langle \psi_{\mathrm{true}}(t) | \psi_{\mathrm{pred}}(t)\rangle|^2$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # m^z
    ax = axes[1]
    mz_pred_mean, mz_pred_std = test_rollout_curves["mz_pred"]
    mz_true_mean, mz_true_std = test_rollout_curves["mz_true"]
    ax.plot(t_axis, mz_true_mean, label="Esatto", color="black", linewidth=2)
    ax.fill_between(t_axis, mz_true_mean - mz_true_std, mz_true_mean + mz_true_std, color="black", alpha=0.15)
    ax.plot(t_axis, mz_pred_mean, label="Rollout Modello", color="tab:blue", linewidth=2)
    ax.fill_between(
        t_axis,
        mz_pred_mean - mz_pred_std,
        mz_pred_mean + mz_pred_std,
        color="tab:blue",
        alpha=0.20,
    )
    ax.set_ylabel("m^z")
    ax.set_title(r"$m^z(t)=\frac{1}{N}\sum_i \langle Z_i\rangle$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # m^x
    ax = axes[2]
    mx_pred_mean, mx_pred_std = test_rollout_curves["mx_pred"]
    mx_true_mean, mx_true_std = test_rollout_curves["mx_true"]
    ax.plot(t_axis, mx_true_mean, label="Esatto", color="black", linewidth=2)
    ax.fill_between(t_axis, mx_true_mean - mx_true_std, mx_true_mean + mx_true_std, color="black", alpha=0.15)
    ax.plot(t_axis, mx_pred_mean, label="Rollout Modello", color="tab:orange", linewidth=2)
    ax.fill_between(
        t_axis,
        mx_pred_mean - mx_pred_std,
        mx_pred_mean + mx_pred_std,
        color="tab:orange",
        alpha=0.20,
    )
    ax.set_ylabel("m^x")
    ax.set_title(r"$m^x(t)=\frac{1}{N}\sum_i \langle X_i\rangle$")
    ax.grid(True, alpha=0.3)

    # c^z (nearest neighbor)
    ax = axes[3]
    cz_pred_mean, cz_pred_std = test_rollout_curves["cz_pred"]
    cz_true_mean, cz_true_std = test_rollout_curves["cz_true"]
    ax.plot(t_axis, cz_true_mean, label="Esatto", color="black", linewidth=2)
    ax.fill_between(t_axis, cz_true_mean - cz_true_std, cz_true_mean + cz_true_std, color="black", alpha=0.15)
    ax.plot(t_axis, cz_pred_mean, label="Rollout Modello", color="tab:green", linewidth=2)
    ax.fill_between(
        t_axis,
        cz_pred_mean - cz_pred_std,
        cz_pred_mean + cz_pred_std,
        color="tab:green",
        alpha=0.20,
    )
    ax.set_ylabel("c^z")
    ax.set_xlabel("Tempo (t)")
    ax.set_title(r"$c^z(t)=\frac{1}{N-1}\sum_i \langle Z_i Z_{i+1}\rangle$")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    rollout_test_path = os.path.join(RESULTS_DIR, "rollout_test_report.png")
    plt.savefig(rollout_test_path, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots.append(rollout_test_path)
    print(f"  [PLOT] {rollout_test_path}")

# --- Fig 5: Distribuzione fidelity sul test set ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribuzione Fidelity sul Test Set", fontsize=13, fontweight="bold")

ax = axes[0]
fid_means = np.asarray(sample_fid_means, dtype=np.float32)
if fid_means.size == 0:
    fid_means = np.array([0.0], dtype=np.float32)
ax.hist(fid_means, bins=40, alpha=0.7, color="steelblue", edgecolor="navy")
ax.axvline(mean_fid, color="red", linestyle="--", linewidth=2, label=f"Media: {mean_fid:.4f}")
ax.set_xlabel("Fidelity media per campione")
ax.set_ylabel("Conteggio")
ax.set_title("Istogramma Fidelity")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
boxplot_data = [vals if vals else [0.0] for vals in per_step_plot_values]
ax.boxplot(boxplot_data,
           tick_labels=[str(t) for t in range(config.SEQ_LEN)])
ax.set_xlabel("Step temporale")
ax.set_ylabel("Fidelity")
ax.set_title("Boxplot Fidelity per step")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fid_dist_path = os.path.join(RESULTS_DIR, "fidelity_distribution.png")
plt.savefig(fid_dist_path, dpi=150, bbox_inches="tight")
plt.close()
generated_plots.append(fid_dist_path)
print(f"  [PLOT] {fid_dist_path}")

# --- Fig 7: LR vs Loss (fase space) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Learning Rate Analysis", fontsize=13, fontweight="bold")

ax = axes[0]
if "lr" in history:
    ax.scatter(history["lr"], history["test_loss"], c=list(epochs_range), cmap="viridis", s=20, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Test Loss")
    ax.set_title("LR vs Test Loss (colorato per epoca)")
    ax.grid(True, alpha=0.3)

ax = axes[1]
if "lr" in history:
    # Tasso di miglioramento della fidelity (derivata discreta)
    fid_improvement = [0] + [
        history["test_fidelity"][i] - history["test_fidelity"][i - 1]
        for i in range(1, actual_epochs)
    ]
    ax.scatter(history["lr"], fid_improvement, c=list(epochs_range), cmap="viridis", s=20, alpha=0.7)
    ax.set_xscale("log")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("dFidelity / epoca")
    ax.set_title("LR vs Tasso di Miglioramento")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
lr_analysis_path = os.path.join(RESULTS_DIR, "lr_analysis.png")
plt.savefig(lr_analysis_path, dpi=150, bbox_inches="tight")
plt.close()
generated_plots.append(lr_analysis_path)
print(f"  [PLOT] {lr_analysis_path}")

print(f"  Plot salvati in {RESULTS_DIR}/ ({len(generated_plots)} file)")
for p in generated_plots:
    print(f"    - {p}")

summary = {
    "results_dir": RESULTS_DIR,
    "plots": [os.path.basename(p) for p in generated_plots],
    "device": config.DEVICE,
    "python": sys.version.split()[0],
    "torch": getattr(torch, "__version__", "unknown"),
    "numpy": getattr(np, "__version__", "unknown"),
    "config": {
        "HAMILTONIAN_TYPE": config.HAMILTONIAN_TYPE,
        "N_QUBITS": int(config.N_QUBITS),
        "DIM_2N": int(config.DIM_2N),
        "DT": float(config.DT),
        "SEQ_LEN": int(config.SEQ_LEN),
        "T1": int(config.T1),
        "T2": int(config.T2),
        "TRAINING_MODE": config.TRAINING_MODE,
        "B_TRAIN": int(config.B_TRAIN),
        "S_TRAIN": int(config.S_TRAIN),
        "B_TEST": int(config.B_TEST),
        "S_TEST": int(config.S_TEST),
        "BATCH_SIZE": int(config.BATCH_SIZE),
        "EPOCHS": int(config.EPOCHS),
        "D_MODEL": int(config.D_MODEL),
        "NUM_HEADS": int(config.NUM_HEADS),
        "NUM_LAYERS": int(config.NUM_LAYERS),
        "DATASET_LOG_EVERY_N_HAMILTONIANS": int(config.DATASET_LOG_EVERY_N_HAMILTONIANS),
        "TRAIN_LOG_EVERY_N_STEPS": int(config.TRAIN_LOG_EVERY_N_STEPS),
        "EVAL_LOG_EVERY_N_STEPS": int(config.EVAL_LOG_EVERY_N_STEPS),
    },
    "training": {
        "actual_epochs": int(actual_epochs),
        "total_train_time_s": float(total_train_time),
        "best_test_epoch": int(best_test_epoch),
        "final_train_loss": float(final_tr_loss),
        "final_test_loss": float(final_te_loss),
        "final_train_fidelity": float(final_tr_fid),
        "final_test_fidelity": float(final_te_fid),
        "mean_train_phase_time_s": float(sum(history["train_phase_time"]) / len(history["train_phase_time"]))
        if history.get("train_phase_time")
        else None,
        "mean_eval_phase_time_s": float(sum(history["eval_phase_time"]) / len(history["eval_phase_time"]))
        if history.get("eval_phase_time")
        else None,
        "mean_train_samples_per_sec": float(sum(history["train_samples_per_sec"]) / len(history["train_samples_per_sec"]))
        if history.get("train_samples_per_sec")
        else None,
    },
    "test_eval": {
        "mean_fidelity": float(mean_fid),
        "mean_loss": float(mean_loss),
        "test_ppl": float(test_ppl),
    },
    "observables_rollout_train": {
        "enabled": bool(config.OBSERVABLE_PLOTS_ENABLED),
        "n_samples": int(obs_curves["n_samples"]) if obs_curves is not None else 0,
        "obs_plot_samples_cfg": int(config.OBSERVABLE_PLOT_SAMPLES),
    },
    "rollout_test": {
        "enabled": bool(config.OBSERVABLE_PLOTS_ENABLED),
        "n_samples": int(test_rollout_curves["n_samples"]) if test_rollout_curves is not None else 0,
        "fidelity_mean_over_horizon": float(test_rollout_curves["fid"][0].mean())
        if test_rollout_curves is not None
        else None,
        "fidelity_last_step": float(test_rollout_curves["fid"][0][-1])
        if test_rollout_curves is not None
        else None,
    },
}

summary_json_path = os.path.join(RESULTS_DIR, "run_summary.json")
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"  [SUMMARY] {summary_json_path}")

summary_txt_path = os.path.join(RESULTS_DIR, "run_summary.txt")
with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write("QUANTUM SEQUENCE PREDICTION - RUN SUMMARY\n")
    f.write("=" * 60 + "\n")
    f.write(f"Device: {summary['device']}\n")
    f.write(f"Python: {summary['python']}  Torch: {summary['torch']}  Numpy: {summary['numpy']}\n")
    f.write("\nConfig:\n")
    for k, v in summary["config"].items():
        f.write(f"  {k}: {v}\n")
    f.write("\nTraining:\n")
    for k, v in summary["training"].items():
        f.write(f"  {k}: {v}\n")
    f.write("\nTest eval:\n")
    for k, v in summary["test_eval"].items():
        f.write(f"  {k}: {v}\n")
    f.write("\nObservables rollout (train):\n")
    for k, v in summary["observables_rollout_train"].items():
        f.write(f"  {k}: {v}\n")
    f.write("\nRollout (test):\n")
    for k, v in summary["rollout_test"].items():
        f.write(f"  {k}: {v}\n")
    f.write("\nPlots:\n")
    for p in summary["plots"]:
        f.write(f"  - {p}\n")
print(f"  [SUMMARY] {summary_txt_path}")


# ============================================================
#  6. RIEPILOGO FINALE
# ============================================================
gap_fid = final_tr_fid - final_te_fid
print(f"\n[6/6] RIEPILOGO FINALE")
print("=" * 50)
print(f"  Test Loss:   {mean_loss:.4f}   PPL: {math.exp(mean_loss):.4f}")
print(f"  Train Loss:  {final_tr_loss:.4f}   PPL: {math.exp(final_tr_loss):.4f}")
print(f"  Gap fidelity: {gap_fid:+.4f}")
print(f"  Tempo:        {total_train_time:.1f}s  ({actual_epochs} ep.)")
print("=" * 50)
