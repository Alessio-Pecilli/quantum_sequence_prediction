import math
import time
import os

import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # Backend non-interattivo per salvare PNG
import matplotlib.pyplot as plt

import config
from input import QuantumStateDataset, generate_quantum_dynamics_dataset
from predictor import QuantumStatePredictor, QuantumFidelityLoss
from trainer import AdvancedTrainer


# ===== Directory per i risultati =====
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
#  1. HEADER + GENERAZIONE DATI
# ============================================================
print("=" * 70)
print("  QUANTUM SEQUENCE PREDICTION — TRAINING PIPELINE")
print("=" * 70)
print(f"  Hamiltoniana:    {config.HAMILTONIAN_TYPE} (fissata)")
print(f"  Qubit:           {config.N_QUBITS}  (dim. Hilbert: {config.DIM_2N})")
print(f"  Sequenza:        {config.SEQ_LEN} step  (dt={config.DT})")
print(f"  Modello:         d={config.D_MODEL}, heads={config.NUM_HEADS}, layers={config.NUM_LAYERS}")
print(f"  Training data:   {config.B_TRAIN} H × {config.S_TRAIN} stati = {config.B_TRAIN * config.S_TRAIN}")
print(f"  Test data:       {config.B_TEST} H × {config.S_TEST} stati = {config.B_TEST * config.S_TEST}")
print(f"  Epoche:          {config.EPOCHS},  batch={config.BATCH_SIZE},  lr={config.LEARNING_RATE}")
print(f"  Scheduler:       {config.LR_SCHEDULER_TYPE} (warmup={config.LR_WARMUP_EPOCHS})")
print(f"  Early stopping:  {'ON' if config.EARLY_STOPPING_ENABLED else 'OFF'} (patience={config.EARLY_STOPPING_PATIENCE})")
print(f"  EMA:             {'ON' if config.EMA_ENABLED else 'OFF'} (decay={config.EMA_DECAY})")
print(f"  Grad clip:       {config.GRAD_CLIP_MAX_NORM},  weight decay: {config.WEIGHT_DECAY}")
print(f"  torch.compile:   {'ON' if config.TORCH_COMPILE else 'OFF'}")
print(f"  Device:          {config.DEVICE}")
print("=" * 70)

print("\n[1/6] Generando dataset...")
t0 = time.time()
train_inputs, train_targets = generate_quantum_dynamics_dataset(
    B=config.B_TRAIN, S=config.S_TRAIN
)
test_inputs, test_targets = generate_quantum_dynamics_dataset(
    B=config.B_TEST, S=config.S_TEST
)
gen_time = time.time() - t0
print(f"  Train: {train_inputs.shape}  |  Test: {test_inputs.shape}  ({gen_time:.1f}s)")

train_inputs = train_inputs.to(config.DEVICE)
train_targets = train_targets.to(config.DEVICE)
test_inputs = test_inputs.to(config.DEVICE)
test_targets = test_targets.to(config.DEVICE)

train_loader = DataLoader(
    QuantumStateDataset(train_inputs, train_targets),
    batch_size=config.BATCH_SIZE, shuffle=True,
    num_workers=config.NUM_WORKERS,
    persistent_workers=config.NUM_WORKERS > 0,
)
test_loader = DataLoader(
    QuantumStateDataset(test_inputs, test_targets),
    batch_size=config.BATCH_SIZE, shuffle=False,
    num_workers=config.NUM_WORKERS,
    persistent_workers=config.NUM_WORKERS > 0,
)


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
#  3. TRAINING (con sistema avanzato)
# ============================================================
print(f"\n[3/6] Training avanzato (max {config.EPOCHS} epoche)...\n")

history = trainer.train(epochs=config.EPOCHS, verbose=True)
total_train_time = trainer.total_train_time

trainer.print_training_summary()


# ============================================================
#  4. VALUTAZIONE FINALE DETTAGLIATA SUL TEST SET
# ============================================================
print(f"\n[4/6] Valutazione finale dettagliata sul Test Set...\n")

model.eval()

all_fidelities = []        # fidelity per campione e step
all_losses_per_sample = []  # loss per campione
all_norms = []              # norma degli stati predetti

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        pred = model(x_batch)

        # Fidelity per campione e step: (batch, seq_len)
        overlap = torch.sum(y_batch.conj() * pred, dim=-1)
        fid_per_step = torch.abs(overlap) ** 2
        all_fidelities.append(fid_per_step.cpu())

        # Loss per campione: media sugli step
        loss_per_sample = 1.0 - fid_per_step.mean(dim=-1)
        all_losses_per_sample.append(loss_per_sample.cpu())

        # Norma stati predetti
        norms = torch.norm(pred, dim=-1)
        all_norms.append(norms.cpu())

all_fidelities = torch.cat(all_fidelities, dim=0)          # (N_test, seq_len)
all_losses_per_sample = torch.cat(all_losses_per_sample)    # (N_test,)
all_norms = torch.cat(all_norms, dim=0)                     # (N_test, seq_len)

# --- Metriche globali ---
mean_fid = all_fidelities.mean().item()
std_fid = all_fidelities.std().item()
median_fid = all_fidelities.median().item()
min_fid = all_fidelities.min().item()
max_fid = all_fidelities.max().item()
q25_fid = torch.quantile(all_fidelities.float(), 0.25).item()
q75_fid = torch.quantile(all_fidelities.float(), 0.75).item()

mean_loss = all_losses_per_sample.mean().item()
std_loss = all_losses_per_sample.std().item()

mean_norm = all_norms.mean().item()
std_norm = all_norms.std().item()

# Fidelity per step temporale
fid_per_step_mean = all_fidelities.mean(dim=0)  # (seq_len,)
fid_per_step_std = all_fidelities.std(dim=0)

# Percentuale campioni sopra soglie
pct_above_90 = (all_fidelities.mean(dim=-1) > 0.90).float().mean().item() * 100
pct_above_80 = (all_fidelities.mean(dim=-1) > 0.80).float().mean().item() * 100
pct_above_50 = (all_fidelities.mean(dim=-1) > 0.50).float().mean().item() * 100

print(f"  {'-' * 50}")
print(f"  METRICHE TEST SET ({len(all_losses_per_sample)} campioni)")
print(f"  {'-' * 50}")
print()
print(f"  FIDELITY (quanto lo stato predetto somiglia al target)")
print(f"    Media:     {mean_fid:.6f}")
print(f"    Std:       {std_fid:.6f}")
print(f"    Mediana:   {median_fid:.6f}")
print(f"    Min:       {min_fid:.6f}")
print(f"    Max:       {max_fid:.6f}")
print(f"    Q25:       {q25_fid:.6f}")
print(f"    Q75:       {q75_fid:.6f}")
print()
print(f"  LOSS (1 - Fidelity)")
print(f"    Media:     {mean_loss:.6f}")
print(f"    Std:       {std_loss:.6f}")
print(f"    PPL:       {math.exp(mean_loss):.6f}")
print()
print(f"  NORMA STATI PREDETTI (deve essere ~ 1.0)")
print(f"    Media:     {mean_norm:.6f}")
print(f"    Std:       {std_norm:.6f}")
print()
print(f"  SOGLIE FIDELITY")
print(f"    Campioni con Fid > 0.90:  {pct_above_90:5.1f}%")
print(f"    Campioni con Fid > 0.80:  {pct_above_80:5.1f}%")
print(f"    Campioni con Fid > 0.50:  {pct_above_50:5.1f}%")
print()
print(f"  FIDELITY PER STEP TEMPORALE")
for t in range(config.SEQ_LEN):
    bar = "#" * int(fid_per_step_mean[t].item() * 40)
    print(f"    t={t:2d}:  {fid_per_step_mean[t].item():.4f} +/- {fid_per_step_std[t].item():.4f}  {bar}")

# --- Confronto Train vs Test (generalization gap) ---
actual_epochs = len(history["train_loss"])
print()
print(f"  {'-' * 50}")
print(f"  GENERALIZATION GAP")
print(f"  {'-' * 50}")
final_tr_loss = history["train_loss"][-1]
final_tr_fid = history["train_fidelity"][-1]
final_te_loss = history["test_loss"][-1]
final_te_fid = history["test_fidelity"][-1]
gap_loss = final_te_loss - final_tr_loss
gap_fid = final_tr_fid - final_te_fid

print(f"    {'':18s} {'Loss':>10s} {'Fidelity':>10s} {'PPL':>10s}")
print(f"    {'-' * 50}")
print(f"    {'Train (finale)':18s} {final_tr_loss:10.4f} {final_tr_fid:10.4f} {math.exp(final_tr_loss):10.4f}")
print(f"    {'Test  (finale)':18s} {final_te_loss:10.4f} {final_te_fid:10.4f} {math.exp(final_te_loss):10.4f}")
print(f"    {'Gap':18s} {gap_loss:+10.4f} {-gap_fid:+10.4f} {math.exp(final_te_loss)-math.exp(final_tr_loss):+10.4f}")
print()

if gap_fid < 0.02:
    verdict = "Eccellente generalizzazione (gap < 2%)"
elif gap_fid < 0.05:
    verdict = "Buona generalizzazione (gap < 5%)"
elif gap_fid < 0.15:
    verdict = "Gap moderato -- valutare regolarizzazione o piu' dati"
else:
    verdict = "Overfitting significativo -- il modello memorizza il training"
print(f"    Verdetto: {verdict}")

# --- Learning dynamics ---
print()
print(f"  {'-' * 50}")
print(f"  DINAMICA DI APPRENDIMENTO")
print(f"  {'-' * 50}")
first_fid = history["train_fidelity"][0]
best_test_fid = max(history["test_fidelity"])
best_test_epoch = history["test_fidelity"].index(best_test_fid) + 1
improvement = history["train_fidelity"][-1] - first_fid

print(f"    Fidelity iniziale (epoca 1):       {first_fid:.4f}")
print(f"    Fidelity finale (epoca {actual_epochs}):      {history['train_fidelity'][-1]:.4f}")
print(f"    Miglioramento totale:              {improvement:+.4f}")
print(f"    Miglior test fidelity:             {best_test_fid:.4f} (epoca {best_test_epoch})")
print(f"    Tempo totale training:             {total_train_time:.1f}s")
print(f"    Tempo medio per epoca:             {total_train_time / max(1, actual_epochs):.2f}s")

# Convergenza: quante epoche per raggiungere 80% del miglioramento finale
target_80 = first_fid + improvement * 0.8
epoch_80 = None
for i, f in enumerate(history["train_fidelity"]):
    if f >= target_80:
        epoch_80 = i + 1
        break
if epoch_80:
    print(f"    Epoche per 80% miglioramento:      {epoch_80}")


# ============================================================
#  5. GRAFICI
# ============================================================
print(f"\n[5/6] Generando grafici in '{RESULTS_DIR}/'...")

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
plt.savefig(os.path.join(RESULTS_DIR, "training_report.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- Fig 5: Distribuzione fidelity sul test set ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribuzione Fidelity sul Test Set", fontsize=13, fontweight="bold")

ax = axes[0]
fid_means = all_fidelities.mean(dim=-1).numpy()
ax.hist(fid_means, bins=40, alpha=0.7, color="steelblue", edgecolor="navy")
ax.axvline(mean_fid, color="red", linestyle="--", linewidth=2, label=f"Media: {mean_fid:.4f}")
ax.axvline(median_fid, color="orange", linestyle="--", linewidth=2, label=f"Mediana: {median_fid:.4f}")
ax.set_xlabel("Fidelity media per campione")
ax.set_ylabel("Conteggio")
ax.set_title("Istogramma Fidelity")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.boxplot([all_fidelities[:, t].numpy() for t in range(config.SEQ_LEN)],
           tick_labels=[str(t) for t in range(config.SEQ_LEN)])
ax.set_xlabel("Step temporale")
ax.set_ylabel("Fidelity")
ax.set_title("Boxplot Fidelity per step")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fidelity_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

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
plt.savefig(os.path.join(RESULTS_DIR, "lr_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"  Salvati:")
print(f"    - {RESULTS_DIR}/training_report.png")
print(f"    - {RESULTS_DIR}/fidelity_distribution.png")
print(f"    - {RESULTS_DIR}/lr_analysis.png")


# ============================================================
#  6. RIEPILOGO FINALE
# ============================================================
print(f"\n[6/6] RIEPILOGO FINALE")
print("=" * 70)
print(f"  Fidelity Test (media):    {mean_fid:.4f} +/- {std_fid:.4f}")
print(f"  Loss Test (media):        {mean_loss:.4f} +/- {std_loss:.4f}")
print(f"  PPL Test:                 {math.exp(mean_loss):.4f}")
print(f"  Generalization gap:       {gap_fid:+.4f}")
print(f"  Miglior test fidelity:    {best_test_fid:.4f} (epoca {best_test_epoch})")
print(f"  Campioni Fid > 0.90:      {pct_above_90:.1f}%")
print(f"  Campioni Fid > 0.80:      {pct_above_80:.1f}%")
print(f"  Epoche effettive:         {actual_epochs}/{config.EPOCHS}")
if actual_epochs < config.EPOCHS:
    print(f"  Early stopping:           Attivato (risparmiato {config.EPOCHS - actual_epochs} epoche)")
if "lr" in history:
    print(f"  LR finale:                {history['lr'][-1]:.2e}")
print(f"  Tempo totale:             {total_train_time:.1f}s")
print(f"  Verdetto:                 {verdict}")
print("=" * 70)