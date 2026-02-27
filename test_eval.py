"""
EVAL DI TEST — mini-training + valutazione su test set indipendente.
Esegui con: python test_eval.py
Usa parametri piccoli per velocità (~30s su CPU).
"""
import math
import time

import torch
from torch.utils.data import DataLoader

import config
from input import QuantumStateDataset, generate_quantum_dynamics_dataset
from predictor import QuantumStatePredictor, QuantumFidelityLoss


# ===== Parametri eval (piccoli per velocità) =====
B_TRAIN, S_TRAIN = 10, 20    # 200 traiettorie train
B_TEST, S_TEST   = 5, 20     # 100 traiettorie test
EPOCHS_EVAL      = 15
BATCH_SIZE       = 32

print("=" * 60)
print("  EVAL DI TEST — mini-training + valutazione")
print("=" * 60)
print(f"  Hamiltoniana: {config.HAMILTONIAN_TYPE}")
print(f"  Qubit: {config.N_QUBITS}, dim Hilbert: {config.DIM_2N}")
print(f"  Train: {B_TRAIN} H × {S_TRAIN} stati = {B_TRAIN * S_TRAIN} campioni")
print(f"  Test:  {B_TEST} H × {S_TEST} stati = {B_TEST * S_TEST} campioni")
print(f"  Epoche: {EPOCHS_EVAL}, batch size: {BATCH_SIZE}")
print(f"  Modello: d={config.D_MODEL}, heads={config.NUM_HEADS}, layers={config.NUM_LAYERS}")
print(f"  Device: {config.DEVICE}")
print("=" * 60)


# --- 1. Generazione dataset ---
print("\n[1] Generando dataset...")
t0 = time.time()
train_in, train_tgt = generate_quantum_dynamics_dataset(B=B_TRAIN, S=S_TRAIN)
test_in, test_tgt = generate_quantum_dynamics_dataset(B=B_TEST, S=S_TEST)
print(f"    Generazione: {time.time() - t0:.2f}s")
print(f"    Train: {train_in.shape}  Test: {test_in.shape}")

train_in = train_in.to(config.DEVICE)
train_tgt = train_tgt.to(config.DEVICE)
test_in = test_in.to(config.DEVICE)
test_tgt = test_tgt.to(config.DEVICE)

train_loader = DataLoader(
    QuantumStateDataset(train_in, train_tgt), batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    QuantumStateDataset(test_in, test_tgt), batch_size=BATCH_SIZE, shuffle=False
)


# --- 2. Modello ---
print("\n[2] Inizializzando modello...")
model = QuantumStatePredictor().to(config.DEVICE)
criterion = QuantumFidelityLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
n_params = sum(p.numel() for p in model.parameters())
print(f"    Parametri: {n_params:,}")


# --- 3. Training ---
print(f"\n[3] Training ({EPOCHS_EVAL} epoche)...\n")
t0 = time.time()
train_history = []

for epoch in range(EPOCHS_EVAL):
    model.train()
    loss_acc, fid_acc = 0.0, 0.0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(x_batch)
        loss, fid = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        loss_acc += loss.item()
        fid_acc += fid.item()

    avg_loss = loss_acc / len(train_loader)
    avg_fid = fid_acc / len(train_loader)
    train_history.append((avg_loss, avg_fid))
    print(f"    Epoca {epoch+1:2d}/{EPOCHS_EVAL}  |  Loss: {avg_loss:.4f}  |  Fidelity: {avg_fid:.4f}")

train_time = time.time() - t0
print(f"\n    Training completato in {train_time:.1f}s")


# --- 4. Valutazione su TEST SET ---
print(f"\n[4] Valutazione su TEST SET (Hamiltoniane mai viste)...\n")
model.eval()

test_loss_acc, test_fid_acc = 0.0, 0.0
n_batches = 0

# Raccogliamo anche le fidelity per posizione temporale
per_step_fid = torch.zeros(config.SEQ_LEN)
per_step_count = 0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        pred = model(x_batch)
        loss, fid = criterion(pred, y_batch)
        test_loss_acc += loss.item()
        test_fid_acc += fid.item()
        n_batches += 1

        # Fidelity per step temporale
        overlap = torch.sum(y_batch.conj() * pred, dim=-1)  # (batch, seq_len)
        step_fid = torch.abs(overlap) ** 2  # (batch, seq_len)
        per_step_fid += step_fid.mean(dim=0).cpu()
        per_step_count += 1

avg_test_loss = test_loss_acc / n_batches
avg_test_fid = test_fid_acc / n_batches
per_step_fid /= per_step_count

print(f"    Test Loss:     {avg_test_loss:.4f}")
print(f"    Test Fidelity: {avg_test_fid:.4f}")

print(f"\n    Fidelity per step temporale:")
for t in range(config.SEQ_LEN):
    bar = "█" * int(per_step_fid[t].item() * 50)
    print(f"      t={t:2d}:  {per_step_fid[t].item():.4f}  {bar}")


# --- 5. Confronto train vs test ---
print(f"\n[5] Confronto Train vs Test\n")
final_train_loss, final_train_fid = train_history[-1]
gap_fid = final_train_fid - avg_test_fid
gap_loss = avg_test_loss - final_train_loss

print(f"    {'':15s} {'Loss':>10s} {'Fidelity':>10s}")
print(f"    {'─' * 37}")
print(f"    {'Train (finale)':15s} {final_train_loss:10.4f} {final_train_fid:10.4f}")
print(f"    {'Test':15s} {avg_test_loss:10.4f} {avg_test_fid:10.4f}")
print(f"    {'Gap':15s} {gap_loss:+10.4f} {-gap_fid:+10.4f}")
print()

if gap_fid < 0.05:
    print("    → Buona generalizzazione (gap fidelity < 5%)")
elif gap_fid < 0.15:
    print("    → Gap moderato — potrebbe servire più dati o regolarizzazione")
else:
    print("    → Overfitting significativo — il modello memorizza il training")


# --- 6. Evoluzione fidelity durante il training ---
print(f"\n[6] Evoluzione Fidelity durante il Training\n")
first_fid = train_history[0][1]
last_fid = train_history[-1][1]
improvement = last_fid - first_fid
print(f"    Epoca 1:  {first_fid:.4f}")
print(f"    Epoca {EPOCHS_EVAL}: {last_fid:.4f}")
print(f"    Miglioramento: {improvement:+.4f}")

if improvement > 0.01:
    print("    → Il modello sta imparando")
elif improvement > 0:
    print("    → Miglioramento minimo — potrebbe servire più training o un modello più grande")
else:
    print("    → Nessun miglioramento — controllare learning rate o architettura")


print(f"\n{'=' * 60}")
print("  EVAL COMPLETATA")
print(f"{'=' * 60}")
