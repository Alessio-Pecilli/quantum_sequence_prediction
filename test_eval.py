"""
TEST EVAL - mini-training + valutazione su test set indipendente.

Esegui con: python test_eval.py
Usa parametri piccoli per velocita' (~30s su CPU).

Include due modalita' di valutazione:
  - Teacher forcing (come durante il training): input[0..SEQ_LEN-1] -> target[1..SEQ_LEN]
  - Rollout autoregressivo: osserva T1 stati (context) e predice T2 step futuri
    usando una finestra fissa di lunghezza T1.
"""

import time
import gc

import torch
from torch.utils.data import DataLoader

import config
from input import QuantumStateDataset, QuantumTrajectoryWindowDataset, generate_quantum_dynamics_dataset
from predictor import QuantumStatePredictor, QuantumFidelityLoss


# ===== Parametri eval (piccoli per velocita') =====
B_TRAIN, S_TRAIN = 10, 20  # 200 traiettorie train
B_TEST, S_TEST = 5, 20  # 100 traiettorie test
EPOCHS_EVAL = 15
BATCH_SIZE = 32

print("=" * 60)
print("  TEST EVAL - mini-training + valutazione")
print("=" * 60)
print(f"  Hamiltoniana: {config.HAMILTONIAN_TYPE}")
print(f"  Qubit: {config.N_QUBITS}, dim Hilbert: {config.DIM_2N}")
print(f"  Train: {B_TRAIN} H x {S_TRAIN} stati = {B_TRAIN * S_TRAIN} campioni")
print(f"  Test:  {B_TEST} H x {S_TEST} stati = {B_TEST * S_TEST} campioni")
print(f"  Epoche: {EPOCHS_EVAL}, batch size: {BATCH_SIZE}")
print(
    f"  Rollout window: T1={config.T1} (context), T2={config.T2} (horizon)  "
    f"(richiede SEQ_LEN+1>={config.T1 + config.T2})"
)
print(f"  Training mode: {config.TRAINING_MODE}")
print(f"  Sequenza dataset (SEQ_LEN): {config.SEQ_LEN}  (dt={config.DT})")
print(f"  Modello: d={config.D_MODEL}, heads={config.NUM_HEADS}, layers={config.NUM_LAYERS}")
print(f"  Device: {config.DEVICE}")
print("=" * 60)


def iter_micro_batches(x_batch, y_batch, micro_bs):
    batch_size = x_batch.shape[0]
    micro_bs = max(1, min(micro_bs, batch_size))
    for start in range(0, batch_size, micro_bs):
        end = min(start + micro_bs, batch_size)
        x_micro = x_batch[start:end]
        y_micro = y_batch[start:end]
        weight = x_micro.shape[0] / batch_size
        yield x_micro, y_micro, weight


# --- 1. Generazione dataset ---
print("\n[1] Generando dataset...")
t0 = time.time()
train_in, train_tgt = generate_quantum_dynamics_dataset(B=B_TRAIN, S=S_TRAIN)
test_in, test_tgt = generate_quantum_dynamics_dataset(B=B_TEST, S=S_TEST)
print(f"    Generazione: {time.time() - t0:.2f}s")
print(f"    Train: {tuple(train_in.shape)}  Test: {tuple(test_in.shape)}")

pin_memory = bool(config.PIN_MEMORY and config.DEVICE == "cuda")
num_workers = 0 if config.MEMORY_SAFE_MODE else config.NUM_WORKERS
loader_kwargs = dict(batch_size=BATCH_SIZE, pin_memory=pin_memory)
if num_workers > 0:
    loader_kwargs["num_workers"] = num_workers
    loader_kwargs["persistent_workers"] = True
else:
    loader_kwargs["num_workers"] = 0

train_full_dataset = QuantumStateDataset(train_in, train_tgt)
test_full_dataset = QuantumStateDataset(test_in, test_tgt)

if config.TRAINING_MODE == "rollout_window":
    train_dataset = QuantumTrajectoryWindowDataset(
        train_in,
        train_tgt,
        context_len=config.T1,
        rollout_horizon=config.T2,
        unroll_steps=config.UNROLL_STEPS,
    )
    test_dataset = QuantumTrajectoryWindowDataset(
        test_in,
        test_tgt,
        context_len=config.T1,
        rollout_horizon=config.T2,
        unroll_steps=config.UNROLL_STEPS,
    )
else:
    train_dataset = train_full_dataset
    test_dataset = test_full_dataset

train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
test_eval_loader = DataLoader(test_full_dataset, shuffle=False, **loader_kwargs)

micro_batch_size = config.MICRO_BATCH_SIZE if config.MICRO_BATCH_SIZE > 0 else BATCH_SIZE
non_blocking = pin_memory


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

    for step, (x_batch_cpu, y_batch_cpu) in enumerate(train_loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        batch_loss, batch_fid = 0.0, 0.0

        for x_micro_cpu, y_micro_cpu, weight in iter_micro_batches(
            x_batch_cpu, y_batch_cpu, micro_batch_size
        ):
            x_micro = x_micro_cpu.to(config.DEVICE, non_blocking=non_blocking)
            y_micro = y_micro_cpu.to(config.DEVICE, non_blocking=non_blocking)

            pred = model(x_micro)
            if y_micro.ndim == 3 and y_micro.shape[1] >= 1 and pred.shape[1] != y_micro.shape[1]:
                pred_for_loss = pred[:, -1, :]
                target_for_loss = y_micro[:, 0, :]
            else:
                pred_for_loss = pred
                target_for_loss = y_micro
            loss, fid = criterion(pred_for_loss, target_for_loss)
            (loss * weight).backward()

            batch_loss += loss.item() * weight
            batch_fid += fid.item() * weight

            del x_micro, y_micro, pred, pred_for_loss, target_for_loss, loss, fid

        optimizer.step()

        loss_acc += batch_loss
        fid_acc += batch_fid

        del x_batch_cpu, y_batch_cpu
        if config.GC_COLLECT_EVERY_N_STEPS > 0 and step % config.GC_COLLECT_EVERY_N_STEPS == 0:
            gc.collect()

    avg_loss = loss_acc / len(train_loader)
    avg_fid = fid_acc / len(train_loader)
    train_history.append((avg_loss, avg_fid))
    print(
        f"    Epoca {epoch+1:2d}/{EPOCHS_EVAL}  |  Loss: {avg_loss:.4f}  |  Fidelity: {avg_fid:.4f}"
    )

train_time = time.time() - t0
print(f"\n    Training completato in {train_time:.1f}s")


# --- 4. Valutazione (teacher forcing) ---
print("\n[4] Valutazione su TEST SET (teacher forcing)...\n")
model.eval()

test_loss_acc, test_fid_acc = 0.0, 0.0
n_batches = 0

per_step_fid = torch.zeros(config.SEQ_LEN, dtype=torch.float64)
per_step_count = 0

with torch.no_grad():
    for step, (x_batch_cpu, y_batch_cpu) in enumerate(test_eval_loader, start=1):
        batch_loss, batch_fid = 0.0, 0.0
        batch_step_fid = torch.zeros(config.SEQ_LEN, dtype=torch.float64)

        for x_micro_cpu, y_micro_cpu, weight in iter_micro_batches(
            x_batch_cpu, y_batch_cpu, micro_batch_size
        ):
            x_micro = x_micro_cpu.to(config.DEVICE, non_blocking=non_blocking)
            y_micro = y_micro_cpu.to(config.DEVICE, non_blocking=non_blocking)

            pred = model(x_micro)
            loss, fid = criterion(pred, y_micro)

            batch_loss += loss.item() * weight
            batch_fid += fid.item() * weight

            overlap = torch.sum(y_micro.conj() * pred, dim=-1)  # (micro_batch, seq_len)
            step_fid = (torch.abs(overlap) ** 2).double()  # (micro_batch, seq_len)
            batch_step_fid += step_fid.mean(dim=0).cpu() * weight

            del x_micro, y_micro, pred, loss, fid, overlap, step_fid

        test_loss_acc += batch_loss
        test_fid_acc += batch_fid
        n_batches += 1
        per_step_fid += batch_step_fid
        per_step_count += 1

        del x_batch_cpu, y_batch_cpu
        if config.GC_COLLECT_EVERY_N_STEPS > 0 and step % config.GC_COLLECT_EVERY_N_STEPS == 0:
            gc.collect()

avg_test_loss = test_loss_acc / n_batches
avg_test_fid = test_fid_acc / n_batches
per_step_fid = per_step_fid / max(1, per_step_count)

print(f"    Test Loss:     {avg_test_loss:.4f}")
print(f"    Test Fidelity: {avg_test_fid:.4f}")

print("\n    Fidelity per step temporale (mostro i primi 20 step):")
for t in range(min(20, config.SEQ_LEN)):
    print(f"      t={t:2d}: {per_step_fid[t].item():.4f}")
if config.SEQ_LEN > 20:
    print(f"      ... (totale {config.SEQ_LEN} step)")


# --- 4b. Valutazione rollout autoregressiva (T1 -> T2) ---
print(f"\n[4b] Valutazione ROLLOUT autoregressiva (T1={config.T1} -> T2={config.T2})...\n")
rollout_fid_sum = torch.zeros(config.T2, dtype=torch.float64)
rollout_count = 0

with torch.no_grad():
    for step, (x_batch_cpu, y_batch_cpu) in enumerate(test_eval_loader, start=1):
        # Ricostruiamo la traiettoria completa: states[t] per t=0..SEQ_LEN
        # inputs:  t=0..SEQ_LEN-1, targets: t=1..SEQ_LEN
        state0 = x_batch_cpu[:, 0:1, :]  # (batch, 1, dim)
        true_states = torch.cat([state0, y_batch_cpu], dim=1)  # (batch, SEQ_LEN+1, dim)

        # Context iniziale (ground-truth) + rollout con finestra fissa di lunghezza T1
        rollout_states = true_states[:, : config.T1, :].to(config.DEVICE, non_blocking=non_blocking)
        preds = []
        for _ in range(config.T2):
            window = rollout_states[:, -config.T1 :, :]  # (batch, T1, dim)
            pred_seq = model(window)  # (batch, T1, dim), pred_seq[:, t] ~ state(t+1)
            next_state = pred_seq[:, -1:, :]  # (batch, 1, dim) ~ state(next)
            preds.append(next_state)
            rollout_states = torch.cat([rollout_states, next_state], dim=1)

        pred_future = torch.cat(preds, dim=1)  # (batch, T2, dim) device
        true_future = true_states[:, config.T1 : config.T1 + config.T2, :].to(
            config.DEVICE, non_blocking=non_blocking
        )

        overlap = torch.sum(true_future.conj() * pred_future, dim=-1)  # (batch, T2)
        fid = (torch.abs(overlap) ** 2).double()  # (batch, T2)

        rollout_fid_sum += fid.sum(dim=0, dtype=torch.float64).cpu()
        rollout_count += fid.shape[0]

        del x_batch_cpu, y_batch_cpu, state0, true_states, rollout_states, preds, pred_future, true_future, overlap, fid
        if config.GC_COLLECT_EVERY_N_STEPS > 0 and step % config.GC_COLLECT_EVERY_N_STEPS == 0:
            gc.collect()

rollout_fid_mean = (rollout_fid_sum / max(1, rollout_count)).float()
print(f"    Rollout mean fidelity (media su batch, per step): {rollout_fid_mean.mean().item():.4f}")
print("\n    Rollout fidelity per step (mostro i primi 50 step):")
max_print = min(config.T2, 50)
for k in range(max_print):
    print(f"      k={k+1:3d}: {rollout_fid_mean[k].item():.4f}")
if config.T2 > max_print:
    print(f"      ... (totale T2={config.T2} step)")


# --- 5. Confronto train vs test ---
print("\n[5] Confronto Train vs Test\n")
final_train_loss, final_train_fid = train_history[-1]
gap_fid = final_train_fid - avg_test_fid
gap_loss = avg_test_loss - final_train_loss

print(f"    {'':15s} {'Loss':>10s} {'Fidelity':>10s}")
print(f"    {'-' * 37}")
print(f"    {'Train (finale)':15s} {final_train_loss:10.4f} {final_train_fid:10.4f}")
print(f"    {'Test':15s} {avg_test_loss:10.4f} {avg_test_fid:10.4f}")
print(f"    {'Gap':15s} {gap_loss:+10.4f} {-gap_fid:+10.4f}")
print()

if gap_fid < 0.05:
    print("    -> Buona generalizzazione (gap fidelity < 5%)")
elif gap_fid < 0.15:
    print("    -> Gap moderato - potrebbe servire piu' dati o regolarizzazione")
else:
    print("    -> Overfitting significativo - il modello memorizza il training")


# --- 6. Evoluzione fidelity durante il training ---
print("\n[6] Evoluzione Fidelity durante il Training\n")
first_fid = train_history[0][1]
last_fid = train_history[-1][1]
improvement = last_fid - first_fid
print(f"    Epoca 1:  {first_fid:.4f}")
print(f"    Epoca {EPOCHS_EVAL}: {last_fid:.4f}")
print(f"    Miglioramento: {improvement:+.4f}")

if improvement > 0.01:
    print("    -> Il modello sta imparando")
elif improvement > 0:
    print("    -> Miglioramento minimo - potrebbe servire piu' training o un modello piu' grande")
else:
    print("    -> Nessun miglioramento - controllare learning rate o architettura")

print(f"\n{'=' * 60}")
print("  EVAL COMPLETATA")
print(f"{'=' * 60}")
