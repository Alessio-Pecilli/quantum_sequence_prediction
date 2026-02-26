import math

import torch
from torch.utils.data import DataLoader

import config
from input import QuantumStateDataset, generate_quantum_dynamics_dataset
from predictor import QuantumStatePredictor, QuantumFidelityLoss


print("=" * 60)
print("QUANTUM SEQUENCE PREDICTION - TRAINING PIPELINE")
print("=" * 60)
print(f"Device: {config.DEVICE}")
print(f"Sistema quantistico: {config.N_QUBITS} qubit (dim. Hilbert: {config.DIM_2N})")
print(f"Dataset: {config.N_HAMILTONIANS} Hamiltoniane × {config.N_STATES_PER_H} stati = {config.N_TOTALE} campioni")
print(f"Sequenza: {config.SEQ_LEN} passi temporali (dt={config.DT})")
print(f"Modello: d={config.D_MODEL}, heads={config.NUM_HEADS}, layers={config.NUM_LAYERS}")
print("=" * 60)

# --- 1. Preparazione Dati (genera dataset reale) ---
print("\n[1/5] Generando dataset quantistico...")
inputs_data, targets_data = generate_quantum_dynamics_dataset()
print(f"✓ Dataset generato: inputs {inputs_data.shape}, targets {targets_data.shape}")

# Spostare i dati sul device corretto
inputs_data = inputs_data.to(config.DEVICE)
targets_data = targets_data.to(config.DEVICE)

# Split Train / Test
n_train = int(config.N_TOTALE * config.TRAIN_SPLIT)
n_test = config.N_TOTALE - n_train

train_dataset = QuantumStateDataset(inputs_data[:n_train], targets_data[:n_train])
test_dataset = QuantumStateDataset(inputs_data[n_train:], targets_data[n_train:])

# DataLoader: pescano i campioni, li mischiano e li impacchettano in batch
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

print(f"✓ Split: {n_train} train / {n_test} test")

# --- 2. Inizializzazione Rete e Ottimizzatore ---
print("\n[2/5] Inizializzando modello...")
model = QuantumStatePredictor().to(config.DEVICE)
criterion = QuantumFidelityLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
print(f"✓ Modello spostato su {config.DEVICE}")
print(f"✓ Parametri totali: {sum(p.numel() for p in model.parameters()):,}")

# --- 3. Il Loop di Addestramento ---
print(f"\n[3/5] Inizio Addestramento...\n")

for epoch in range(config.EPOCHS):
    model.train()

    loss_accumulata = 0.0
    fidelity_accumulata = 0.0

    for batch_idx, (x_batch, target_batch) in enumerate(train_loader):
        x_batch = x_batch.to(config.DEVICE)
        target_batch = target_batch.to(config.DEVICE)

        optimizer.zero_grad()

        predicted_state = model(x_batch)
        loss, mean_fidelity = criterion(predicted_state, target_batch)

        loss.backward()
        optimizer.step()

        loss_accumulata += loss.item()
        fidelity_accumulata += mean_fidelity.item()

    avg_loss = loss_accumulata / len(train_loader)
    avg_fidelity = fidelity_accumulata / len(train_loader)

    print(f"Epoca {epoch+1}/{config.EPOCHS} | Loss: {avg_loss:.4f} | Fidelity: {avg_fidelity:.4f} | Perplexity: {math.exp(avg_loss):.4f}")

# --- 4. Valutazione sul Test Set ---
print("\n[4/5] Valutazione sul Test Set...\n")

model.eval()

test_loss_acc = 0.0
test_fidelity_acc = 0.0

with torch.no_grad():
    for x_batch, target_batch in test_loader:
        x_batch = x_batch.to(config.DEVICE)
        target_batch = target_batch.to(config.DEVICE)
        
        predicted_state = model(x_batch)
        loss, mean_fidelity = criterion(predicted_state, target_batch)

        test_loss_acc += loss.item()
        test_fidelity_acc += mean_fidelity.item()

avg_test_loss = test_loss_acc / len(test_loader)
avg_test_fidelity = test_fidelity_acc / len(test_loader)

print(f"Test Loss:       {avg_test_loss:.4f}")
print(f"Test Fidelity:   {avg_test_fidelity:.4f}")
print(f"Test Perplexity: {math.exp(avg_test_loss):.4f}")

print("\n[5/5] Training completato!")
print("=" * 60)