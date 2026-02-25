import math

import torch
from torch.utils.data import Dataset, DataLoader

import config
from predictor import QuantumStatePredictor, QuantumFidelityLoss


class QuantumStateDataset(Dataset):
    """Dataset per coppie (input, target) di stati quantistici."""
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# --- 1. Preparazione Dati (placeholder con dati random) ---
dati_input = torch.randn(config.N_TOTALE, config.SEQ_LEN, config.D_MODEL)
dati_target = torch.randn(config.N_TOTALE, config.SEQ_LEN, config.D_MODEL, dtype=torch.complex64)  # I tuoi target reali

# Split Train / Test
n_train = int(config.N_TOTALE * config.TRAIN_SPLIT)
n_test = config.N_TOTALE - n_train

train_dataset = QuantumStateDataset(dati_input[:n_train], dati_target[:n_train])
test_dataset = QuantumStateDataset(dati_input[n_train:], dati_target[n_train:])

# DataLoader: pescano i campioni, li mischiano e li impacchettano in batch
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# --- 2. Inizializzazione Rete e Ottimizzatore ---
model = QuantumStatePredictor()
criterion = QuantumFidelityLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# --- 3. Il Loop di Addestramento ---
print(f"Dataset: {n_train} train / {n_test} test")
print("Inizio Addestramento...\n")

for epoch in range(config.EPOCHS):
    model.train()

    loss_accumulata = 0.0
    fidelity_accumulata = 0.0

    for batch_idx, (x_batch, target_batch) in enumerate(train_loader):

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
print("\n--- Valutazione sul Test Set ---\n")

model.eval()

test_loss_acc = 0.0
test_fidelity_acc = 0.0

with torch.no_grad():
    for x_batch, target_batch in test_loader:
        predicted_state = model(x_batch)
        loss, mean_fidelity = criterion(predicted_state, target_batch)

        test_loss_acc += loss.item()
        test_fidelity_acc += mean_fidelity.item()

avg_test_loss = test_loss_acc / len(test_loader)
avg_test_fidelity = test_fidelity_acc / len(test_loader)

print(f"Test Loss:       {avg_test_loss:.4f}")
print(f"Test Fidelity:   {avg_test_fidelity:.4f}")
print(f"Test Perplexity: {math.exp(avg_test_loss):.4f}")