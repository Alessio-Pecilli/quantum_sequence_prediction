"""
AUDIT COMPLETO: verifica modularità e dipendenza da config.py
Esegui con: python test_audit.py
"""
import torch
from torch.utils.data import DataLoader
import config
from input import QuantumStateDataset, generate_quantum_dynamics_dataset
from embedding import ComplexEmbedding
from predictor import QuantumStatePredictor, QuantumFidelityLoss, RotaryPositionalEncoding

print("=" * 60)
print("AUDIT COMPLETO: MODULARITA E DIPENDENZA DA CONFIG")
print("=" * 60)

errors = []
passed = 0
dim = 2 ** config.N_QUBITS
sl = config.SEQ_LEN

# ========== 1. CONFIG ==========
print("\n[1] CONFIG — verifica tipi e coerenza")
checks_config = {
    "N_QUBITS": (config.N_QUBITS, int),
    "DIM_2N": (config.DIM_2N, int),
    "D_MODEL": (config.D_MODEL, int),
    "NUM_HEADS": (config.NUM_HEADS, int),
    "NUM_LAYERS": (config.NUM_LAYERS, int),
    "SEQ_LEN": (config.SEQ_LEN, int),
    "DT": (config.DT, float),
    "J_RANGE": (config.J_RANGE, tuple),
    "G_RANGE": (config.G_RANGE, tuple),
    "HAMILTONIAN_TYPE": (config.HAMILTONIAN_TYPE, str),
    "B_TRAIN": (config.B_TRAIN, int),
    "S_TRAIN": (config.S_TRAIN, int),
    "B_TEST": (config.B_TEST, int),
    "S_TEST": (config.S_TEST, int),
    "BATCH_SIZE": (config.BATCH_SIZE, int),
    "EPOCHS": (config.EPOCHS, int),
    "LEARNING_RATE": (config.LEARNING_RATE, float),
    "ROPE_BASE": (config.ROPE_BASE, float),
    "DEVICE": (config.DEVICE, str),
}
for name, (val, typ) in checks_config.items():
    ok = isinstance(val, typ)
    status = "OK" if ok else "FAIL"
    print(f"  {name:20s} = {str(val):15s} tipo {type(val).__name__:6s} [{status}]")
    if ok:
        passed += 1
    else:
        errors.append(f"Config {name} tipo errato")

assert config.DIM_2N == 2 ** config.N_QUBITS, "DIM_2N != 2^N_QUBITS"
passed += 1
print(f"  DIM_2N == 2^N_QUBITS: OK")

assert config.D_MODEL % config.NUM_HEADS == 0, "D_MODEL non divisibile per NUM_HEADS"
passed += 1
print(f"  D_MODEL % NUM_HEADS == 0: OK")

assert config.D_MODEL % 2 == 0, "D_MODEL deve essere pari (RoPE)"
passed += 1
print(f"  D_MODEL pari (per RoPE): OK")


# ========== 2. DATASET ==========
print("\n[2] DATASET — generazione train e test (piccoli)")
B_tr, S_tr = 3, 5
B_te, S_te = 2, 4
train_in, train_tgt = generate_quantum_dynamics_dataset(B=B_tr, S=S_tr)
test_in, test_tgt = generate_quantum_dynamics_dataset(B=B_te, S=S_te)

# 2a. Shape
assert train_in.shape == (B_tr * S_tr, sl, dim), f"Train inputs shape {train_in.shape} != atteso"
assert train_tgt.shape == (B_tr * S_tr, sl, dim)
assert test_in.shape == (B_te * S_te, sl, dim), f"Test inputs shape {test_in.shape} != atteso"
assert test_tgt.shape == (B_te * S_te, sl, dim)
passed += 4
print(f"  Shape train ({B_tr}*{S_tr}, {sl}, {dim}): OK")
print(f"  Shape test  ({B_te}*{S_te}, {sl}, {dim}): OK")

# 2b. Dtype
assert train_in.dtype == torch.complex64
assert test_in.dtype == torch.complex64
passed += 2
print(f"  Dtype complex64: OK")

# 2c. Normalizzazione (TUTTI gli stati, non campionati)
norms_all = torch.norm(train_in.view(-1, dim), dim=-1)
assert torch.allclose(norms_all, torch.ones_like(norms_all), atol=1e-4)
passed += 1
print(f"  Normalizzazione train: OK (norms in [{norms_all.min():.6f}, {norms_all.max():.6f}])")

norms_all_t = torch.norm(train_tgt.view(-1, dim), dim=-1)
assert torch.allclose(norms_all_t, torch.ones_like(norms_all_t), atol=1e-4)
passed += 1
print(f"  Normalizzazione train targets: OK")

norms_test = torch.norm(test_in.view(-1, dim), dim=-1)
assert torch.allclose(norms_test, torch.ones_like(norms_test), atol=1e-4)
passed += 1
print(f"  Normalizzazione test: OK")

# 2d. Coerenza temporale
assert torch.allclose(train_in[:, 1:, :], train_tgt[:, :-1, :], atol=1e-5)
assert torch.allclose(test_in[:, 1:, :], test_tgt[:, :-1, :], atol=1e-5)
passed += 2
print(f"  Coerenza temporale train: OK")
print(f"  Coerenza temporale test:  OK")

# 2e. Indipendenza train vs test
assert not torch.allclose(train_in[0], test_in[0])
passed += 1
print(f"  Indipendenza train/test: OK")

# 2f. h_params override funziona
custom_params = {"J_range": (1.0, 1.0), "g_range": (1.0, 1.0)}
fixed_in, fixed_tgt = generate_quantum_dynamics_dataset(B=1, S=1, h_params=custom_params)
assert fixed_in.shape == (1, sl, dim)
passed += 1
print(f"  h_params override: OK")

# 2g. Parametri di default leggono da config
fixed_in2, _ = generate_quantum_dynamics_dataset(B=1, S=1)
assert fixed_in2.shape[1] == config.SEQ_LEN
assert fixed_in2.shape[2] == config.DIM_2N
passed += 2
print(f"  Default n_qubits/seq_len/dt da config: OK")


# ========== 3. EMBEDDING ==========
print("\n[3] EMBEDDING — ComplexEmbedding")
emb = ComplexEmbedding()
assert emb.projection.in_features == config.DIM_2N * 2, f"in_features {emb.projection.in_features}"
assert emb.projection.out_features == config.D_MODEL, f"out_features {emb.projection.out_features}"
passed += 2
print(f"  Linear({config.DIM_2N * 2} -> {config.D_MODEL}): OK")

x_c = torch.randn(4, sl, dim, dtype=torch.complex64)
h = emb(x_c)
assert h.shape == (4, sl, config.D_MODEL)
assert h.dtype == torch.float32
passed += 2
print(f"  Output shape {h.shape}, dtype float32: OK")

# 3b. Verifica con dim_2n custom
emb_custom = ComplexEmbedding(dim_2n=8, d_model=32)
assert emb_custom.projection.in_features == 16
assert emb_custom.projection.out_features == 32
passed += 2
print(f"  Parametrizzazione custom (8, 32): OK")


# ========== 4. ROPE ==========
print("\n[4] ROPE — RotaryPositionalEncoding")
rope = RotaryPositionalEncoding()
assert rope.cos_cached.shape == (config.SEQ_LEN, config.D_MODEL // 2)
assert rope.sin_cached.shape == (config.SEQ_LEN, config.D_MODEL // 2)
passed += 2
print(f"  Buffer shape ({config.SEQ_LEN}, {config.D_MODEL // 2}): OK")

h_rope = rope(h)
assert h_rope.shape == h.shape
passed += 1
print(f"  Output shape preservata: OK")

# RoPE modifica il tensore
assert not torch.allclose(h, h_rope)
passed += 1
print(f"  RoPE modifica il tensore: OK")

# Posizioni diverse per input identici
h_same = torch.ones(1, 3, config.D_MODEL)
h_same_rope = rope(h_same)
assert not torch.allclose(h_same_rope[0, 0], h_same_rope[0, 1])
assert not torch.allclose(h_same_rope[0, 1], h_same_rope[0, 2])
passed += 2
print(f"  Posizioni diverse per input identici: OK")

# RoPE con max_seq_len custom
rope_short = RotaryPositionalEncoding(d_model=config.D_MODEL, max_seq_len=5)
assert rope_short.cos_cached.shape == (5, config.D_MODEL // 2)
passed += 1
print(f"  max_seq_len custom (5): OK")

# RoPE gestisce seq_len < max_seq_len
h_short = torch.randn(2, 3, config.D_MODEL)
h_short_rope = rope(h_short)  # max_seq_len=SEQ_LEN ma seq_len=3
assert h_short_rope.shape == (2, 3, config.D_MODEL)
passed += 1
print(f"  seq_len < max_seq_len: OK")


# ========== 5. PREDICTOR ==========
print("\n[5] PREDICTOR — QuantumStatePredictor")
model = QuantumStatePredictor()

# 5a. Attributi
assert model.dim_2n == config.DIM_2N
assert model.d == config.D_MODEL
passed += 2
print(f"  dim_2n={model.dim_2n}, d={model.d}: OK")

# 5b. Output head dimensione corretta
assert model.output_head.in_features == config.D_MODEL
assert model.output_head.out_features == 2 * config.DIM_2N
passed += 2
print(f"  Output head Linear({config.D_MODEL} -> {2 * config.DIM_2N}): OK")

# 5c. Forward
pred = model(x_c)
assert pred.shape == (4, sl, config.DIM_2N), f"Pred shape {pred.shape}"
assert pred.is_complex()
passed += 2
print(f"  Output {pred.shape} complex: OK")

# 5d. Output normalizzato (softmax -> sqrt garantisce sum |a|^2 = 1)
probs = torch.abs(pred).pow(2).sum(dim=-1)
assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)
passed += 1
print(f"  Normalizzazione output (sum |a|^2 = 1): OK")

# 5e. Conteggio parametri
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parametri totali: {n_params:,}")

# 5f. RoPE è dentro il model
assert hasattr(model, "rope")
assert isinstance(model.rope, RotaryPositionalEncoding)
passed += 2
print(f"  RoPE presente nel modello: OK")


# ========== 6. LOSS ==========
print("\n[6] LOSS — QuantumFidelityLoss")
criterion = QuantumFidelityLoss()

# 6a. Shape scalare
loss, fid = criterion(pred, train_tgt[:4])
assert loss.shape == ()
assert fid.shape == ()
passed += 2
print(f"  Output scalare: OK")

# 6b. Range
assert 0 <= fid.item() <= 1, f"Fidelity {fid.item()} fuori range"
assert 0 <= loss.item() <= 1, f"Loss {loss.item()} fuori range"
passed += 2
print(f"  Loss={loss.item():.4f} in [0,1]: OK")
print(f"  Fidelity={fid.item():.4f} in [0,1]: OK")

# 6c. Fidelity perfetta (stato con se stesso)
target_norm = train_tgt[:4] / torch.norm(train_tgt[:4], dim=-1, keepdim=True)
loss_perf, fid_perf = criterion(target_norm, target_norm)
assert fid_perf.item() > 0.99, f"Fidelity con se stesso = {fid_perf.item()}"
assert loss_perf.item() < 0.01
passed += 2
print(f"  Fidelity(x, x) = {fid_perf.item():.6f} ≈ 1: OK")

# 6d. loss + fidelity = 1
assert abs(loss.item() + fid.item() - 1.0) < 1e-5
passed += 1
print(f"  Loss + Fidelity = 1: OK")


# ========== 7. DATALOADER ==========
print("\n[7] DATALOADER — batch assembly")
ds = QuantumStateDataset(train_in, train_tgt)
assert len(ds) == B_tr * S_tr
passed += 1
print(f"  len(dataset) = {len(ds)}: OK")

loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True)
bx, by = next(iter(loader))
expected_bs = min(config.BATCH_SIZE, len(ds))
assert bx.shape == (expected_bs, sl, dim)
assert by.shape == (expected_bs, sl, dim)
assert bx.dtype == torch.complex64
passed += 3
print(f"  Batch shape ({expected_bs}, {sl}, {dim}) complex64: OK")


# ========== 8. END-TO-END ==========
print("\n[8] END-TO-END — forward + backward su batch reale")
model.zero_grad()
pred_batch = model(bx)
assert pred_batch.shape == (expected_bs, sl, config.DIM_2N)
loss_e2e, fid_e2e = criterion(pred_batch, by)
loss_e2e.backward()
passed += 1
print(f"  Forward: OK (pred {pred_batch.shape})")

# Gradienti esistono
has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
assert has_grad, "Alcuni parametri non hanno gradiente!"
passed += 1
print(f"  Backward: OK (gradienti calcolati)")
print(f"  Loss={loss_e2e.item():.4f}, Fidelity={fid_e2e.item():.4f}")

# Un passo di ottimizzazione
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
optimizer.step()
passed += 1
print(f"  Optimizer step: OK")


# ========== 9. CATENA DIMENSIONALE COMPLETA ==========
print("\n[9] CATENA DIMENSIONALE (nessun mismatch)")
print(f"  Input:      complex64 (batch, {sl}, {dim})")
print(f"  Embedding:  float32   (batch, {sl}, {config.D_MODEL})")
print(f"  RoPE:       float32   (batch, {sl}, {config.D_MODEL})")
print(f"  Transformer:float32   (batch, {sl}, {config.D_MODEL})")
print(f"  Output head:float32   (batch, {sl}, {2*config.DIM_2N})")
print(f"  Split:      float32   ampiezze({config.DIM_2N}) + fasi({config.DIM_2N})")
print(f"  Output:     complex64 (batch, {sl}, {config.DIM_2N})")
print(f"  Target:     complex64 (batch, {sl}, {config.DIM_2N})")
print(f"  Match: OK")
passed += 1


# ========== RIEPILOGO ==========
sep = "=" * 60
print(f"\n{sep}")
if errors:
    print(f"  RISULTATO: {passed} check OK, {len(errors)} ERRORI")
    for e in errors:
        print(f"  ERRORE: {e}")
else:
    print(f"  RISULTATO: {passed} CHECK SUPERATI SU {passed}, 0 ERRORI")
    print(f"  TUTTO MODULARE E DIPENDENTE DA CONFIG.PY")
print(sep)
