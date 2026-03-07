import torch
from torch.utils.data import Dataset, DataLoader

import config


class QuantumStateDataset(Dataset):
    def __init__(self, inputs_data, targets_data):
        """
        Qui passi i tuoi dati reali.
        inputs_data: Tensore di shape (N_samples, seq_len, d) - Reale
        targets_data: Tensore di shape (N_samples, seq_len, d) - Complesso
        """
        self.inputs = inputs_data
        self.targets = targets_data

        # Un check di sicurezza (best practice)
        assert len(self.inputs) == len(self.targets), "Mismatch tra input e target!"

    def __len__(self):
        # Ritorna il numero totale di campioni nel dataset
        return len(self.inputs)

    def __getitem__(self, idx):
        # Estrae un singolo campione. Il DataLoader chiamerÃ  questo metodo
        # per assemblare i batch in automatico.
        x = self.inputs[idx]
        y_target = self.targets[idx]

        return x, y_target


# --- MATRICI DI PAULI BASE ---
# Definite rigorosamente in campo complesso
I = torch.eye(2, dtype=torch.complex64)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)


def get_pauli_string(operator_list):
    """
    Costruisce l'operatore denso 2^n x 2^n tramite prodotto di Kronecker.
    operator_list: lista di tensori 2x2 (es. [I, X, I, I] per applicare X al secondo qubit su 4)
    """
    res = operator_list[0]
    for op in operator_list[1:]:
        res = torch.kron(res, op)
    return res


def build_tfim_hamiltonian(n_qubits, J, g):
    """
    MODULO ISOLATO: Costruisce l'Hamiltoniana TFIM per n_qubits.
    In futuro, se cambi sistema fisico, modifichi solo il corpo di questa funzione.
    Ritorna: Tensore (2^n, 2^n) complesso.
    """
    dim = 2**n_qubits
    H = torch.zeros((dim, dim), dtype=torch.complex64)

    # Termine di interazione (ZZ) sui vicini (condizioni al contorno aperte)
    for i in range(n_qubits - 1):
        ops = [I] * n_qubits
        ops[i] = Z
        ops[i + 1] = Z
        H -= J * get_pauli_string(ops)

    # Termine di campo trasverso (X) su ogni qubit
    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = X
        H -= g * get_pauli_string(ops)

    return H


def _build_hamiltonian(h_type, n_qubits, params):
    """
    Factory: costruisce l'Hamiltoniana del tipo richiesto con parametri casuali.
    h_type: stringa identificativa (es. "TFIM")
    n_qubits: numero di qubit
    params: dizionario con i range dei parametri (definiti in config)
    Ritorna: (H, descrizione_parametri)
    """
    if h_type == "TFIM":
        J = torch.empty(1).uniform_(*params["J_range"]).item()
        g = torch.empty(1).uniform_(*params["g_range"]).item()
        H = build_tfim_hamiltonian(n_qubits, J=J, g=g)
        return H
    raise ValueError(f"Tipo di Hamiltoniana '{h_type}' non supportato. Opzioni: TFIM")


def generate_quantum_dynamics_dataset(
    B,
    S,
    n_qubits=config.N_QUBITS,
    h_type=config.HAMILTONIAN_TYPE,
    seq_len=config.SEQ_LEN,
    dt=config.DT,
    h_params=None,
):
    """
    Genera un dataset di B * S traiettorie quantistiche.

    Protocollo:
      - Il tipo di Hamiltoniana Ã¨ fissato (h_type).
      - Si estraggono B set di parametri casuali â†’ B Hamiltoniane diverse dello stesso tipo.
      - Per ogni b = 1..B si estraggono S stati iniziali casuali e si evolvono con H_b.
      - Totale campioni: B * S.

    Questa funzione va chiamata separatamente per training e test,
    cosÃ¬ da ottenere Hamiltoniane e stati iniziali completamente indipendenti.

    Args:
        B: Numero di Hamiltoniane casuali da campionare
        S: Numero di stati iniziali casuali per ogni Hamiltoniana
        n_qubits: Numero di qubit del sistema
        h_type: Tipo di Hamiltoniana ("TFIM", ...)
        seq_len: Lunghezza della sequenza (quanti step di evoluzione)
        dt: Passo temporale dell'evoluzione
    """
    dim = 2**n_qubits
    total_samples = B * S

    # Parametri specifici per il tipo di Hamiltoniana (default da config)
    if h_params is None:
        h_params = {
            "J_range": config.J_RANGE,
            "g_range": config.G_RANGE,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generazione su GPU (se disponibile); si torna su CPU solo al termine.
    inputs_data = torch.empty((total_samples, seq_len, dim), dtype=torch.complex64, device=device)
    targets_data = torch.empty((total_samples, seq_len, dim), dtype=torch.complex64, device=device)

    times = torch.arange(seq_len + 1, device=device, dtype=torch.float32).unsqueeze(1)  # (seq_len+1, 1)

    sample_idx = 0
    for _ in range(B):
        # 1. Costruzione Hamiltoniana b-esima (parametri estratti casualmente)
        H_b = _build_hamiltonian(h_type, n_qubits, h_params).to(device)

        # 2. Diagonalizzazione: H = V diag(E) Vâ€   (eigh perchÃ© H Ã¨ Hermitiana)
        #    Costo O(dimÂ³) una tantum, poi evoluzione diventa banale.
        E_b, V_b = torch.linalg.eigh(H_b)

        # 3. Fasi di evoluzione nella base diagonale per tutti i time-step
        #    phase(t) = exp(-i * E * dt * t),  shape (seq_len+1, dim)
        all_phases = torch.exp(-1j * E_b.unsqueeze(0) * dt * times)  # (seq_len+1, dim)

        # 4. Estrazione stati iniziali casuali (normalizzati) su device
        psi_real = torch.randn((S, dim), device=device)
        psi_imag = torch.randn((S, dim), device=device)
        psi = torch.complex(psi_real, psi_imag)
        psi = psi / torch.linalg.vector_norm(psi, dim=-1, keepdim=True).clamp(min=1e-8)

        # 5. Proiezione nella base degli autostati: c = Vâ€  |Ïˆ>
        #    (row form) c^T = (Vâ€  |Ïˆ>)^T = <Ïˆ| V*  ->  psi @ V.conj()
        coeffs = psi @ V_b.conj()  # (S, dim)

        # 6. Evoluzione vettorizzata nella base diagonale
        all_coeffs = coeffs.unsqueeze(1) * all_phases.unsqueeze(0)  # (S, seq_len+1, dim)

        # 7. Ritorno alla base computazionale: |Ïˆ(t)> = V c(t)
        trajectory_tensor = all_coeffs @ V_b.T  # (S, seq_len+1, dim)

        # 8. Split tramite shift temporale:
        # Input: da t=0 a t=seq_len-1
        # Target: da t=1 a t=seq_len
        inputs_data[sample_idx : sample_idx + S] = trajectory_tensor[:, :-1]
        targets_data[sample_idx : sample_idx + S] = trajectory_tensor[:, 1:]
        sample_idx += S

    return inputs_data.cpu(), targets_data.cpu()


# ============================================================
# TEST STANDALONE â€” esegui con: python input.py
# ============================================================
if __name__ == "__main__":
    import time

    def check_dataset(name, inputs, targets, B, S, seq_len, n_qubits):
        """Esegue tutti i controlli diagnostici su un dataset generato."""
        dim = 2**n_qubits
        total = B * S

        print(f"\n{'â”€' * 50}")
        print(f"  DATASET: {name}")
        print(f"{'â”€' * 50}")

        # --- Shape ---
        print(f"\n  [Shape]")
        print(f"    inputs:  {inputs.shape}  (atteso: ({total}, {seq_len}, {dim}))")
        print(f"    targets: {targets.shape}  (atteso: ({total}, {seq_len}, {dim}))")
        assert inputs.shape == (total, seq_len, dim), "Shape inputs errata!"
        assert targets.shape == (total, seq_len, dim), "Shape targets errata!"
        print("    âœ“ OK")

        # --- Dtype ---
        print("\n  [Dtype]")
        print(f"    inputs dtype:  {inputs.dtype}")
        print(f"    targets dtype: {targets.dtype}")
        assert inputs.dtype == torch.complex64
        assert targets.dtype == torch.complex64
        print("    âœ“ OK (complex64)")

        # --- Normalizzazione: ogni stato deve avere norma â‰ˆ 1 ---
        print("\n  [Normalizzazione stati]")
        # Controlla un campione di stati (tutti i tempi di 10 traiettorie casuali)
        idxs = torch.randint(0, total, (min(10, total),))
        norms_in = torch.norm(inputs[idxs], dim=-1)  # (10, seq_len)
        norms_tgt = torch.norm(targets[idxs], dim=-1)
        print(f"    Norma inputs  â€” min: {norms_in.min():.6f}, max: {norms_in.max():.6f}, media: {norms_in.mean():.6f}")
        print(f"    Norma targets â€” min: {norms_tgt.min():.6f}, max: {norms_tgt.max():.6f}, media: {norms_tgt.mean():.6f}")
        assert torch.allclose(norms_in, torch.ones_like(norms_in), atol=1e-4), "Norma inputs fuori tolleranza!"
        assert torch.allclose(norms_tgt, torch.ones_like(norms_tgt), atol=1e-4), "Norma targets fuori tolleranza!"
        print("    âœ“ Tutti â‰ˆ 1.0 (atol=1e-4)")

        # --- Coerenza input/target: target[t] == input[t+1] ---
        print("\n  [Coerenza temporale (target[t] == input[t+1])]")
        # Per ogni campione, targets[:, t, :] deve essere uguale a inputs[:, t+1, :]
        match = torch.allclose(inputs[idxs, 1:, :], targets[idxs, :-1, :], atol=1e-5)
        print(f"    Campione di {len(idxs)} traiettorie: {'âœ“ coerente' if match else 'âœ— INCOERENTE!'}")
        assert match, "Input e target non sono temporalmente coerenti!"

        # --- UnitarietÃ : |<psi(t)|psi(t)>| = 1, overlap tra step vicini ---
        print("\n  [Overlap tra step consecutivi (primo campione)]")
        sample = inputs[0]  # (seq_len, dim)
        for t in range(min(5, seq_len - 1)):
            overlap = torch.abs(torch.sum(sample[t].conj() * sample[t + 1])).item()
            print(f"    |<Ïˆ({t})|Ïˆ({t+1})>| = {overlap:.6f}")

        # --- Statistiche ampiezze e fasi ---
        print("\n  [Statistiche ampiezze (primo campione, t=0)]")
        s0 = inputs[0, 0]  # (dim,)
        amps = torch.abs(s0)
        phases = torch.angle(s0)
        print(f"    Ampiezze â€” min: {amps.min():.4f}, max: {amps.max():.4f}, media: {amps.mean():.4f}")
        print(f"    Fasi     â€” min: {phases.min():.4f}, max: {phases.max():.4f}, media: {phases.mean():.4f}")

        # --- NaN / Inf check ---
        print("\n  [NaN/Inf check]")
        has_nan = torch.isnan(torch.view_as_real(inputs)).any() or torch.isnan(torch.view_as_real(targets)).any()
        has_inf = torch.isinf(torch.view_as_real(inputs)).any() or torch.isinf(torch.view_as_real(targets)).any()
        print(f"    NaN: {'âœ— TROVATI' if has_nan else 'âœ“ nessuno'}")
        print(f"    Inf: {'âœ— TROVATI' if has_inf else 'âœ“ nessuno'}")
        assert not has_nan, "Trovati valori NaN!"
        assert not has_inf, "Trovati valori Inf!"

        print(f"\n  âœ“ TUTTI I CHECK SUPERATI per {name}\n")

    # ---- Parametri di test (piccoli per velocitÃ ) ----
    B_tr, S_tr = 5, 10  # 50 traiettorie train
    B_te, S_te = 3, 10  # 30 traiettorie test
    n_qubits = config.N_QUBITS
    seq_len = config.SEQ_LEN

    print("=" * 50)
    print("  TEST MODULO input.py")
    print(f"  Hamiltoniana: {config.HAMILTONIAN_TYPE}")
    print(f"  Qubit: {n_qubits}, dim Hilbert: {2**n_qubits}")
    print(f"  Train: B={B_tr}, S={S_tr} â†’ {B_tr*S_tr} campioni")
    print(f"  Test:  B={B_te}, S={S_te} â†’ {B_te*S_te} campioni")
    print(f"  Seq len: {seq_len}, dt: {config.DT}")
    print("=" * 50)

    # --- Generazione TRAIN ---
    t0 = time.time()
    train_in, train_tgt = generate_quantum_dynamics_dataset(B=B_tr, S=S_tr)
    t_train = time.time() - t0
    print(f"\n  Generazione TRAIN: {t_train:.2f}s")
    check_dataset("TRAIN", train_in, train_tgt, B_tr, S_tr, seq_len, n_qubits)

    # --- Generazione TEST ---
    t0 = time.time()
    test_in, test_tgt = generate_quantum_dynamics_dataset(B=B_te, S=S_te)
    t_test = time.time() - t0
    print(f"\n  Generazione TEST: {t_test:.2f}s")
    check_dataset("TEST", test_in, test_tgt, B_te, S_te, seq_len, n_qubits)

    # --- Verifica indipendenza train/test ---
    print(f"{'â”€' * 50}")
    print("  INDIPENDENZA TRAIN vs TEST")
    print(f"{'â”€' * 50}")
    # I primi campioni di train e test devono essere diversi (generati con H diverse)
    diff = torch.norm(train_in[0] - test_in[0]).item()
    print(f"  â€–train[0] - test[0]â€– = {diff:.6f}  (deve essere > 0)")
    assert diff > 1e-6, "Train e test sembrano identici!"
    print("  âœ“ Dataset indipendenti\n")

    # --- Test DataLoader ---
    print(f"{'â”€' * 50}")
    print("  TEST DataLoader")
    print(f"{'â”€' * 50}")
    ds = QuantumStateDataset(train_in, train_tgt)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"  Batch x shape: {batch_x.shape}  (atteso: (16, {seq_len}, {2**n_qubits}))")
    print(f"  Batch y shape: {batch_y.shape}")
    assert batch_x.shape == (16, seq_len, 2**n_qubits)
    print("  âœ“ DataLoader funzionante\n")

    print("=" * 50)
    print("  TUTTI I TEST SUPERATI âœ“")
    print("=" * 50)
