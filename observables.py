import torch


def precompute_observables(n_qubits: int, device: torch.device | str):
    """
    Pre-computa strutture utili per il calcolo di osservabili su stati in base computazionale.

    Convenzione ordine qubit coerente con input.py (Kronecker in ordine ops[0], ops[1], ...):
      - qubit 0 = bit piu' significativo (MSB)
      - qubit N-1 = bit meno significativo (LSB)

    Ritorna:
      z_eigs:       (n_qubits, 2^n) float32 con autovalori di Z_i
      zz_nn_eigs:   (n_qubits-1, 2^n) float32 con autovalori di Z_i Z_{i+1} (vicini, bordo aperto)
      x_flip_idx:   lista length n_qubits, ciascuno (2^n,) int64 con indici flippati per X_i
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits deve essere >= 1, ricevuto: {n_qubits}")

    device = torch.device(device)
    dim = 1 << n_qubits
    indices = torch.arange(dim, device=device, dtype=torch.int64)

    # qubit i -> shift = (n_qubits - 1 - i)
    shifts = torch.arange(n_qubits, device=device, dtype=torch.int64)
    shifts = (n_qubits - 1) - shifts  # (n_qubits,)

    bits = (indices.unsqueeze(0) >> shifts.unsqueeze(1)) & 1  # (n_qubits, dim)
    z_eigs = (1.0 - 2.0 * bits.float()).to(torch.float32)  # (n_qubits, dim) in {+1, -1}

    # Nearest-neighbor ZZ (open boundary): (0,1), (1,2), ..., (N-2,N-1)
    if n_qubits >= 2:
        zz_nn_eigs = z_eigs[:-1] * z_eigs[1:]  # (n_qubits-1, dim)
    else:
        zz_nn_eigs = torch.empty((0, dim), device=device, dtype=torch.float32)

    x_flip_idx = []
    for i in range(n_qubits):
        shift = int((n_qubits - 1) - i)
        flip_mask = 1 << shift
        x_flip_idx.append(indices ^ flip_mask)  # (dim,)

    return z_eigs, zz_nn_eigs, x_flip_idx


@torch.no_grad()
def batch_observables(
    states: torch.Tensor,
    z_eigs: torch.Tensor,
    zz_nn_eigs: torch.Tensor,
    x_flip_idx: list[torch.Tensor],
):
    """
    Calcola (per batch) le medie su siti/pairs:
      m^z  = (1/N) * sum_i <Z_i>
      m^x  = (1/N) * sum_i <X_i>
      c^z  = (1/(N-1)) * sum_i <Z_i Z_{i+1}>  (solo vicini, bordo aperto)

    Args:
      states: (batch, dim) complesso, normalizzato
    Returns:
      mz, mx, cz: tensori float32 shape (batch,)
    """
    if states.ndim != 2:
        raise ValueError(f"states deve avere shape (batch, dim), ricevuto: {tuple(states.shape)}")

    batch, dim = states.shape
    n_qubits = int(z_eigs.shape[0])
    if dim != int(z_eigs.shape[1]):
        raise ValueError(f"dim mismatch: states dim={dim}, z_eigs dim={int(z_eigs.shape[1])}")

    # --- Z e ZZ: operatori diagonali -> uso delle probabilita' |psi|^2
    probs = torch.abs(states) ** 2  # (batch, dim), float32/float64
    probs = probs.to(torch.float32)

    exp_z_sites = probs @ z_eigs.T  # (batch, n_qubits)
    mz = exp_z_sites.mean(dim=1)  # (batch,)

    if n_qubits >= 2:
        exp_zz_pairs = probs @ zz_nn_eigs.T  # (batch, n_qubits-1)
        cz = exp_zz_pairs.mean(dim=1)  # (batch,)
    else:
        cz = torch.zeros((batch,), device=states.device, dtype=torch.float32)

    # --- X: flip del bit i-esimo (non diagonale)
    psi_conj = states.conj()
    mx_sum = torch.zeros((batch,), device=states.device, dtype=torch.float32)
    for i in range(n_qubits):
        flipped = states[:, x_flip_idx[i]]  # (batch, dim)
        exp_x_i = torch.sum(psi_conj * flipped, dim=1)  # (batch,) complex
        mx_sum += exp_x_i.real.to(torch.float32)
    mx = mx_sum / max(1, n_qubits)

    return mz, mx, cz

