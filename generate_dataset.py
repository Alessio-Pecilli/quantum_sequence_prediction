from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch


I2 = torch.eye(2, dtype=torch.complex64)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
N_OP = 0.5 * (I2 - Z)
P0_OP = 0.5 * (I2 + Z)


def kron_all(operators: list[torch.Tensor]) -> torch.Tensor:
    result = operators[0]
    for op in operators[1:]:
        result = torch.kron(result, op)
    return result


def build_tfim_hamiltonian(
    n_qubits: int,
    coupling_j: float,
    field_h: float,
) -> torch.Tensor:
    dim = 2 ** n_qubits
    hamiltonian = torch.zeros((dim, dim), dtype=torch.complex64)

    for bond in range(n_qubits - 1):
        ops = [I2] * n_qubits
        ops[bond] = Z
        ops[bond + 1] = Z
        hamiltonian -= float(coupling_j) * kron_all(ops)

    for qubit in range(n_qubits):
        ops = [I2] * n_qubits
        ops[qubit] = X
        hamiltonian -= float(field_h) * kron_all(ops)

    return hamiltonian


def build_xxz_hamiltonian(
    n_qubits: int,
    coupling_j: float,
    anisotropy_delta: float,
) -> torch.Tensor:
    dim = 2 ** n_qubits
    hamiltonian = torch.zeros((dim, dim), dtype=torch.complex64)
    for bond in range(n_qubits - 1):
        ops_xx = [I2] * n_qubits
        ops_yy = [I2] * n_qubits
        ops_zz = [I2] * n_qubits
        ops_xx[bond] = X
        ops_xx[bond + 1] = X
        ops_yy[bond] = Y
        ops_yy[bond + 1] = Y
        ops_zz[bond] = Z
        ops_zz[bond + 1] = Z
        hamiltonian += float(coupling_j) * (kron_all(ops_xx) + kron_all(ops_yy))
        hamiltonian += float(anisotropy_delta) * kron_all(ops_zz)
    return hamiltonian


def jw_annihilation_operator(n_qubits: int, orbital: int) -> torch.Tensor:
    lowering = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64)
    ops: list[torch.Tensor] = []
    for index in range(n_qubits):
        if index < orbital:
            ops.append(Z)
        elif index == orbital:
            ops.append(lowering)
        else:
            ops.append(I2)
    return kron_all(ops)


def jw_creation_operator(n_qubits: int, orbital: int) -> torch.Tensor:
    return jw_annihilation_operator(n_qubits, orbital).conj().T


def jw_number_operator(n_qubits: int, orbital: int) -> torch.Tensor:
    ops = [I2] * n_qubits
    ops[orbital] = N_OP
    return kron_all(ops)


def build_fermi_hubbard_hamiltonian(
    n_qubits: int,
    hopping_t: float,
    onsite_u: float,
) -> torch.Tensor:
    if n_qubits != 4:
        raise ValueError("Fermi-Hubbard qui richiede 4 qubit (2 siti spinful).")
    dim = 2 ** n_qubits
    hamiltonian = torch.zeros((dim, dim), dtype=torch.complex64)

    # Jordan-Wigner per 2 siti spinful:
    # [0, 1] = sito 0 (up, down), [2, 3] = sito 1 (up, down).
    for left, right in ((0, 2), (1, 3)):
        cdag_l = jw_creation_operator(n_qubits, left)
        c_l = jw_annihilation_operator(n_qubits, left)
        cdag_r = jw_creation_operator(n_qubits, right)
        c_r = jw_annihilation_operator(n_qubits, right)
        hamiltonian -= float(hopping_t) * (cdag_l @ c_r + cdag_r @ c_l)

    for up_orbital, down_orbital in ((0, 1), (2, 3)):
        n_up = jw_number_operator(n_qubits, up_orbital)
        n_down = jw_number_operator(n_qubits, down_orbital)
        hamiltonian += float(onsite_u) * (n_up @ n_down)
    return hamiltonian


def build_max3sat_hamiltonian(
    n_qubits: int,
    clause_weights: torch.Tensor,
) -> torch.Tensor:
    if n_qubits != 4:
        raise ValueError("Max-3-SAT qui richiede 4 qubit.")
    if clause_weights.ndim != 1 or clause_weights.numel() != 2:
        raise ValueError("clause_weights deve avere shape (2,) per le clausole (0,1,2) e (1,2,3).")
    dim = 2 ** n_qubits
    hamiltonian = torch.zeros((dim, dim), dtype=torch.complex64)

    # Due clausole locali a 3 variabili su 4 qubit:
    # C0=(0,1,2), C1=(1,2,3). Ogni clausola penalizza |000>.
    for clause_start, value in enumerate(clause_weights.tolist()):
        ops = [I2] * n_qubits
        for qubit in range(clause_start, clause_start + 3):
            ops[qubit] = P0_OP
        hamiltonian += float(value) * kron_all(ops)
    return hamiltonian


def sample_haar_random_states(
    num_samples: int,
    dim: int,
    seed: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    real = torch.randn((num_samples, dim), generator=generator, dtype=torch.float32)
    imag = torch.randn((num_samples, dim), generator=generator, dtype=torch.float32)
    states = torch.complex(real, imag)
    norms = torch.linalg.vector_norm(states, dim=-1, keepdim=True).clamp(min=1e-8)
    return (states / norms).to(dtype)


def sample_tfim_params(
    num_samples: int,
    seed: int,
    low: float = 0.2,
    high: float = 2.0,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    params = torch.empty((num_samples, 2), dtype=torch.float32)
    params.uniform_(float(low), float(high), generator=generator)
    return params


def sample_hamiltonian_instance(
    n_qubits: int,
    rng: random.Random,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    h_id = int(rng.choice([0, 1, 2, 3]))
    if h_id == 0:
        p1 = float(rng.uniform(0.2, 2.0))
        p2 = float(rng.uniform(0.2, 2.0))
        hamiltonian = build_tfim_hamiltonian(n_qubits, p1, p2)
        params = torch.tensor([0.0, p1, p2, 0.0, 0.0, 0.0], dtype=torch.float32)
        return h_id, hamiltonian, params
    if h_id == 1:
        p1 = float(rng.uniform(0.2, 2.0))
        p2 = float(rng.uniform(0.2, 2.0))
        hamiltonian = build_xxz_hamiltonian(n_qubits, p1, p2)
        params = torch.tensor([1.0, p1, p2, 0.0, 0.0, 0.0], dtype=torch.float32)
        return h_id, hamiltonian, params
    if h_id == 2:
        p1 = float(rng.uniform(0.2, 2.0))
        p2 = float(rng.uniform(0.2, 2.0))
        hamiltonian = build_fermi_hubbard_hamiltonian(n_qubits, p1, p2)
        params = torch.tensor([2.0, p1, p2, 0.0, 0.0, 0.0], dtype=torch.float32)
        return h_id, hamiltonian, params

    p1 = float(rng.uniform(0.2, 2.0))
    p2 = float(rng.uniform(0.2, 2.0))
    clause_weights = torch.tensor([p1, p2], dtype=torch.float32)
    hamiltonian = build_max3sat_hamiltonian(n_qubits, clause_weights)
    params = torch.tensor([3.0, p1, p2, 0.0, 0.0, 0.0], dtype=torch.float32)
    return h_id, hamiltonian, params


def evolve_batched_per_trajectory(
    initial_states: torch.Tensor,
    unitaries: torch.Tensor,
    num_states: int,
) -> torch.Tensor:
    num_samples, dim = initial_states.shape
    trajectories = torch.empty((num_samples, num_states, dim), dtype=initial_states.dtype)
    current = initial_states
    trajectories[:, 0, :] = current

    # Stati memorizzati come righe: psi_{t+1} = psi_t @ U^T, con una U diversa per traiettoria.
    right_operators = unitaries.transpose(1, 2).contiguous()
    for t in range(1, num_states):
        current = torch.bmm(current.unsqueeze(1), right_operators).squeeze(1)
        current = current / torch.linalg.vector_norm(current, dim=-1, keepdim=True).clamp(min=1e-8)
        trajectories[:, t, :] = current

    return trajectories


def parse_args() -> argparse.Namespace:
    try:
        import config

        default_dt = float(config.TIME_STEP)
        default_num_states = int(config.NUM_STATES)
        default_n_qubits = int(config.N_QUBITS)
    except Exception:
        default_dt = 1.0
        default_num_states = 101
        default_n_qubits = 4

    parser = argparse.ArgumentParser(description="Genera un dataset Haar multi-Hamiltoniano per 4 qubit.")
    parser.add_argument("--num-trajectories", type=int, default=5000)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--num-states", type=int, default=default_num_states)
    parser.add_argument("--n-qubits", type=int, default=default_n_qubits)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dt", type=float, default=default_dt)
    parser.add_argument(
        "--dtype",
        choices=("complex64", "complex128"),
        default="complex64",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_size + args.test_size != args.num_trajectories:
        raise ValueError("train_size + test_size deve coincidere con num_trajectories.")
    if args.n_qubits != 4:
        raise ValueError("Questo script e' configurato per il caso richiesto a 4 qubit (dim=16).")
    if args.num_states < 2:
        raise ValueError("num_states deve essere >= 2.")

    dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    dim = 2 ** args.n_qubits

    rng = random.Random(args.seed + 11)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    all_states: list[torch.Tensor] = []
    all_params: list[torch.Tensor] = []
    for _ in range(args.num_trajectories):
        initial_state = sample_haar_random_states(1, dim, seed=int(torch.randint(0, 10_000_000, (1,), generator=generator).item()), dtype=dtype)[0]
        _, hamiltonian, params = sample_hamiltonian_instance(args.n_qubits, rng)
        unitary = torch.linalg.matrix_exp((-1j * float(args.dt)) * hamiltonian.to(dtype))
        trajectory = torch.empty((args.num_states, dim), dtype=dtype)
        current = initial_state
        trajectory[0] = current
        for step in range(1, args.num_states):
            current = unitary @ current
            current = current / torch.linalg.vector_norm(current).clamp(min=1e-8)
            trajectory[step] = current
        all_states.append(trajectory)
        all_params.append(params)
    trajectories = torch.stack(all_states, dim=0)
    all_params_tensor = torch.stack(all_params, dim=0).to(torch.float32)

    if trajectories.shape != (args.num_trajectories, args.num_states, dim):
        raise RuntimeError(f"Shape inattesa del dataset: {tuple(trajectories.shape)}")

    train_states = trajectories[: args.train_size].contiguous()
    test_states = trajectories[args.train_size :].contiguous()
    train_params = all_params_tensor[: args.train_size].contiguous().to(torch.float32)
    test_params = all_params_tensor[args.train_size :].contiguous().to(torch.float32)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_states.pt"
    test_path = output_dir / "test_states.pt"
    train_params_path = output_dir / "train_params.pt"
    test_params_path = output_dir / "test_params.pt"
    torch.save(train_states, train_path)
    torch.save(test_states, test_path)
    torch.save(train_params, train_params_path)
    torch.save(test_params, test_params_path)

    print("Dataset generato correttamente")
    print(f"  Hamiltonians: TFIM/XXZ/Fermi-Hubbard/Max-3-SAT | n_qubits={args.n_qubits} | dim={dim}")
    print(f"  Params vector shape: [H_ID, p1, p2, 0, 0, 0] | dt={args.dt}")
    print(f"  Tensor shape totale: {tuple(trajectories.shape)} | dtype={trajectories.dtype}")
    print(f"  Train: {tuple(train_states.shape)} -> {train_path}")
    print(f"  Test:  {tuple(test_states.shape)} -> {test_path}")
    print(f"  Train params: {tuple(train_params.shape)} -> {train_params_path}")
    print(f"  Test params:  {tuple(test_params.shape)} -> {test_params_path}")


if __name__ == "__main__":
    main()
