from __future__ import annotations

import argparse
from pathlib import Path

import torch


I2 = torch.eye(2, dtype=torch.complex64)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)


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

        default_j = float(config.COUPLING_MEAN)
        default_h = float(config.FIELD_STRENGTH)
        default_dt = float(config.TIME_STEP)
    except Exception:
        default_j = 1.0
        default_h = 1.0
        default_dt = 1.0

    parser = argparse.ArgumentParser(description="Genera un dataset Haar+TFIM per 4 qubit.")
    parser.add_argument("--num-trajectories", type=int, default=5000)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--num-states", type=int, default=12)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--j", type=float, default=default_j)
    parser.add_argument("--h", type=float, default=default_h)
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
        raise ValueError("Questo script e' configurato per il caso richiesto a 4 qubit.")
    if args.num_states < 2:
        raise ValueError("num_states deve essere >= 2.")

    dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    dim = 2 ** args.n_qubits

    initial_states = sample_haar_random_states(
        num_samples=args.num_trajectories,
        dim=dim,
        seed=args.seed,
        dtype=dtype,
    )
    all_params = sample_tfim_params(args.num_trajectories, seed=args.seed + 1)
    unitaries = []
    for coupling_j, field_h in all_params.tolist():
        hamiltonian = build_tfim_hamiltonian(
            n_qubits=args.n_qubits,
            coupling_j=float(coupling_j),
            field_h=float(field_h),
        ).to(dtype)
        unitaries.append(torch.matrix_exp((-1j * float(args.dt)) * hamiltonian))
    unitary_tensor = torch.stack(unitaries, dim=0)
    trajectories = evolve_batched_per_trajectory(
        initial_states=initial_states,
        unitaries=unitary_tensor,
        num_states=args.num_states,
    )

    if trajectories.shape != (args.num_trajectories, args.num_states, dim):
        raise RuntimeError(f"Shape inattesa del dataset: {tuple(trajectories.shape)}")

    train_states = trajectories[: args.train_size].contiguous()
    test_states = trajectories[args.train_size :].contiguous()
    train_params = all_params[: args.train_size].contiguous().to(torch.float32)
    test_params = all_params[args.train_size :].contiguous().to(torch.float32)

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
    print(f"  Hamiltonian: TFIM open chain | n_qubits={args.n_qubits} | dim={dim}")
    print(f"  Params: J,h ~ U(0.2, 2.0) per traiettoria | dt={args.dt}")
    print(f"  Tensor shape totale: {tuple(trajectories.shape)} | dtype={trajectories.dtype}")
    print(f"  Train: {tuple(train_states.shape)} -> {train_path}")
    print(f"  Test:  {tuple(test_states.shape)} -> {test_path}")
    print(f"  Train params: {tuple(train_params.shape)} -> {train_params_path}")
    print(f"  Test params:  {tuple(test_params.shape)} -> {test_params_path}")


if __name__ == "__main__":
    main()
