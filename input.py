from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import config


I = torch.eye(2, dtype=torch.complex64)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

SQRT2_INV = 1.0 / math.sqrt(2.0)
LOCAL_CLIFFORD_STATES = (
    ("0", torch.tensor([1.0, 0.0], dtype=torch.complex64)),
    ("1", torch.tensor([0.0, 1.0], dtype=torch.complex64)),
    ("+", torch.tensor([SQRT2_INV, SQRT2_INV], dtype=torch.complex64)),
    ("-", torch.tensor([SQRT2_INV, -SQRT2_INV], dtype=torch.complex64)),
    ("+i", torch.tensor([SQRT2_INV, 1j * SQRT2_INV], dtype=torch.complex64)),
    ("-i", torch.tensor([SQRT2_INV, -1j * SQRT2_INV], dtype=torch.complex64)),
)


@dataclass
class HamiltonianData:
    couplings: list[float]
    field_strength: float
    backend: str
    hamiltonian: torch.Tensor
    evolution_operator: torch.Tensor


@dataclass
class DatasetSplit:
    states: torch.Tensor
    initial_state_codes: list[int]
    initial_state_family: str
    support_size: int

    @property
    def inputs(self) -> torch.Tensor:
        return self.states[:, :-1]

    @property
    def targets(self) -> torch.Tensor:
        return self.states[:, 1:]

    @property
    def num_sequences(self) -> int:
        return int(self.states.shape[0])


@dataclass
class QuantumDatasetBundle:
    train: DatasetSplit
    test: DatasetSplit
    hamiltonian: HamiltonianData
    basis_support_size: int
    used_support_fraction: float
    initial_state_family_reason: str


class QuantumSequenceDataset(Dataset):
    def __init__(self, states: torch.Tensor):
        if states.ndim != 3:
            raise ValueError(f"states deve avere shape (batch, num_states, dim), ricevuto {tuple(states.shape)}")
        if states.shape[1] < 2:
            raise ValueError("Ogni traiettoria deve contenere almeno 2 stati.")
        self.states = states
        self.inputs = states[:, :-1]
        self.targets = states[:, 1:]

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index]


def get_pauli_string(operator_list: list[torch.Tensor]) -> torch.Tensor:
    result = operator_list[0]
    for op in operator_list[1:]:
        result = torch.kron(result, op)
    return result


def build_tfim_hamiltonian(
    n_qubits: int,
    couplings: torch.Tensor,
    field_strength: float = config.FIELD_STRENGTH,
) -> torch.Tensor:
    if n_qubits < 1:
        raise ValueError(f"n_qubits deve essere >= 1, ricevuto {n_qubits}")

    dim = 2 ** n_qubits
    hamiltonian = torch.zeros((dim, dim), dtype=torch.complex64)

    if n_qubits > 1 and couplings.numel() != n_qubits - 1:
        raise ValueError(
            f"Numero coupling non valido: attesi {n_qubits - 1}, ricevuti {couplings.numel()}"
        )

    for bond, coupling in enumerate(couplings):
        ops = [I] * n_qubits
        ops[bond] = Z
        ops[bond + 1] = Z
        hamiltonian -= float(coupling) * get_pauli_string(ops)

    for qubit in range(n_qubits):
        ops = [I] * n_qubits
        ops[qubit] = X
        hamiltonian -= float(field_strength) * get_pauli_string(ops)

    return hamiltonian


def choose_initial_state_family(total_sequences: int, n_qubits: int) -> tuple[str, int, str]:
    basis_support = 2 ** n_qubits
    support_fraction = total_sequences / basis_support

    if config.INITIAL_STATE_FAMILY == "basis":
        return "basis", basis_support, "forzato da config"

    if config.INITIAL_STATE_FAMILY == "local_clifford":
        return (
            "local_clifford",
            6 ** n_qubits,
            "forzato da config; prodotti di sole Pauli su bitstring non aumentano il supporto fisico",
        )

    if total_sequences > basis_support:
        return (
            "local_clifford",
            6 ** n_qubits,
            "supporto basis insufficiente; uso local Clifford perche' le sole Pauli non aumentano il supporto fisico",
        )

    if support_fraction > config.BASIS_SUPPORT_FRACTION_LIMIT:
        return (
            "local_clifford",
            6 ** n_qubits,
            "2K non abbastanza piccolo rispetto a 2^n; uso local Clifford perche' le sole Pauli non aumentano il supporto fisico",
        )

    return "basis", basis_support, "supporto basis sufficiente"


def sample_couplings(n_qubits: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if n_qubits == 1:
        return torch.empty((0,), dtype=torch.float32)
    return torch.normal(
        mean=config.COUPLING_MEAN,
        std=config.COUPLING_STD,
        size=(n_qubits - 1,),
        generator=generator,
    ).to(torch.float32)


def compute_evolution_operator(
    hamiltonian: torch.Tensor,
    time_step: float = config.TIME_STEP,
) -> tuple[torch.Tensor, str]:
    dim = int(hamiltonian.shape[0])

    if config.EVOLUTION_BACKEND == "exact_diag":
        backend = "exact_diag"
    elif config.EVOLUTION_BACKEND == "matrix_exp":
        backend = "matrix_exp"
    elif dim <= config.EXACT_DIAG_MAX_DIM:
        backend = "exact_diag"
    else:
        backend = "matrix_exp"

    if backend == "exact_diag":
        eigenvalues, eigenvectors = torch.linalg.eigh(hamiltonian)
        phases = torch.exp(1j * eigenvalues * float(time_step))
        evolution_operator = eigenvectors @ torch.diag(phases.to(torch.complex64)) @ eigenvectors.conj().T
        return evolution_operator.to(torch.complex64), backend

    evolution_operator = torch.matrix_exp(1j * hamiltonian * float(time_step))
    return evolution_operator.to(torch.complex64), backend


def basis_state_from_code(code: int, n_qubits: int) -> torch.Tensor:
    dim = 2 ** n_qubits
    state = torch.zeros((dim,), dtype=torch.complex64)
    state[int(code)] = 1.0 + 0.0j
    return state


def local_clifford_state_from_code(code: int, n_qubits: int) -> torch.Tensor:
    factors: list[torch.Tensor] = []
    remaining = int(code)
    for _ in range(n_qubits):
        remaining, digit = divmod(remaining, len(LOCAL_CLIFFORD_STATES))
        factors.append(LOCAL_CLIFFORD_STATES[digit][1])

    state = factors[-1]
    for factor in reversed(factors[:-1]):
        state = torch.kron(state, factor)
    return state.to(torch.complex64)


def initial_state_from_code(code: int, family: str, n_qubits: int) -> torch.Tensor:
    if family == "basis":
        return basis_state_from_code(code, n_qubits)
    if family == "local_clifford":
        return local_clifford_state_from_code(code, n_qubits)
    raise ValueError(f"Famiglia di stati iniziali non supportata: {family}")


def sample_initial_state_codes(total_sequences: int, support_size: int, seed: int) -> list[int]:
    if total_sequences > support_size:
        raise ValueError(
            f"Richiesti {total_sequences} stati iniziali distinti ma supporto disponibile={support_size}."
        )
    rng = random.Random(seed)
    return rng.sample(range(support_size), total_sequences)


def build_initial_states(
    codes: list[int],
    family: str,
    n_qubits: int,
) -> torch.Tensor:
    return torch.stack([initial_state_from_code(code, family, n_qubits) for code in codes], dim=0)


def evolve_sequences(
    initial_states: torch.Tensor,
    evolution_operator: torch.Tensor,
    num_states: int,
    device: str | torch.device = config.DEVICE,
) -> torch.Tensor:
    device = torch.device(device)
    current = initial_states.to(device)
    operator = evolution_operator.to(device)

    trajectories = torch.empty(
        (initial_states.shape[0], num_states, initial_states.shape[1]),
        dtype=torch.complex64,
        device=device,
    )
    trajectories[:, 0] = current

    for step in range(1, num_states):
        current = torch.einsum("ij,bj->bi", operator, current)
        current = current / torch.linalg.vector_norm(current, dim=-1, keepdim=True).clamp(min=1e-8)
        trajectories[:, step] = current

    return trajectories.cpu()


def generate_fixed_tfim_dataset(
    train_sequences: int = config.TRAIN_SEQUENCES,
    test_sequences: int = config.TEST_SEQUENCES,
    n_qubits: int = config.N_QUBITS,
    num_states: int = config.NUM_STATES,
    seed: int = config.SEED,
) -> QuantumDatasetBundle:
    total_sequences = int(train_sequences) + int(test_sequences)
    family, support_size, reason = choose_initial_state_family(total_sequences, n_qubits)
    basis_support_size = 2 ** n_qubits
    used_support_fraction = total_sequences / basis_support_size

    couplings = sample_couplings(n_qubits, seed + 11)
    hamiltonian = build_tfim_hamiltonian(
        n_qubits=n_qubits,
        couplings=couplings,
        field_strength=config.FIELD_STRENGTH,
    )
    evolution_operator, backend = compute_evolution_operator(hamiltonian, config.TIME_STEP)

    initial_state_codes = sample_initial_state_codes(total_sequences, support_size, seed + 23)
    initial_states = build_initial_states(initial_state_codes, family, n_qubits)
    all_states = evolve_sequences(initial_states, evolution_operator, num_states)

    train_split = DatasetSplit(
        states=all_states[:train_sequences],
        initial_state_codes=initial_state_codes[:train_sequences],
        initial_state_family=family,
        support_size=support_size,
    )
    test_split = DatasetSplit(
        states=all_states[train_sequences:],
        initial_state_codes=initial_state_codes[train_sequences:],
        initial_state_family=family,
        support_size=support_size,
    )

    return QuantumDatasetBundle(
        train=train_split,
        test=test_split,
        hamiltonian=HamiltonianData(
            couplings=[float(value) for value in couplings.tolist()],
            field_strength=float(config.FIELD_STRENGTH),
            backend=backend,
            hamiltonian=hamiltonian.cpu(),
            evolution_operator=evolution_operator.cpu(),
        ),
        basis_support_size=basis_support_size,
        used_support_fraction=float(used_support_fraction),
        initial_state_family_reason=reason,
    )
