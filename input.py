from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import config


I = torch.eye(2, dtype=torch.complex64)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

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
    if config.INITIAL_STATE_FAMILY == "x_basis":
        support_size = 2 ** n_qubits
        reason = (
            "stati iniziali solo in base X clampata; campionamento con rimpiazzo attivo"
            if total_sequences > support_size
            else "stati iniziali solo in base X clampata; campionamento senza rimpiazzo"
        )
        family = "x_basis"
    elif config.INITIAL_STATE_FAMILY == "xyz_basis":
        support_size = 3 * (2 ** n_qubits)
        reason = (
            "bitstring binaria convertita in una base X/Y/Z scelta per traiettoria; "
            "campionamento con rimpiazzo attivo"
            if total_sequences > support_size
            else "bitstring binaria convertita in una base X/Y/Z scelta per traiettoria; "
            "campionamento senza rimpiazzo"
        )
        family = "xyz_basis"
    else:
        raise ValueError(
            f"INITIAL_STATE_FAMILY={config.INITIAL_STATE_FAMILY!r} non supportata."
        )

    if total_sequences > support_size and not config.INITIAL_STATE_SAMPLE_WITH_REPLACEMENT:
        raise ValueError(
            f"Richiesti {total_sequences} stati ma il supporto {family} e' {support_size}. "
            "Attiva QSP_INITIAL_STATE_SAMPLE_WITH_REPLACEMENT=1 per campionare con rimpiazzo."
        )
    return family, support_size, reason


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


def bits_from_code(code: int, n_qubits: int) -> list[int]:
    code = int(code)
    return [int((code >> shift) & 1) for shift in range(n_qubits - 1, -1, -1)]


def x_basis_state_from_code(code: int, n_qubits: int) -> torch.Tensor:
    bits = bits_from_code(code, n_qubits)
    sqrt2_inv = 1.0 / (2.0 ** 0.5)
    plus = torch.tensor([sqrt2_inv, sqrt2_inv], dtype=torch.complex64)
    minus = torch.tensor([sqrt2_inv, -sqrt2_inv], dtype=torch.complex64)
    state = plus if bits[0] == 0 else minus
    for bit in bits[1:]:
        state = torch.kron(state, plus if bit == 0 else minus)
    return state.to(torch.complex64)


def _decode_xyz_basis_code(code: int, n_qubits: int) -> tuple[int, int]:
    bitstring_support = 2 ** n_qubits
    basis_index = int(code) // bitstring_support
    bit_code = int(code) % bitstring_support
    if basis_index not in {0, 1, 2}:
        raise ValueError(f"Codice xyz_basis non valido: {code}")
    return basis_index, bit_code


def _basis_label_from_index(basis_index: int) -> str:
    return ("X", "Y", "Z")[basis_index]


def _local_basis_state(bit: int, basis_label: str) -> torch.Tensor:
    if basis_label == "X":
        sqrt2_inv = 1.0 / (2.0 ** 0.5)
        return torch.tensor(
            [sqrt2_inv, sqrt2_inv if bit == 0 else -sqrt2_inv],
            dtype=torch.complex64,
        )
    if basis_label == "Y":
        sqrt2_inv = 1.0 / (2.0 ** 0.5)
        phase = 1j if bit == 0 else -1j
        return torch.tensor([sqrt2_inv, sqrt2_inv * phase], dtype=torch.complex64)
    if basis_label == "Z":
        return torch.tensor([1.0, 0.0], dtype=torch.complex64) if bit == 0 else torch.tensor(
            [0.0, 1.0], dtype=torch.complex64
        )
    raise ValueError(f"Base locale non supportata: {basis_label}")


def xyz_basis_state_from_code(code: int, n_qubits: int) -> torch.Tensor:
    basis_index, bit_code = _decode_xyz_basis_code(code, n_qubits)
    basis_label = _basis_label_from_index(basis_index)
    bits = bits_from_code(bit_code, n_qubits)
    state = _local_basis_state(bits[0], basis_label)
    for bit in bits[1:]:
        state = torch.kron(state, _local_basis_state(bit, basis_label))
    return state.to(torch.complex64)


def initial_state_from_code(code: int, family: str, n_qubits: int) -> torch.Tensor:
    if family == "x_basis":
        return x_basis_state_from_code(code, n_qubits)
    if family == "xyz_basis":
        return xyz_basis_state_from_code(code, n_qubits)
    raise ValueError(f"Famiglia di stati iniziali non supportata: {family}")


def sample_initial_state_codes(total_sequences: int, support_size: int, seed: int) -> list[int]:
    if total_sequences > support_size and not config.INITIAL_STATE_SAMPLE_WITH_REPLACEMENT:
        raise ValueError(
            f"Richiesti {total_sequences} stati iniziali distinti ma supporto disponibile={support_size}."
        )
    rng = random.Random(seed)
    if total_sequences <= support_size:
        return rng.sample(range(support_size), total_sequences)
    return [rng.randrange(support_size) for _ in range(total_sequences)]


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

    def clamp_global_phase_first_amplitude_batch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Clamping gauge per batch:
        impone che x[:, 0] abbia fase zero (quindi x[:, 0] sia reale >= 0) per ogni traiettoria.
        Se |x[:,0]| ~ 0 non clampa quella traiettoria per evitare instabilita' numeriche.
        """
        if x.ndim != 2:
            raise ValueError(f"x deve essere 2D (batch, dim), ricevuto shape={tuple(x.shape)}")
        a0 = x[:, 0]
        mag = torch.abs(a0)
        mask = mag > eps
        # factor = exp(-i*angle(a0)) -> a0 * factor diventa reale e positivo.
        factor = torch.ones_like(a0)
        factor[mask] = torch.conj(a0[mask] / mag[mask])
        return x * factor[:, None]

    trajectories = torch.empty(
        (initial_states.shape[0], num_states, initial_states.shape[1]),
        dtype=torch.complex64,
        device=device,
    )
    current = clamp_global_phase_first_amplitude_batch(current)
    trajectories[:, 0] = current

    for step in range(1, num_states):
        current = torch.einsum("ij,bj->bi", operator, current)
        current = current / torch.linalg.vector_norm(current, dim=-1, keepdim=True).clamp(min=1e-8)
        current = clamp_global_phase_first_amplitude_batch(current)
        trajectories[:, step] = current

    return trajectories.cpu()


def _format_complex(z: complex, ndigits: int = 6) -> str:
    return f"{z.real:+.{ndigits}f}{z.imag:+.{ndigits}f}j"


def _print_clamped_dataset_audit(
    split_name: str,
    states: torch.Tensor,
    codes: list[int],
    n_qubits: int,
) -> None:
    if not config.CLAMP_AUDIT_PRINT:
        return

    max_sequences = min(int(config.CLAMP_AUDIT_MAX_SEQUENCES), int(states.shape[0]))
    max_states = min(int(config.CLAMP_AUDIT_MAX_STATES), int(states.shape[1]))
    print(f"\n[ClampAudit:{split_name}] showing {max_sequences} sequence(s), first {max_states} state(s)")
    for seq_idx in range(max_sequences):
        code = int(codes[seq_idx])
        basis_label = "X"
        bit_code = code
        if config.INITIAL_STATE_FAMILY == "xyz_basis":
            basis_index, bit_code = _decode_xyz_basis_code(code, n_qubits)
            basis_label = _basis_label_from_index(basis_index)
        bits = bits_from_code(bit_code, n_qubits)
        bitstring = "".join(str(bit) for bit in bits)
        if config.CLAMP_AUDIT_PRINT_BITSTRINGS:
            print(
                f"  seq={seq_idx:03d} code={code} bitstring={bitstring} "
                f"(base {basis_label})"
            )
        for t in range(max_states):
            state = states[seq_idx, t]
            a0 = state[0].item()
            print(
                f"    t={t:02d} psi[0]={_format_complex(a0)} "
                f"| |psi[0]|={abs(a0):.6f} angle={float(torch.angle(state[0]).item()):+.6f} rad"
            )
            if config.CLAMP_AUDIT_PRINT_COEFFS:
                for idx in range(state.numel()):
                    print(f"      coeff[{idx:>2d}]={_format_complex(state[idx].item())}")


def generate_fixed_tfim_dataset(
    train_sequences: int = config.TRAIN_SEQUENCES,
    test_sequences: int = config.TEST_SEQUENCES,
    n_qubits: int = config.N_QUBITS,
    num_states: int = config.NUM_STATES,
    seed: int = config.SEED,
) -> QuantumDatasetBundle:
    total_sequences = int(train_sequences) + int(test_sequences)
    family, support_size, reason = choose_initial_state_family(total_sequences, n_qubits)
    basis_support_size = support_size
    used_support_fraction = total_sequences / support_size

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

    _print_clamped_dataset_audit("train", train_split.states, train_split.initial_state_codes, n_qubits)
    _print_clamped_dataset_audit("test", test_split.states, test_split.initial_state_codes, n_qubits)

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
