from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import config
import generate_dataset as hgen


I = torch.eye(2, dtype=torch.complex64)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
PARAM_VECTOR_DIM = 6


def _dist_is_initialized() -> bool:
    return bool(dist.is_available() and dist.is_initialized())


def _dist_rank_world() -> tuple[int, int]:
    if not _dist_is_initialized():
        return 0, 1
    return int(dist.get_rank()), int(dist.get_world_size())


def _split_count(total: int, rank: int, world_size: int) -> int:
    base = int(total) // int(world_size)
    rem = int(total) % int(world_size)
    return base + (1 if rank < rem else 0)


def _dist_log(message: str, *, force: bool = False) -> None:
    rank, world_size = _dist_rank_world()
    if not force and world_size <= 1:
        return
    print(f"[dataset][rank {rank}/{world_size}] {message}", flush=True)

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
    params: torch.Tensor
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
    def __init__(self, states: torch.Tensor, params: torch.Tensor):
        if states.ndim != 3:
            raise ValueError(f"states deve avere shape (batch, num_states, dim), ricevuto {tuple(states.shape)}")
        if states.shape[1] < 2:
            raise ValueError("Ogni traiettoria deve contenere almeno 2 stati.")
        if params.ndim != 2 or params.shape[1] != PARAM_VECTOR_DIM:
            raise ValueError(
                f"params deve avere shape (batch, {PARAM_VECTOR_DIM}), ricevuto {tuple(params.shape)}"
            )
        if params.shape[0] != states.shape[0]:
            raise ValueError(
                "states e params devono avere lo stesso numero di traiettorie: "
                f"{states.shape[0]} vs {params.shape[0]}"
            )
        self.states = states
        self.params = params.to(torch.float32)
        self.inputs = states[:, :-1]
        self.targets = states[:, 1:]

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index], self.params[index]


def build_uniform_couplings(
    n_qubits: int,
    coupling_strength: float = config.COUPLING_MEAN,
) -> torch.Tensor:
    if n_qubits < 1:
        raise ValueError(f"n_qubits deve essere >= 1, ricevuto {n_qubits}")
    if n_qubits == 1:
        return torch.empty((0,), dtype=torch.float32)
    return torch.full((n_qubits - 1,), float(coupling_strength), dtype=torch.float32)


def sample_haar_random_states(
    num_samples: int,
    dim: int,
    seed: int,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    real = torch.randn((num_samples, dim), generator=generator, dtype=torch.float32)
    imag = torch.randn((num_samples, dim), generator=generator, dtype=torch.float32)
    states = torch.complex(real, imag)
    # Gaussiani complessi iid + normalizzazione L2 batch-wise -> campionamento Haar.
    norms = torch.linalg.vector_norm(states, dim=-1, keepdim=True).clamp(min=1e-8)
    return (states / norms).to(dtype)


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
    forced_x_basis = bool(config.FORCE_X_BASIS_ONLY)
    requested_family = "x_basis" if forced_x_basis else config.INITIAL_STATE_FAMILY

    if requested_family == "x_basis":
        support_size = 2 ** n_qubits
        reason = (
            "stati iniziali solo in base X clampata; campionamento con rimpiazzo attivo"
            if total_sequences > support_size
            else "stati iniziali solo in base X clampata; campionamento senza rimpiazzo"
        )
        if forced_x_basis and config.INITIAL_STATE_FAMILY != "x_basis":
            reason = "override FORCE_X_BASIS_ONLY attivo; " + reason
        family = "x_basis"
    elif requested_family == "xyz_basis":
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
            "Per xyz_basis contano come distinti le coppie (bitstring, base globale X/Y/Z). "
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


def evolve_haar_tfim_sequences_with_params(
    initial_states: torch.Tensor,
    params: torch.Tensor,
    num_states: int,
    n_qubits: int,
    time_step: float = config.TIME_STEP,
    device: str | torch.device = config.DEVICE,
) -> tuple[torch.Tensor, str]:
    if params.ndim != 2 or params.shape[1] != PARAM_VECTOR_DIM:
        raise ValueError(
            f"params deve avere shape (batch, {PARAM_VECTOR_DIM}), ricevuto {tuple(params.shape)}"
        )
    if initial_states.shape[0] != params.shape[0]:
        raise ValueError(
            "initial_states e params devono avere lo stesso batch size: "
            f"{initial_states.shape[0]} vs {params.shape[0]}"
        )

    trajectories = []
    backend_name = "per_trajectory_exact_diag"
    for initial_state, trajectory_params in zip(initial_states, params):
        coupling_j = float(trajectory_params[1].item())
        field_h = float(trajectory_params[2].item())
        couplings = build_uniform_couplings(n_qubits, coupling_strength=coupling_j)
        hamiltonian = build_tfim_hamiltonian(
            n_qubits=n_qubits,
            couplings=couplings,
            field_strength=field_h,
        )
        evolution_operator, backend = compute_evolution_operator(hamiltonian, time_step)
        backend_name = f"per_trajectory_{backend}"
        trajectory = evolve_sequences(
            initial_states=initial_state.unsqueeze(0),
            evolution_operator=evolution_operator,
            num_states=num_states,
            device=device,
        )
        trajectories.append(trajectory[0])
    return torch.stack(trajectories, dim=0), backend_name


def _format_complex(z: complex, ndigits: int = 6) -> str:
    return f"{z.real:+.{ndigits}f}{z.imag:+.{ndigits}f}j"


def _print_clamped_dataset_audit(
    split_name: str,
    states: torch.Tensor,
    codes: list[int],
    n_qubits: int,
    family: str,
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
        if family == "xyz_basis":
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
    *,
    enable_distributed: bool = True,
) -> QuantumDatasetBundle:
    if (
        enable_distributed
        and config.HPC_DISTRIBUTED
        and config.HPC_DISTRIBUTED_DATASET
        and _dist_is_initialized()
    ):
        rank, world_size = _dist_rank_world()
        local_train = _split_count(int(train_sequences), rank, world_size)
        local_test = _split_count(int(test_sequences), rank, world_size)
        local_seed = int(seed) + int(config.HPC_DIST_SEED_STRIDE) * int(rank)
        if rank == 0:
            print(
                (
                    f"[dataset] distributed generation enabled | world_size={world_size} | "
                    f"global train={int(train_sequences)} test={int(test_sequences)} | "
                    f"n_qubits={int(n_qubits)} num_states={int(num_states)}"
                ),
                flush=True,
            )
        _dist_log(
            f"local shard start | train={local_train} test={local_test} seed={local_seed}"
        )

        if (local_train + local_test) > 0:
            local_bundle = generate_fixed_tfim_dataset(
                train_sequences=local_train,
                test_sequences=local_test,
                n_qubits=n_qubits,
                num_states=num_states,
                seed=local_seed,
                enable_distributed=False,
            )
        else:
            dim = 2 ** int(n_qubits)
            empty_hamiltonian = HamiltonianData(
                couplings=[],
                field_strength=float("nan"),
                backend="empty_shard",
                hamiltonian=torch.empty((0, 0), dtype=torch.complex64),
                evolution_operator=torch.empty((0, 0), dtype=torch.complex64),
            )
            local_bundle = QuantumDatasetBundle(
                train=DatasetSplit(
                    states=torch.empty((0, int(num_states), dim), dtype=torch.complex64),
                    params=torch.empty((0, PARAM_VECTOR_DIM), dtype=torch.float32),
                    initial_state_codes=[],
                    initial_state_family="empty_shard",
                    support_size=0,
                ),
                test=DatasetSplit(
                    states=torch.empty((0, int(num_states), dim), dtype=torch.complex64),
                    params=torch.empty((0, PARAM_VECTOR_DIM), dtype=torch.float32),
                    initial_state_codes=[],
                    initial_state_family="empty_shard",
                    support_size=0,
                ),
                hamiltonian=empty_hamiltonian,
                basis_support_size=0,
                used_support_fraction=0.0,
                initial_state_family_reason="empty shard",
            )
        _dist_log(
            (
                "local shard ready | "
                f"train_shape={tuple(local_bundle.train.states.shape)} "
                f"test_shape={tuple(local_bundle.test.states.shape)}"
            )
        )

        gathered: list[object] = [None for _ in range(world_size)]
        payload = {
            "train_states": local_bundle.train.states,
            "train_params": local_bundle.train.params,
            "train_codes": local_bundle.train.initial_state_codes,
            "test_states": local_bundle.test.states,
            "test_params": local_bundle.test.params,
            "test_codes": local_bundle.test.initial_state_codes,
            "hamiltonian": local_bundle.hamiltonian,
            "basis_support_size": int(local_bundle.basis_support_size),
            "used_support_fraction": float(local_bundle.used_support_fraction),
            "initial_state_family_reason": str(local_bundle.initial_state_family_reason),
            "initial_state_family": str(local_bundle.train.initial_state_family),
        }
        dist.all_gather_object(gathered, payload)

        reconstructed: QuantumDatasetBundle | None = None
        if rank == 0:
            chunks = [item for item in gathered if isinstance(item, dict)]
            non_empty_chunks = [
                item
                for item in chunks
                if int(item["train_states"].shape[0]) + int(item["test_states"].shape[0]) > 0
            ]
            train_states = torch.cat([item["train_states"] for item in chunks], dim=0).contiguous()
            train_params = torch.cat([item["train_params"] for item in chunks], dim=0).contiguous()
            test_states = torch.cat([item["test_states"] for item in chunks], dim=0).contiguous()
            test_params = torch.cat([item["test_params"] for item in chunks], dim=0).contiguous()
            train_codes: list[int] = []
            test_codes: list[int] = []
            for item in chunks:
                train_codes.extend([int(v) for v in item["train_codes"]])
                test_codes.extend([int(v) for v in item["test_codes"]])

            reference_chunk = non_empty_chunks[0] if non_empty_chunks else chunks[0]
            train_family = str(reference_chunk["initial_state_family"])
            reason = (
                f"distributed_generation(world_size={world_size}) | "
                + " | ".join(str(item["initial_state_family_reason"]) for item in chunks)
            )
            shard_sizes = [
                f"rank{idx}:train={int(item['train_states'].shape[0])},test={int(item['test_states'].shape[0])}"
                for idx, item in enumerate(chunks)
            ]
            print(
                (
                    "[dataset] gathered distributed shards | "
                    + " | ".join(shard_sizes)
                    + f" | final train={int(train_sequences)} test={int(test_sequences)}"
                ),
                flush=True,
            )
            reconstructed = QuantumDatasetBundle(
                train=DatasetSplit(
                    states=train_states,
                    params=train_params[: int(train_sequences)].contiguous(),
                    initial_state_codes=train_codes[: int(train_sequences)],
                    initial_state_family=train_family,
                    support_size=int(train_sequences) + int(test_sequences),
                ),
                test=DatasetSplit(
                    states=test_states,
                    params=test_params[: int(test_sequences)].contiguous(),
                    initial_state_codes=test_codes[: int(test_sequences)],
                    initial_state_family=train_family,
                    support_size=int(train_sequences) + int(test_sequences),
                ),
                hamiltonian=reference_chunk["hamiltonian"],
                basis_support_size=int(train_sequences) + int(test_sequences),
                used_support_fraction=1.0,
                initial_state_family_reason=reason,
            )

        broadcast_obj = [reconstructed]
        dist.broadcast_object_list(broadcast_obj, src=0)
        final_bundle = broadcast_obj[0]
        if not isinstance(final_bundle, QuantumDatasetBundle):
            raise RuntimeError("Broadcast dataset fallito: oggetto non valido.")
        return final_bundle

    if config.DATASET_SOURCE in {"haar_tfim", "haar_multi_hamiltonian"}:
        return generate_haar_tfim_dataset(
            train_sequences=train_sequences,
            test_sequences=test_sequences,
            n_qubits=n_qubits,
            num_states=num_states,
            seed=seed,
            enable_distributed=enable_distributed,
        )

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
        params=torch.tensor(
            [0.0, float(config.COUPLING_MEAN), float(config.FIELD_STRENGTH), 0.0, 0.0, 0.0],
            dtype=torch.float32,
        ).repeat(train_sequences, 1),
        initial_state_codes=initial_state_codes[:train_sequences],
        initial_state_family=family,
        support_size=support_size,
    )
    test_split = DatasetSplit(
        states=all_states[train_sequences:],
        params=torch.tensor(
            [0.0, float(config.COUPLING_MEAN), float(config.FIELD_STRENGTH), 0.0, 0.0, 0.0],
            dtype=torch.float32,
        ).repeat(test_sequences, 1),
        initial_state_codes=initial_state_codes[train_sequences:],
        initial_state_family=family,
        support_size=support_size,
    )

    _print_clamped_dataset_audit(
        "train",
        train_split.states,
        train_split.initial_state_codes,
        n_qubits,
        family,
    )
    _print_clamped_dataset_audit(
        "test",
        test_split.states,
        test_split.initial_state_codes,
        n_qubits,
        family,
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


def generate_haar_tfim_dataset(
    train_sequences: int = config.TRAIN_SEQUENCES,
    test_sequences: int = config.TEST_SEQUENCES,
    n_qubits: int = config.N_QUBITS,
    num_states: int = config.NUM_STATES,
    seed: int = config.SEED,
    *,
    enable_distributed: bool = True,
) -> QuantumDatasetBundle:
    if int(n_qubits) < 1:
        raise ValueError(f"n_qubits deve essere >= 1, ricevuto: {n_qubits}")
    if int(num_states) < 2:
        raise ValueError("Il setup richiesto usa almeno 2 stati per traiettoria.")

    total_sequences = int(train_sequences) + int(test_sequences)
    dim = 2 ** int(n_qubits)
    rng = random.Random(seed + 31)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 23)

    trajectories: list[torch.Tensor] = []
    params_list: list[torch.Tensor] = []
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(total_sequences):
        initial_state = sample_haar_random_states(
            num_samples=1,
            dim=dim,
            seed=int(torch.randint(0, 10_000_000, (1,), generator=generator).item()),
            dtype=torch.complex64,
        )[0]
        h_id, hamiltonian, params = hgen.sample_hamiltonian_instance(int(n_qubits), rng)
        unitary = torch.linalg.matrix_exp((-1j * float(config.TIME_STEP)) * hamiltonian)
        trajectory = torch.empty((num_states, dim), dtype=torch.complex64)
        current = initial_state
        trajectory[0] = current
        for step in range(1, num_states):
            current = unitary @ current
            current = current / torch.linalg.vector_norm(current).clamp(min=1e-8)
            trajectory[step] = current
        trajectories.append(trajectory)
        params_list.append(params.to(torch.float32))
        class_counts[h_id] += 1

    all_states = torch.stack(trajectories, dim=0).contiguous()
    params = torch.stack(params_list, dim=0).to(torch.float32).contiguous()
    initial_state_codes = list(range(total_sequences))
    class_names = {
        0: "TFIM",
        1: "XXZ",
        2: "Fermi-Hubbard",
        3: "Max-3-SAT",
    }
    active_class_names = [class_names[h_id] for h_id, count in class_counts.items() if count > 0]
    reason = (
        "stati iniziali Haar random da gaussiane complesse normalizzate; "
        "classi Hamiltoniane campionate per traiettoria tra quelle supportate "
        f"(active={active_class_names}), "
        "params=[H_ID,p1,p2,0,0,0], "
        f"n_qubits={int(n_qubits)}, num_states={int(num_states)}, "
        f"dt={float(config.TIME_STEP):.6g}, class_counts={class_counts}"
    )

    train_split = DatasetSplit(
        states=all_states[:train_sequences],
        params=params[:train_sequences].contiguous(),
        initial_state_codes=initial_state_codes[:train_sequences],
        initial_state_family="haar_random",
        support_size=total_sequences,
    )
    test_split = DatasetSplit(
        states=all_states[train_sequences:],
        params=params[train_sequences:].contiguous(),
        initial_state_codes=initial_state_codes[train_sequences:],
        initial_state_family="haar_random",
        support_size=total_sequences,
    )

    return QuantumDatasetBundle(
        train=train_split,
        test=test_split,
        hamiltonian=HamiltonianData(
            couplings=[],
            field_strength=float("nan"),
            backend="per_trajectory_matrix_exp",
            hamiltonian=torch.empty((0, 0), dtype=torch.complex64),
            evolution_operator=torch.empty((0, 0), dtype=torch.complex64),
        ),
        basis_support_size=total_sequences,
        used_support_fraction=1.0,
        initial_state_family_reason=reason,
    )
