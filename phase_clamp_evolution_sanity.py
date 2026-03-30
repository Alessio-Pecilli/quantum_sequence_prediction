import math
import os
import random

import torch

import config
from input import (
    build_tfim_hamiltonian,
    compute_evolution_operator,
    sample_couplings,
)


def clamp_global_phase_first_amplitude(psi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Clamping gauge: impone che psi[0] abbia fase zero (quindi psi[0] sia reale >= 0).
    Se |psi[0]| e' quasi zero, non clampa (evita divisione numericamente instabile).
    """
    if psi.ndim != 1:
        raise ValueError(f"psi deve essere 1D (dim,), ricevuto shape={tuple(psi.shape)}")
    a0 = psi[0]
    mag = torch.abs(a0)
    if float(mag) < eps:
        return psi
    unit = a0 / mag  # exp(i*theta)
    return psi * torch.conj(unit)  # exp(-i*theta)


def format_complex(z: complex, ndigits: int = 6) -> str:
    return f"{z.real:+.{ndigits}f}{z.imag:+.{ndigits}f}j"


def tensor_product_x_basis(bits: list[int]) -> torch.Tensor:
    """
    Codifica in base X:
      bit=0 -> |+> = (|0>+|1>)/sqrt(2)
      bit=1 -> |-> = (|0>-|1>)/sqrt(2)
    """
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    plus = torch.tensor([sqrt2_inv, sqrt2_inv], dtype=torch.complex64)
    minus = torch.tensor([sqrt2_inv, -sqrt2_inv], dtype=torch.complex64)

    state = plus if bits[0] == 0 else minus
    for b in bits[1:]:
        state = torch.kron(state, plus if b == 0 else minus)
    return state.to(torch.complex64)


def evolve_with_step_clamp(
    psi0: torch.Tensor,
    evolution_operator: torch.Tensor,
    steps: int,
    eps_norm: float = 1e-8,
    eps_phase: float = 1e-7,
) -> None:
    psi = psi0

    print("\nInitial state (t=0) after clamping:")
    psi = clamp_global_phase_first_amplitude(psi)
    a0 = psi[0]
    phase0 = torch.angle(a0).item()
    print(f"psi[0] = {format_complex(a0)}; |psi[0]|={abs(a0.item()):.6f}; angle={phase0:.6f} rad")

    for idx in range(psi.numel()):
        print(f"  {idx:>2d}: {format_complex(psi[idx].item())}")

    for t in range(1, steps + 1):
        # Apply U for the next time.
        psi = torch.matmul(evolution_operator, psi)

        # Numerical safety: renormalize (should be ~unitary).
        psi = psi / torch.linalg.vector_norm(psi).clamp(min=eps_norm)

        psi_pre = psi
        psi = clamp_global_phase_first_amplitude(psi)
        a0_pre = psi_pre[0]
        a0_post = psi[0]

        imag_post = float(torch.imag(a0_post).abs().item())
        real_post = float(torch.real(a0_post).item())
        ok = imag_post <= eps_phase and real_post >= -eps_phase

        phase_pre = torch.angle(a0_pre).item()
        phase_post = torch.angle(a0_post).item()

        print(f"\nAfter applying U (t={t} before clamping): psi[0]={format_complex(a0_pre)}; angle={phase_pre:.6f} rad")
        print(f"After clamping (t={t}): psi[0]={format_complex(a0_post)}; angle={phase_post:.6f} rad; ok={ok}")

        if not ok:
            print(f"  WARNING: clamp failed tolerance: |Im(psi[0])|={imag_post:.3e}, Re(psi[0])={real_post:.3e}")

        print(f"All coefficients at t={t}:")
        for idx in range(psi.numel()):
            print(f"  {idx:>2d}: {format_complex(psi[idx].item())}")


def main() -> None:
    # Dim "low" for a quick sanity check.
    # Uses config.N_QUBITS but caps it to keep the coefficient printout readable.
    cap = int(os.getenv("QSP_TEST_MAX_N_QUBITS", "3"))
    n_qubits = int(min(int(config.N_QUBITS), cap))

    seed = int(os.getenv("QSP_TEST_SEED", str(config.SEED)))
    steps = int(os.getenv("QSP_TEST_STEPS", "5"))

    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    dim = 2**n_qubits

    rng = random.Random(seed)
    bits = [rng.randint(0, 1) for _ in range(n_qubits)]
    plus_minus = ["+" if b == 0 else "-" for b in bits]

    print(f"Random bitstring (0/1), n_qubits={n_qubits}: {bits}")
    print(f"Encoding in X basis: {plus_minus} (0->|+>, 1->|->)")
    print(f"State dimension: dim={dim}")

    psi0 = tensor_product_x_basis(bits)
    psi0 = psi0 / torch.linalg.vector_norm(psi0).clamp(min=1e-8)

    # Build the TFIM Hamiltonian and evolution operator U = exp(i H dt).
    couplings = sample_couplings(n_qubits, seed + 11)
    hamiltonian = build_tfim_hamiltonian(
        n_qubits=n_qubits,
        couplings=couplings,
        field_strength=config.FIELD_STRENGTH,
    )
    evolution_operator, backend = compute_evolution_operator(hamiltonian, config.TIME_STEP)

    print(f"\nHamiltonian backend for U: {backend}")
    print(f"TIME_STEP={config.TIME_STEP}")
    print("Starting evolution with step-by-step global-phase clamping...")

    evolve_with_step_clamp(
        psi0=psi0,
        evolution_operator=evolution_operator,
        steps=steps,
    )


if __name__ == "__main__":
    main()

