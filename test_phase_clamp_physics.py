import math

import torch

import config
from input import generate_fixed_tfim_dataset
from predictor import QuantumSequencePredictor, clamp_global_phase, quantum_fidelity
from trainer import set_seed


def _angle_wrap(x: torch.Tensor) -> torch.Tensor:
    """Map angles to (-pi, pi]."""
    return (x + math.pi) % (2 * math.pi) - math.pi


def _assert_allclose(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float, msg: str):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg} (max|diff|={diff:.3e}, atol={atol}, rtol={rtol})")


@torch.no_grad()
def assert_phase_clamp_physics(
    states: torch.Tensor,
    *,
    eps: float = 1e-8,
    atol: float = 5e-5,
    verbose: bool = False,
    label: str = "",
):
    """
    Suite di check "fisicamente naturali" per il gauge fixing della fase globale.

    Per ogni stato ψ (complesso):
      - clamp_global_phase moltiplica ψ per una fase globale unitaria => norma invariata.
      - la componente 0 diventa reale e >= 0 (quando |ψ0| > eps).
      - i rapporti gauge-invariant r_k = ψ_k / ψ_0 sono preservati:
          ψ'_k / ψ'_0 == ψ_k / ψ_0   (quando |ψ0| > eps).
      - la mappa è idempotente: clamp(clamp(ψ)) == clamp(ψ).
    """
    if not torch.is_complex(states):
        raise AssertionError(f"attesi stati complessi, dtype={states.dtype}")

    clamped = clamp_global_phase(states, eps=eps)

    # (1) Norma invariata (a precisione numerica).
    n0 = torch.linalg.vector_norm(states, dim=-1)
    n1 = torch.linalg.vector_norm(clamped, dim=-1)
    if verbose:
        max_abs_norm_diff = float((n0 - n1).abs().max().item())
        max_rel_norm_diff = float(((n0 - n1).abs() / n0.clamp(min=eps)).max().item())
        print(f"[{label}] (1) norm invariance:")
        print(f"  max abs diff in ||psi|| = {max_abs_norm_diff:.3e}")
        print(f"  max rel diff in ||psi|| = {max_rel_norm_diff:.3e}")
    _assert_allclose(n0, n1, atol=5e-6, rtol=5e-6, msg="Norma non invariata sotto fase globale")

    # (2) c0 reale e >=0 quando significativo.
    c0 = clamped[..., 0]
    abs0 = torch.abs(c0)
    valid = abs0 > eps
    if verbose:
        valid_count = int(valid.sum().item())
        total_count = int(valid.numel())
        print(f"[{label}] (2) gauge fixing on psi0:")
        print(f"  valid (|psi0|>eps): {valid_count}/{total_count} (eps={eps:g})")
        if valid.any():
            max_im = float(c0.imag[valid].abs().max().item())
            min_re = float(c0.real[valid].min().item())
            print(f"  max |Im(psi0')| = {max_im:.3e} (atol={atol:g})")
            print(f"  min  Re(psi0')  = {min_re:.3e} (should be >= 0)")
    if valid.any():
        if not (c0.imag[valid].abs() <= atol).all().item():
            worst = float(c0.imag[valid].abs().max().item())
            raise AssertionError(f"Im(ψ0) non ~0 dopo clamping (max={worst:.3e})")
        if not (c0.real[valid] >= -atol).all().item():
            worst = float(c0.real[valid].min().item())
            raise AssertionError(f"Re(ψ0) < 0 dopo clamping (min={worst:.3e})")

    # (3) Rapporti gauge-invariant preservati: ψ_k/ψ_0 invarianti.
    # Usiamo solo campioni validi per evitare divisioni instabili.
    psi0 = states[..., 0]
    psi0p = clamped[..., 0]
    if valid.any():
        ratios = states[valid] / psi0[valid].unsqueeze(-1)
        ratios_p = clamped[valid] / psi0p[valid].unsqueeze(-1)
        if verbose:
            max_abs_ratio_diff = float((ratios - ratios_p).abs().max().item())
            max_rel_ratio_diff = float(
                ((ratios - ratios_p).abs() / ratios.abs().clamp(min=eps)).max().item()
            )
            print(f"[{label}] (3) gauge-invariant ratios r_k=psi_k/psi_0:")
            print(f"  max abs diff in r = {max_abs_ratio_diff:.3e}")
            print(f"  max rel diff in r = {max_rel_ratio_diff:.3e}")
        _assert_allclose(ratios, ratios_p, atol=2e-4, rtol=2e-4, msg="Rapporti ψ/ψ0 non preservati")
    elif verbose:
        print(f"[{label}] (3) gauge-invariant ratios: skipped (no valid samples)")

    # (4) Idempotenza.
    clamped2 = clamp_global_phase(clamped, eps=eps)
    if verbose:
        max_abs_idem = float((clamped - clamped2).abs().max().item())
        print(f"[{label}] (4) idempotence:")
        print(f"  max |clamp(clamp(psi)) - clamp(psi)| = {max_abs_idem:.3e}")
    _assert_allclose(clamped, clamped2, atol=2e-5, rtol=2e-5, msg="Clamping non idempotente")

    # (5) Controllo “angolare” esplicito: arg(ψ'_0) ~ 0 dove valido.
    if valid.any():
        phase0 = torch.angle(c0[valid])
        phase0 = _angle_wrap(phase0)
        if verbose:
            max_abs_phase0 = float(phase0.abs().max().item())
            print(f"[{label}] (5) angle check on psi0':")
            print(f"  max |arg(psi0')| = {max_abs_phase0:.3e} rad")
        if not (phase0.abs() <= 5e-4).all().item():
            worst = float(phase0.abs().max().item())
            raise AssertionError(f"arg(ψ0) non ~0 dopo clamping (max={worst:.3e})")
    elif verbose:
        print(f"[{label}] (5) angle check: skipped (no valid samples)")


@torch.no_grad()
def assert_fidelity_invariant_under_clamp(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    atol: float = 5e-6,
    verbose: bool = False,
    label: str = "",
):
    """
    Fidelity deve essere invariante sotto clamping indipendente:
      F(a,b) == F(clamp(a), clamp(b))
    """
    f0 = quantum_fidelity(a, b)
    f1 = quantum_fidelity(clamp_global_phase(a), clamp_global_phase(b))
    if verbose:
        max_abs = float((f0 - f1).abs().max().item())
        mean_abs = float((f0 - f1).abs().mean().item())
        print(f"[{label}] (6) fidelity invariance:")
        print(f"  max abs diff in F = {max_abs:.3e} (atol={atol:g})")
        print(f"  mean abs diff in F = {mean_abs:.3e}")
    _assert_allclose(f0, f1, atol=atol, rtol=0.0, msg="Fidelity non invariante sotto clamping")


def main():
    set_seed(config.SEED)
    bundle = generate_fixed_tfim_dataset()

    # --- Check sui dati (training + test) ---
    verbose = True

    assert_phase_clamp_physics(bundle.train.states, verbose=verbose, label="train.states")
    assert_phase_clamp_physics(bundle.test.states, verbose=verbose, label="test.states")
    assert_fidelity_invariant_under_clamp(
        bundle.train.targets[:8],
        bundle.train.targets[:8],
        verbose=verbose,
        label="train.targets self",
    )

    # --- Check sulle predizioni del modello addestrato ---
    model = QuantumSequencePredictor().to(config.DEVICE)
    checkpoint_path = config.CHECKPOINT_PATH
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()

    x = bundle.train.inputs[:8].to(config.DEVICE)
    params = bundle.train.params[:8].to(config.DEVICE)
    pred = model(x, params).cpu()
    assert_phase_clamp_physics(pred, verbose=verbose, label="model pred (train.inputs[:8])")

    # Invarianza della fidelity tra pred e target sotto clamping (dovrebbe valere sempre).
    y = bundle.train.targets[:8]
    assert_fidelity_invariant_under_clamp(pred, y, verbose=verbose, label="pred vs target (train[:8])")

    print("PHASE CLAMP PHYSICS AUDIT OK")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  output_parametrization: {config.OUTPUT_PARAMETRIZATION}")


if __name__ == "__main__":
    main()

