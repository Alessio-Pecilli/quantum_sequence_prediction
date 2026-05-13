import json
import os
from pathlib import Path

import torch

from physical_ttn import PhysicalQubitTTNDecoder, PhysicalQubitTTNEncoder
from predictor import quantum_fidelity
from trainer import set_seed


def _random_normalized_complex(*shape: int) -> torch.Tensor:
    real = torch.randn(*shape, dtype=torch.float32)
    imag = torch.randn(*shape, dtype=torch.float32)
    states = torch.complex(real, imag)
    return states / torch.linalg.vector_norm(states, dim=-1, keepdim=True).clamp(min=1e-8)


def main():
    num_qubits = int(os.getenv("QSP_SANITY_N_QUBITS", "3"))
    d_model = int(os.getenv("QSP_SANITY_D_MODEL", "48"))
    bond_dim = int(os.getenv("QSP_SANITY_TTN_BOND_DIM", "16"))
    root_dim = int(os.getenv("QSP_SANITY_TTN_ROOT_DIM", str(bond_dim)))
    batch_size = int(os.getenv("QSP_SANITY_BATCH_SIZE", "64"))
    steps = int(os.getenv("QSP_SANITY_STEPS", "250"))
    learning_rate = float(os.getenv("QSP_SANITY_LR", "2e-3"))
    seed = int(os.getenv("QSP_SANITY_SEED", "7"))
    output_path = Path(os.getenv("QSP_SANITY_OUTPUT_JSON", "physical_ttn_autoencoder_sanity.json"))

    set_seed(seed)
    dim_2n = 2 ** num_qubits
    states = _random_normalized_complex(batch_size, dim_2n)
    encoder = PhysicalQubitTTNEncoder(
        num_qubits=num_qubits,
        d_model=d_model,
        bond_dim=bond_dim,
        root_dim=root_dim,
        use_bond_cap=True,
    )
    decoder = PhysicalQubitTTNDecoder(
        num_qubits=num_qubits,
        d_model=d_model,
        bond_dim=bond_dim,
        root_dim=root_dim,
        use_bond_cap=True,
    )
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
    )

    with torch.no_grad():
        initial_recon = decoder(encoder(states))
        initial_fidelity = float(quantum_fidelity(initial_recon, states).mean().item())

    history = []
    for step in range(steps):
        recon = decoder(encoder(states))
        fidelity = quantum_fidelity(recon, states)
        loss = torch.nn.functional.mse_loss(torch.view_as_real(recon), torch.view_as_real(states))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % max(1, steps // 10) == 0 or step == 0:
            history.append(
                {
                    "step": int(step + 1),
                    "loss": float(loss.item()),
                    "mean_fidelity": float(fidelity.mean().item()),
                }
            )

    with torch.no_grad():
        final_recon = decoder(encoder(states))
        final_fidelity = float(quantum_fidelity(final_recon, states).mean().item())

    summary = {
        "num_qubits": num_qubits,
        "dim_2n": dim_2n,
        "d_model": d_model,
        "bond_dim": bond_dim,
        "root_dim": root_dim,
        "batch_size": batch_size,
        "steps": steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "initial_mean_fidelity": initial_fidelity,
        "final_mean_fidelity": final_fidelity,
        "fidelity_improvement": final_fidelity - initial_fidelity,
        "history": history,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    assert final_fidelity > initial_fidelity + 0.05, (
        "Physical TTN autoencoder sanity failed: fidelity did not improve clearly. "
        f"initial={initial_fidelity:.6f}, final={final_fidelity:.6f}"
    )


if __name__ == "__main__":
    main()
