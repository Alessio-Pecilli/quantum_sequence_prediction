from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BACKENDS = ("dense_legacy", "flat_ttn", "physical_ttn")


def _flatten_stats(values):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }
    count = len(ordered)

    def quantile(q: float) -> float:
        if count == 1:
            return ordered[0]
        index = q * (count - 1)
        low = int(index)
        high = min(low + 1, count - 1)
        alpha = index - low
        return ordered[low] * (1.0 - alpha) + ordered[high] * alpha

    return {
        "mean": sum(ordered) / count,
        "median": quantile(0.5),
        "p10": quantile(0.1),
        "p90": quantile(0.9),
    }


def _run_backend_child() -> None:
    import torch

    import config
    from embedding import ComplexEmbedding, FlatCoefficientTTNEncoder
    from input import generate_fixed_tfim_dataset
    from physical_ttn import PhysicalQubitTTNDecoder, PhysicalQubitTTNEncoder
    from predictor import DenseLegacyDecoder, FlatCoefficientTTNDecoder, quantum_fidelity
    from trainer import build_model, evaluate_teacher_forced, set_seed, train_model

    backend = str(config.EMBEDDING_BACKEND)
    output_path = Path(os.environ["QSP_ABLATION_CHILD_OUTPUT"])
    set_seed(config.SEED)
    dataset = generate_fixed_tfim_dataset()
    model = build_model()
    parameter_count = int(sum(param.numel() for param in model.parameters() if param.requires_grad))

    train_start = time.perf_counter()
    history, _, _ = train_model(
        model,
        dataset.train.states,
        dataset.train.params,
        validation_states=dataset.test.states,
        validation_params=dataset.test.params,
    )
    train_elapsed = time.perf_counter() - train_start

    model.eval()
    with torch.no_grad():
        train_pred = model(
            dataset.train.inputs.to(config.DEVICE),
            dataset.train.params.to(config.DEVICE),
        ).cpu()
        val_pred = model(
            dataset.test.inputs.to(config.DEVICE),
            dataset.test.params.to(config.DEVICE),
        ).cpu()
        train_fidelity_matrix = quantum_fidelity(train_pred, dataset.train.targets.cpu())
        val_fidelity_matrix = quantum_fidelity(val_pred, dataset.test.targets.cpu())

    train_teacher = evaluate_teacher_forced(model, dataset.train.states, dataset.train.params)
    val_teacher = evaluate_teacher_forced(model, dataset.test.states, dataset.test.params)

    auto_states = dataset.train.states[: min(32, int(dataset.train.states.shape[0])), 0, :].clone()
    if backend == "dense_legacy":
        encoder = ComplexEmbedding(dim_2n=config.DIM_2N, d_model=config.D_MODEL)
        decoder = DenseLegacyDecoder(dim_2n=config.DIM_2N, d_model=config.D_MODEL)
    elif backend == "flat_ttn":
        encoder = FlatCoefficientTTNEncoder(
            num_qubits=config.N_QUBITS,
            latent_dim=config.TTN_LATENT_DIM,
            d_model=config.D_MODEL,
        )
        decoder = FlatCoefficientTTNDecoder(
            num_qubits=config.N_QUBITS,
            latent_dim=config.TTN_LATENT_DIM,
            d_model=config.D_MODEL,
        )
    elif backend == "physical_ttn":
        encoder = PhysicalQubitTTNEncoder(
            num_qubits=config.N_QUBITS,
            d_model=config.D_MODEL,
            bond_dim=config.TTN_BOND_DIM,
            root_dim=config.TTN_ROOT_DIM,
            use_bond_cap=config.TTN_USE_BOND_CAP,
            pairing=config.TTN_TREE_PAIRING,
        )
        decoder = PhysicalQubitTTNDecoder(
            num_qubits=config.N_QUBITS,
            d_model=config.D_MODEL,
            bond_dim=config.TTN_BOND_DIM,
            root_dim=config.TTN_ROOT_DIM,
            use_bond_cap=config.TTN_USE_BOND_CAP,
            pairing=config.TTN_TREE_PAIRING,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    auto_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=float(config.LEARNING_RATE),
    )
    auto_steps = int(os.getenv("QSP_ABLATION_AUTO_STEPS", "80"))
    with torch.no_grad():
        auto_init = float(quantum_fidelity(decoder(encoder(auto_states)), auto_states).mean().item())
    for _ in range(auto_steps):
        recon = decoder(encoder(auto_states))
        auto_loss = torch.nn.functional.mse_loss(torch.view_as_real(recon), torch.view_as_real(auto_states))
        auto_optimizer.zero_grad(set_to_none=True)
        auto_loss.backward()
        auto_optimizer.step()
    with torch.no_grad():
        auto_final = float(quantum_fidelity(decoder(encoder(auto_states)), auto_states).mean().item())

    payload = {
        "backend": backend,
        "num_parameters": parameter_count,
        "epochs": int(config.EPOCHS),
        "epoch_time_seconds": float(train_elapsed / max(1, int(config.EPOCHS))),
        "train_loss": float(history.train_loss[-1]) if history.train_loss else float("nan"),
        "validation_loss": float(val_teacher.loss),
        "train_fidelity_mean": float(train_fidelity_matrix.mean().item()),
        "validation_fidelity_mean": float(val_fidelity_matrix.mean().item()),
        "validation_fidelity_median": float(torch.quantile(val_fidelity_matrix.flatten(), 0.5).item()),
        "validation_fidelity_p10": float(torch.quantile(val_fidelity_matrix.flatten(), 0.1).item()),
        "validation_fidelity_p90": float(torch.quantile(val_fidelity_matrix.flatten(), 0.9).item()),
        "train_teacher_forced_mean": float(train_teacher.mean_fidelity),
        "validation_teacher_forced_mean": float(val_teacher.mean_fidelity),
        "reconstruction_fidelity_autoencoder_initial": auto_init,
        "reconstruction_fidelity_autoencoder_final": auto_final,
        "reconstruction_fidelity_autoencoder_gain": auto_final - auto_init,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_parent() -> None:
    seed = os.getenv("QSP_SEED", "7")
    num_qubits = os.getenv("QSP_N_QUBITS", "4")
    epochs = os.getenv("QSP_ABLATION_EPOCHS", os.getenv("QSP_EPOCHS", "3"))
    train_sequences = os.getenv("QSP_ABLATION_TRAIN_SEQUENCES", os.getenv("QSP_TRAIN_SEQUENCES", "48"))
    test_sequences = os.getenv("QSP_ABLATION_TEST_SEQUENCES", os.getenv("QSP_TEST_SEQUENCES", "24"))
    num_states = os.getenv("QSP_ABLATION_NUM_STATES", os.getenv("QSP_NUM_STATES", "9"))
    output_path = Path(os.getenv("QSP_ABLATION_OUTPUT_JSON", "ablation_ttn_backends.json"))

    runs = []
    for backend in BACKENDS:
        child_output = REPO_ROOT / f"ablation_{backend}.json"
        env = os.environ.copy()
        env.update(
            {
                "QSP_SEED": seed,
                "QSP_N_QUBITS": num_qubits,
                "QSP_NUM_STATES": num_states,
                "QSP_TRAIN_SEQUENCES": train_sequences,
                "QSP_TEST_SEQUENCES": test_sequences,
                "QSP_EPOCHS": epochs,
                "QSP_EMBEDDING_BACKEND": backend,
                "QSP_RESULTS_DIR_NAME": f"results_ablation_{backend}",
                "QSP_AUTO_RESUME": "0",
                "QSP_SAVE_MODEL": "0",
                "QSP_TRAIN_DIAGNOSTICS": "0",
                "QSP_MULTISTEP_TRAIN_VERBOSE": "0",
                "QSP_ABLATION_CHILD_OUTPUT": str(child_output),
            }
        )
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "ablation_ttn_backends.py"), "--child"],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        if completed.stdout.strip():
            print(completed.stdout.strip())
        if completed.stderr.strip():
            print(completed.stderr.strip())
        runs.append(json.loads(child_output.read_text(encoding="utf-8")))

    summary = {
        "seed": int(seed),
        "num_qubits": int(num_qubits),
        "num_states": int(num_states),
        "train_sequences": int(train_sequences),
        "test_sequences": int(test_sequences),
        "epochs": int(epochs),
        "results": runs,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for run in runs:
        print(
            f"{run['backend']:13s} | "
            f"train_fid={run['train_fidelity_mean']:.6f} | "
            f"val_mean={run['validation_fidelity_mean']:.6f} | "
            f"val_med={run['validation_fidelity_median']:.6f} | "
            f"val_p10={run['validation_fidelity_p10']:.6f} | "
            f"val_p90={run['validation_fidelity_p90']:.6f} | "
            f"train_loss={run['train_loss']:.6f} | "
            f"val_loss={run['validation_loss']:.6f} | "
            f"epoch_s={run['epoch_time_seconds']:.6f} | "
            f"params={run['num_parameters']} | "
            f"ae_fid={run['reconstruction_fidelity_autoencoder_final']:.6f}"
        )
    print(f"Saved ablation summary to {output_path}")


if __name__ == "__main__":
    if "--child" in sys.argv:
        _run_backend_child()
    else:
        _run_parent()
