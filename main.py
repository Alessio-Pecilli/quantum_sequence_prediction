from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(1)

import config
from input import generate_fixed_tfim_dataset
from trainer import (
    TrainingHistory,
    build_model,
    evaluate_autoregressive,
    evaluate_teacher_forced,
    exposure_bias_detected,
    plot_training_curves,
    resolve_partial_warmup_steps,
    set_seed,
    train_model,
)


def _build_empty_resume_state() -> dict[str, Any]:
    return {
        "start_epoch": 1,
        "history": TrainingHistory(epochs=[], train_loss=[], train_fidelity=[]),
        "optimizer_state_dict": None,
        "scheduler_state_dict": None,
        "best_loss": None,
        "best_state": None,
        "resumed": False,
        "last_epoch": 0,
    }


def _try_resume_from_last_checkpoint(model) -> dict[str, Any]:
    if not config.AUTO_RESUME:
        return _build_empty_resume_state()
    if not config.LAST_CHECKPOINT_PATH.exists():
        return _build_empty_resume_state()

    try:
        payload = torch.load(config.LAST_CHECKPOINT_PATH, map_location=config.DEVICE)
        model.load_state_dict(payload["model_state_dict"])
        history_payload = payload.get("history", {})
        history = TrainingHistory(
            epochs=[int(value) for value in history_payload.get("epochs", [])],
            train_loss=[float(value) for value in history_payload.get("train_loss", [])],
            train_fidelity=[float(value) for value in history_payload.get("train_fidelity", [])],
        )
        last_epoch = int(payload.get("epoch", 0))
        return {
            "start_epoch": last_epoch + 1,
            "history": history,
            "optimizer_state_dict": payload.get("optimizer_state_dict"),
            "scheduler_state_dict": payload.get("scheduler_state_dict"),
            "best_loss": payload.get("best_loss"),
            "best_state": payload.get("best_state_dict"),
            "resumed": last_epoch > 0,
            "last_epoch": last_epoch,
        }
    except Exception as exc:
        print(
            "Resume automatico:     checkpoint non leggibile, riparto da epoca 1 "
            f"({type(exc).__name__})"
        )
        return _build_empty_resume_state()


def _as_serializable(result) -> dict[str, Any]:
    return {
        "loss": float(result.loss),
        "mean_fidelity": float(result.mean_fidelity),
        "fidelity_curve": [None if np.isnan(value) else float(value) for value in result.fidelity_curve],
        "coverage_curve": [float(value) for value in result.coverage_curve],
    }


def _plot_split_curves(
    ax,
    title: str,
    teacher_forced,
    autoregressive,
    partial_results: dict[int, Any],
):
    x = np.arange(1, len(teacher_forced.fidelity_curve) + 1)
    ax.plot(
        x,
        teacher_forced.fidelity_curve,
        label="Metodo 1: teacher forced",
        linewidth=2.4,
        color="#1f618d",
    )
    ax.plot(
        x,
        autoregressive.fidelity_curve,
        label="Metodo 2: rollout libero",
        linewidth=2.4,
        color="#b03a2e",
    )

    palette = ["#117a65", "#7d6608", "#6c3483", "#566573"]
    for color, (warmup_n1, result) in zip(palette, sorted(partial_results.items())):
        ax.plot(
            x,
            result.fidelity_curve,
            label=f"Metodo 3: warmup N1={warmup_n1}",
            linewidth=2.0,
            linestyle="--",
            color=color,
        )

    ax.set_title(title)
    ax.set_xlabel("Indice stato predetto")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)


def _flush_device_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    # Usa i valori di `config.py` (eventualmente sovrascrivibili via env vars).

    set_seed(config.SEED)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = generate_fixed_tfim_dataset()
    total_sequences = dataset.train.num_sequences + dataset.test.num_sequences

    print("=" * 78)
    print("Quantum Sequence Prediction | TFIM fixed-H pipeline")
    print("=" * 78)
    print(f"Device:                {config.DEVICE}")
    print(f"Qubits:                {config.N_QUBITS} (dim={config.DIM_2N})")
    print(f"Stati per traiettoria: {config.NUM_STATES} (predizioni={config.SEQ_LEN})")
    print(
        "Dataset:               "
        f"train={dataset.train.num_sequences}, test={dataset.test.num_sequences}, total={total_sequences}"
    )
    print(
        "Hamiltoniana:          "
        f"TFIM open chain, h_x={dataset.hamiltonian.field_strength:.3f}, "
        f"U=exp(i H dt), dt={config.TIME_STEP:.3f}, backend={dataset.hamiltonian.backend}"
    )
    print(
        "Coupling J_i:          "
        + ", ".join(f"{value:.4f}" for value in dataset.hamiltonian.couplings)
    )
    print(
        "Stati iniziali:        "
        f"{dataset.train.initial_state_family} | supporto basis=2^{config.N_QUBITS}={dataset.basis_support_size} | "
        f"uso totale/supporto basis={dataset.used_support_fraction:.3f}"
    )
    print(f"Motivo famiglia:       {dataset.initial_state_family_reason}")
    print(
        "Modello:               "
        f"d_model={config.D_MODEL}, heads={config.NUM_HEADS}, layers={config.NUM_LAYERS}, "
        f"ff={config.DIM_FEEDFORWARD}, dropout={config.DROPOUT}"
    )
    print(
        "Training:              "
        f"epochs={config.EPOCHS}, batch={config.BATCH_SIZE}, lr={config.LEARNING_RATE:.2e}, "
        f"wd={config.WEIGHT_DECAY:.2e}"
    )
    print(
        "Rollout robustness:    "
        f"ss_max_p={config.SCHEDULED_SAMPLING_MAX_PROB:.2f}, "
        f"ss_ramp_epochs={config.SCHEDULED_SAMPLING_RAMP_EPOCHS}, "
        f"aux_weight={config.ROLLOUT_AUX_WEIGHT:.2f}, "
        f"curriculum_epochs={config.ROLLOUT_CURRICULUM_EPOCHS}"
    )
    print("=" * 78)

    model = build_model()
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parametri modello:     {num_params:,}")
    resume_state = _try_resume_from_last_checkpoint(model)
    if resume_state["resumed"]:
        print(
            f"Resume automatico:     checkpoint trovato a epoca {resume_state['last_epoch']}, "
            f"riparto da epoca {resume_state['start_epoch']}"
        )
    else:
        print("Resume automatico:     nessun checkpoint valido, training da epoca 1")

    print("\n[1/4] Training")
    history = train_model(
        model,
        dataset.train.states,
        start_epoch=int(resume_state["start_epoch"]),
        history=resume_state["history"],
        optimizer_state_dict=resume_state["optimizer_state_dict"],
        scheduler_state_dict=resume_state["scheduler_state_dict"],
        best_loss=resume_state["best_loss"],
        best_state=resume_state["best_state"],
    )
    plot_training_curves(history)

    print("\n[2/4] Valutazione teacher forced")
    train_teacher = evaluate_teacher_forced(model, dataset.train.states)
    test_teacher = evaluate_teacher_forced(model, dataset.test.states)
    _flush_device_memory()

    print("\n[3/4] Valutazione autoregressiva")
    train_rollout = evaluate_autoregressive(model, dataset.train.states, warmup_states=1)
    test_rollout = evaluate_autoregressive(model, dataset.test.states, warmup_states=1)
    _flush_device_memory()

    train_exposure_bias = exposure_bias_detected(
        train_teacher.fidelity_curve,
        train_rollout.fidelity_curve,
    )
    test_exposure_bias = exposure_bias_detected(
        test_teacher.fidelity_curve,
        test_rollout.fidelity_curve,
    )
    add_partial_curves = train_exposure_bias or test_exposure_bias

    partial_results_train: dict[int, Any] = {}
    partial_results_test: dict[int, Any] = {}
    warmup_n1_values: list[int] = []
    if add_partial_curves:
        warmup_n1_values = resolve_partial_warmup_steps(config.SEQ_LEN)
        print(
            "\n[4/4] Exposure bias rilevato, aggiungo warmup parziali N1="
            + ", ".join(str(value) for value in warmup_n1_values)
        )
        for warmup_n1 in warmup_n1_values:
            warmup_states = warmup_n1 + 1
            partial_results_train[warmup_n1] = evaluate_autoregressive(
                model,
                dataset.train.states,
                warmup_states=warmup_states,
            )
            partial_results_test[warmup_n1] = evaluate_autoregressive(
                model,
                dataset.test.states,
                warmup_states=warmup_states,
            )
    else:
        print("\n[4/4] Nessun exposure bias marcato: mantengo solo metodo 1 e 2.")
    _flush_device_memory()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    _plot_split_curves(
        axes[0],
        "Train Set",
        train_teacher,
        train_rollout,
        partial_results_train,
    )
    _plot_split_curves(
        axes[1],
        "Test Set",
        test_teacher,
        test_rollout,
        partial_results_test,
    )
    fig.suptitle("Fidelity vero vs predetto in funzione del tempo", fontsize=14)
    fig.tight_layout()
    fig.savefig(config.FIDELITY_PLOT_PATH, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    print("\nMetriche aggregate")
    print(
        f"  Train | teacher={train_teacher.mean_fidelity:.6f} | rollout={train_rollout.mean_fidelity:.6f}"
    )
    print(
        f"  Test  | teacher={test_teacher.mean_fidelity:.6f} | rollout={test_rollout.mean_fidelity:.6f}"
    )

    summary = {
        "seed": int(config.SEED),
        "device": config.DEVICE,
        "config": {
            "N_QUBITS": int(config.N_QUBITS),
            "DIM_2N": int(config.DIM_2N),
            "NUM_STATES": int(config.NUM_STATES),
            "SEQ_LEN": int(config.SEQ_LEN),
            "TRAIN_SEQUENCES": int(config.TRAIN_SEQUENCES),
            "TEST_SEQUENCES": int(config.TEST_SEQUENCES),
            "COUPLING_MEAN": float(config.COUPLING_MEAN),
            "COUPLING_STD": float(config.COUPLING_STD),
            "FIELD_STRENGTH": float(config.FIELD_STRENGTH),
            "TIME_STEP": float(config.TIME_STEP),
            "SCHEDULED_SAMPLING_MAX_PROB": float(config.SCHEDULED_SAMPLING_MAX_PROB),
            "SCHEDULED_SAMPLING_RAMP_EPOCHS": int(config.SCHEDULED_SAMPLING_RAMP_EPOCHS),
            "ROLLOUT_AUX_WEIGHT": float(config.ROLLOUT_AUX_WEIGHT),
            "ROLLOUT_CURRICULUM_EPOCHS": int(config.ROLLOUT_CURRICULUM_EPOCHS),
            "AUTO_RESUME": bool(config.AUTO_RESUME),
            "CHECKPOINT_EVERY_EPOCH": int(config.CHECKPOINT_EVERY_EPOCH),
            "CHECKPOINT_EVERY_BATCH": int(config.CHECKPOINT_EVERY_BATCH),
            "EMPTY_CACHE_EVERY_EPOCH": bool(config.EMPTY_CACHE_EVERY_EPOCH),
            "INITIAL_STATE_FAMILY": config.INITIAL_STATE_FAMILY,
            "active_env_overrides": config.get_active_env_overrides(),
        },
        "dataset": {
            "basis_support_size": int(dataset.basis_support_size),
            "used_support_fraction": float(dataset.used_support_fraction),
            "initial_state_family": dataset.train.initial_state_family,
            "initial_state_family_reason": dataset.initial_state_family_reason,
            "train_initial_state_codes": dataset.train.initial_state_codes,
            "test_initial_state_codes": dataset.test.initial_state_codes,
        },
        "hamiltonian": {
            "couplings": dataset.hamiltonian.couplings,
            "field_strength": float(dataset.hamiltonian.field_strength),
            "backend": dataset.hamiltonian.backend,
        },
        "training_history": {
            "epochs": history.epochs,
            "train_loss": history.train_loss,
            "train_fidelity": history.train_fidelity,
        },
        "evaluation": {
            "train_teacher_forced": _as_serializable(train_teacher),
            "train_autoregressive": _as_serializable(train_rollout),
            "test_teacher_forced": _as_serializable(test_teacher),
            "test_autoregressive": _as_serializable(test_rollout),
            "exposure_bias": {
                "train": bool(train_exposure_bias),
                "test": bool(test_exposure_bias),
            },
            "partial_warmup_n1_values": warmup_n1_values,
            "train_partial_warmups": {
                str(key): _as_serializable(value) for key, value in partial_results_train.items()
            },
            "test_partial_warmups": {
                str(key): _as_serializable(value) for key, value in partial_results_test.items()
            },
        },
        "artifacts": {
            "fidelity_plot": str(config.FIDELITY_PLOT_PATH),
            "training_curves_plot": str(config.TRAINING_CURVES_PATH),
            "last_checkpoint": str(config.LAST_CHECKPOINT_PATH),
            "checkpoint": str(config.CHECKPOINT_PATH),
        },
    }

    with config.SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nPlot fidelity:         {config.FIDELITY_PLOT_PATH}")
    print(f"Plot training:         {config.TRAINING_CURVES_PATH}")
    print(f"Summary JSON:          {config.SUMMARY_PATH}")
    if config.SAVE_MODEL:
        print(f"Checkpoint:            {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
