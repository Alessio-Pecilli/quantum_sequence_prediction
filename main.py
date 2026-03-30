from __future__ import annotations

import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(1)

import config
from input import generate_fixed_tfim_dataset
from trainer import (
    build_model,
    evaluate_autoregressive,
    evaluate_teacher_forced,
    exposure_bias_detected,
    plot_training_curves,
    resolve_partial_warmup_steps,
    set_seed,
    train_model,
)


def _as_serializable(result) -> dict[str, object]:
    return {
        "loss": float(result.loss),
        "mean_fidelity": float(result.mean_fidelity),
        "fidelity_curve": [None if np.isnan(v) else float(v) for v in result.fidelity_curve],
        "coverage_curve": [float(v) for v in result.coverage_curve],
    }


def _plot_split_curves(ax, title: str, teacher_forced, autoregressive, partial_results: dict[int, object]):
    x = np.arange(1, len(teacher_forced.fidelity_curve) + 1)
    ax.plot(x, teacher_forced.fidelity_curve, label="Metodo 1: teacher forced", linewidth=2.3, color="#1f618d")
    ax.plot(x, autoregressive.fidelity_curve, label="Metodo 2: rollout libero", linewidth=2.3, color="#b03a2e")
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


def main():
    set_seed(config.SEED)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = generate_fixed_tfim_dataset()

    print("=" * 78)
    print("Quantum Sequence Prediction | X-basis clamped pipeline")
    print("=" * 78)
    print(f"Device:                {config.DEVICE}")
    print(f"Qubits:                {config.N_QUBITS} (dim={config.DIM_2N})")
    print(f"Stati per traiettoria: {config.NUM_STATES} (predizioni={config.SEQ_LEN})")
    print(f"Dataset:               train={dataset.train.num_sequences}, test={dataset.test.num_sequences}")
    print(f"Stati iniziali:        {dataset.train.initial_state_family}")
    print(f"Motivo famiglia:       {dataset.initial_state_family_reason}")
    print("=" * 78)

    model = build_model()
    history = train_model(model, dataset.train.states)
    plot_training_curves(history)

    train_teacher = evaluate_teacher_forced(model, dataset.train.states)
    test_teacher = evaluate_teacher_forced(model, dataset.test.states)
    rollout_warmup = int(config.ROLLOUT_WARMUP_STATES)
    train_rollout = evaluate_autoregressive(model, dataset.train.states, warmup_states=rollout_warmup)
    test_rollout = evaluate_autoregressive(model, dataset.test.states, warmup_states=rollout_warmup)

    add_partial_curves = exposure_bias_detected(train_teacher.fidelity_curve, train_rollout.fidelity_curve) or (
        exposure_bias_detected(test_teacher.fidelity_curve, test_rollout.fidelity_curve)
    )
    partial_results_train: dict[int, object] = {}
    partial_results_test: dict[int, object] = {}
    warmup_n1_values: list[int] = []
    if add_partial_curves:
        warmup_n1_values = resolve_partial_warmup_steps(config.SEQ_LEN)
        print("\nExposure bias rilevato: aggiungo curve metodo 3 per N1=" + ", ".join(str(v) for v in warmup_n1_values))
        for warmup_n1 in warmup_n1_values:
            warmup_states = warmup_n1 + 1
            partial_results_train[warmup_n1] = evaluate_autoregressive(
                model, dataset.train.states, warmup_states=warmup_states
            )
            partial_results_test[warmup_n1] = evaluate_autoregressive(
                model, dataset.test.states, warmup_states=warmup_states
            )
    else:
        print("\nNessun exposure bias marcato: mantengo solo metodo 1 e 2.")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.3), sharey=True)
    _plot_split_curves(axes[0], "Train Set", train_teacher, train_rollout, partial_results_train)
    _plot_split_curves(axes[1], "Test Set", test_teacher, test_rollout, partial_results_test)
    fig.suptitle("Fidelity vero vs predetto nel tempo", fontsize=14)
    fig.tight_layout()
    fig.savefig(config.FIDELITY_PLOT_PATH, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

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
            "INITIAL_STATE_FAMILY": config.INITIAL_STATE_FAMILY,
            "ROLLOUT_WARMUP_STATES": int(config.ROLLOUT_WARMUP_STATES),
            "PARTIAL_WARMUP_STEPS": config.PARTIAL_WARMUP_STEPS,
            "CLAMP_AUDIT_PRINT": bool(config.CLAMP_AUDIT_PRINT),
            "CLAMP_AUDIT_MAX_SEQUENCES": int(config.CLAMP_AUDIT_MAX_SEQUENCES),
            "CLAMP_AUDIT_MAX_STATES": int(config.CLAMP_AUDIT_MAX_STATES),
            "active_env_overrides": config.get_active_env_overrides(),
        },
        "dataset": {
            "initial_state_family": dataset.train.initial_state_family,
            "initial_state_family_reason": dataset.initial_state_family_reason,
            "train_initial_state_codes": dataset.train.initial_state_codes,
            "test_initial_state_codes": dataset.test.initial_state_codes,
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
            "partial_warmup_n1_values": warmup_n1_values,
            "train_partial_warmups": {str(k): _as_serializable(v) for k, v in partial_results_train.items()},
            "test_partial_warmups": {str(k): _as_serializable(v) for k, v in partial_results_test.items()},
        },
        "artifacts": {
            "fidelity_plot": str(config.FIDELITY_PLOT_PATH),
            "training_curves_plot": str(config.TRAINING_CURVES_PATH),
            "summary_json": str(config.SUMMARY_PATH),
        },
    }
    with config.SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nMetriche aggregate")
    print(f"  Train | teacher={train_teacher.mean_fidelity:.6f} | rollout={train_rollout.mean_fidelity:.6f}")
    print(f"  Test  | teacher={test_teacher.mean_fidelity:.6f} | rollout={test_rollout.mean_fidelity:.6f}")
    print(f"\nPlot fidelity:  {config.FIDELITY_PLOT_PATH}")
    print(f"Plot training:  {config.TRAINING_CURVES_PATH}")
    print(f"Summary JSON:   {config.SUMMARY_PATH}")


if __name__ == "__main__":
    main()
