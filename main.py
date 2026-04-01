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
    AdaptiveTrainingTrace,
    TrainingHistory,
    build_model,
    compute_observable_curves,
    evaluate_autoregressive,
    evaluate_multistep,
    exposure_bias_detected,
    plot_observable_curves,
    plot_training_curves,
    resolve_partial_warmup_steps,
    set_seed,
    try_resume_from_last_checkpoint,
    train_model,
)


def _as_serializable(result) -> dict[str, object]:
    return {
        "loss": float(result.loss),
        "mean_fidelity": float(result.mean_fidelity),
        "fidelity_curve": [None if np.isnan(v) else float(v) for v in result.fidelity_curve],
        "coverage_curve": [float(v) for v in result.coverage_curve],
    }


def _history_as_serializable(history: TrainingHistory) -> dict[str, object]:
    return {
        "epochs": [int(epoch) for epoch in history.epochs],
        "train_loss": [float(value) for value in history.train_loss],
        "train_fidelity": [float(value) for value in history.train_fidelity],
    }


def _observable_curves_as_serializable(curves) -> dict[str, object]:
    return {
        "time_indices": [int(v) for v in curves.time_indices.tolist()],
        "physical_time": [float(v) for v in curves.physical_time.tolist()],
        "mz_exact": [float(v) for v in curves.mz_exact.tolist()],
        "mz_multistep": [float(v) for v in curves.mz_multistep.tolist()],
        "mz_rollout": [float(v) for v in curves.mz_rollout.tolist()],
        "mx_exact": [float(v) for v in curves.mx_exact.tolist()],
        "mx_multistep": [float(v) for v in curves.mx_multistep.tolist()],
        "mx_rollout": [float(v) for v in curves.mx_rollout.tolist()],
        "cz_exact": [float(v) for v in curves.cz_exact.tolist()],
        "cz_multistep": [float(v) for v in curves.cz_multistep.tolist()],
        "cz_rollout": [float(v) for v in curves.cz_rollout.tolist()],
    }


def _adaptive_training_as_serializable(trace: AdaptiveTrainingTrace) -> dict[str, object]:
    return {
        "enabled": bool(trace.enabled),
        "initial_horizon": int(trace.initial_horizon),
        "initial_teacher_steps": int(trace.initial_teacher_steps),
        "final_horizon": int(trace.final_horizon),
        "final_teacher_steps": int(trace.final_teacher_steps),
        "epoch_summaries": [
            {
                "epoch": int(summary.epoch),
                "horizon": int(summary.horizon),
                "teacher_steps": int(summary.teacher_steps),
                "head_loss": float(summary.head_loss),
                "tail_loss": float(summary.tail_loss),
                "head_fidelity": float(summary.head_fidelity),
                "tail_fidelity": float(summary.tail_fidelity),
                "mean_offset_losses": [float(v) for v in summary.mean_offset_losses],
                "mean_offset_fidelities": [float(v) for v in summary.mean_offset_fidelities],
                "mean_offset_weights": [float(v) for v in summary.mean_offset_weights],
            }
            for summary in trace.epoch_summaries
        ],
    }


def _load_history_from_last_checkpoint() -> TrainingHistory:
    if not config.LAST_CHECKPOINT_PATH.exists():
        return TrainingHistory(epochs=[], train_loss=[], train_fidelity=[])

    payload = torch.load(config.LAST_CHECKPOINT_PATH, map_location="cpu")
    history = payload.get("history", {})
    return TrainingHistory(
        epochs=[int(epoch) for epoch in history.get("epochs", [])],
        train_loss=[float(value) for value in history.get("train_loss", [])],
        train_fidelity=[float(value) for value in history.get("train_fidelity", [])],
    )


def _load_trained_model(model):
    if not config.CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint non trovato: {config.CHECKPOINT_PATH}. "
            "Disattiva QSP_EVAL_ONLY oppure genera prima best_model.pt."
        )
    state_dict = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return _load_history_from_last_checkpoint()


def _plot_split_curves(ax, title: str, multistep, autoregressive, partial_results: dict[int, object]):
    x = np.arange(1, len(multistep.fidelity_curve) + 1)
    ax.plot(x, multistep.fidelity_curve, label="Metodo 1: multi-step", linewidth=2.3, color="#117a65")
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
    resume_status = {
        "enabled": bool(config.AUTO_RESUME),
        "resumed": False,
        "reason": "eval-only attivo" if config.EVAL_ONLY else "auto-resume disattivato",
    }

    print("=" * 78)
    print("Quantum Sequence Prediction | X-basis clamped pipeline")
    print("=" * 78)
    print(f"Device:                {config.DEVICE}")
    print(f"Qubits:                {config.N_QUBITS} (dim={config.DIM_2N})")
    print(f"Stati per traiettoria: {config.NUM_STATES} (predizioni={config.SEQ_LEN})")
    print(f"Dataset:               train={dataset.train.num_sequences}, test={dataset.test.num_sequences}")
    print(f"Stati iniziali:        {dataset.train.initial_state_family}")
    print(f"Motivo famiglia:       {dataset.initial_state_family_reason}")
    print(
        f"Checkpoint best:       "
        f"{'trovato' if config.CHECKPOINT_PATH.exists() else 'assente'} | {config.CHECKPOINT_PATH}"
    )
    print(
        f"Checkpoint last:       "
        f"{'trovato' if config.LAST_CHECKPOINT_PATH.exists() else 'assente'} | {config.LAST_CHECKPOINT_PATH}"
    )
    print("=" * 78)

    model = build_model()
    if config.EVAL_ONLY:
        history = _load_trained_model(model)
        adaptive_trace = AdaptiveTrainingTrace(
            enabled=bool(config.ADAPTIVE_MULTISTEP_ENABLED),
            initial_horizon=int(config.ADAPTIVE_H_MIN if config.ADAPTIVE_MULTISTEP_ENABLED else config.MULTISTEP_H),
            initial_teacher_steps=int(
                min(
                    config.ADAPTIVE_TEACHER_MAX if config.ADAPTIVE_MULTISTEP_ENABLED else config.MULTISTEP_H,
                    config.MULTISTEP_TEACHER_FORCING_STEPS,
                )
            ),
            final_horizon=int(config.MULTISTEP_H),
            final_teacher_steps=int(config.MULTISTEP_TEACHER_FORCING_STEPS),
            epoch_summaries=[],
        )
        if history.epochs:
            plot_training_curves(history)
        print(f"Modalita eval-only:    checkpoint caricato da {config.CHECKPOINT_PATH}")
    else:
        resume_state = try_resume_from_last_checkpoint(model)
        if config.AUTO_RESUME:
            resume_status = {
                "enabled": True,
                "resumed": bool(resume_state.resumed),
                "reason": str(resume_state.reason),
            }
            if resume_state.resumed:
                print(f"Auto-resume:           {resume_state.reason}")
            else:
                print(f"Auto-resume saltato:   {resume_state.reason}")
        history, adaptive_trace = train_model(
            model,
            dataset.train.states,
            start_epoch=resume_state.start_epoch,
            history=resume_state.history,
            optimizer_state_dict=resume_state.optimizer_state_dict,
            scheduler_state_dict=resume_state.scheduler_state_dict,
            best_loss=resume_state.best_loss,
            best_state=resume_state.best_state,
        )
        plot_training_curves(history)

    train_multistep = evaluate_multistep(model, dataset.train.states)
    test_multistep = evaluate_multistep(model, dataset.test.states)
    rollout_warmup = int(config.ROLLOUT_WARMUP_STATES)
    train_rollout = evaluate_autoregressive(model, dataset.train.states, warmup_states=rollout_warmup)
    test_rollout = evaluate_autoregressive(model, dataset.test.states, warmup_states=rollout_warmup)

    add_partial_curves = exposure_bias_detected(train_multistep.fidelity_curve, train_rollout.fidelity_curve) or (
        exposure_bias_detected(test_multistep.fidelity_curve, test_rollout.fidelity_curve)
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
    _plot_split_curves(axes[0], "Train Set", train_multistep, train_rollout, partial_results_train)
    _plot_split_curves(axes[1], "Test Set", test_multistep, test_rollout, partial_results_test)
    fig.suptitle("Fidelity multi-step vs rollout nel tempo", fontsize=14)
    fig.tight_layout()
    fig.savefig(config.FIDELITY_PLOT_PATH, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    test_seq_idx = min(int(config.OBSERVABLES_TEST_SEQUENCE_INDEX), int(dataset.test.num_sequences) - 1)
    test_single_sequence = dataset.test.states[test_seq_idx : test_seq_idx + 1]
    test_observables = compute_observable_curves(model, test_single_sequence, warmup_states=rollout_warmup)

    plot_observable_curves(
        curves=test_observables,
        warmup_states=rollout_warmup,
        output_path=config.OBSERVABLES_TEST_PLOT_PATH,
        title=(
            f"Osservabili | test sequence idx={test_seq_idx} | "
            f"{test_observables.time_indices.size} stati per traiettoria, warmup={rollout_warmup}"
        ),
    )

    summary = {
        "seed": int(config.SEED),
        "device": config.DEVICE,
        "config": {
            "N_QUBITS": int(config.N_QUBITS),
            "DIM_2N": int(config.DIM_2N),
            "NUM_STATES": int(config.NUM_STATES),
            "SEQ_LEN": int(config.SEQ_LEN),
            "MULTISTEP_H": int(config.MULTISTEP_H),
            "MULTISTEP_TEACHER_FORCING_STEPS": int(config.MULTISTEP_TEACHER_FORCING_STEPS),
            "ADAPTIVE_MULTISTEP_ENABLED": bool(config.ADAPTIVE_MULTISTEP_ENABLED),
            "ADAPTIVE_STATS_EMA": float(config.ADAPTIVE_STATS_EMA),
            "ADAPTIVE_WEIGHT_ALPHA": float(config.ADAPTIVE_WEIGHT_ALPHA),
            "ADAPTIVE_WEIGHT_MIN": float(config.ADAPTIVE_WEIGHT_MIN),
            "ADAPTIVE_WEIGHT_MAX": float(config.ADAPTIVE_WEIGHT_MAX),
            "ADAPTIVE_H_MIN": int(config.ADAPTIVE_H_MIN),
            "ADAPTIVE_H_MAX": int(config.ADAPTIVE_H_MAX),
            "ADAPTIVE_TEACHER_MIN": int(config.ADAPTIVE_TEACHER_MIN),
            "ADAPTIVE_TEACHER_MAX": int(config.ADAPTIVE_TEACHER_MAX),
            "ADAPTIVE_H_LOSS_THRESHOLD": float(config.ADAPTIVE_H_LOSS_THRESHOLD),
            "ADAPTIVE_H_FIDELITY_THRESHOLD": float(config.ADAPTIVE_H_FIDELITY_THRESHOLD),
            "ADAPTIVE_TEACHER_LOSS_THRESHOLD": float(config.ADAPTIVE_TEACHER_LOSS_THRESHOLD),
            "ADAPTIVE_TEACHER_FIDELITY_THRESHOLD": float(config.ADAPTIVE_TEACHER_FIDELITY_THRESHOLD),
            "TRAIN_SEQUENCES": int(config.TRAIN_SEQUENCES),
            "TEST_SEQUENCES": int(config.TEST_SEQUENCES),
            "INITIAL_STATE_FAMILY": config.INITIAL_STATE_FAMILY,
            "ROLLOUT_WARMUP_STATES": int(config.ROLLOUT_WARMUP_STATES),
            "EVAL_ONLY": bool(config.EVAL_ONLY),
            "AUTO_RESUME": bool(config.AUTO_RESUME),
            "PARTIAL_WARMUP_STEPS": config.PARTIAL_WARMUP_STEPS,
            "OBSERVABLES_TEST_SEQUENCE_INDEX": int(config.OBSERVABLES_TEST_SEQUENCE_INDEX),
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
        "resume": resume_status,
        "training_history": _history_as_serializable(history),
        "adaptive_training": _adaptive_training_as_serializable(adaptive_trace),
        "evaluation": {
            "train_multistep": _as_serializable(train_multistep),
            "train_autoregressive": _as_serializable(train_rollout),
            "test_multistep": _as_serializable(test_multistep),
            "test_autoregressive": _as_serializable(test_rollout),
            "test_observables_sequence_index": int(test_seq_idx),
            "test_observables": _observable_curves_as_serializable(test_observables),
            "partial_warmup_n1_values": warmup_n1_values,
            "train_partial_warmups": {str(k): _as_serializable(v) for k, v in partial_results_train.items()},
            "test_partial_warmups": {str(k): _as_serializable(v) for k, v in partial_results_test.items()},
        },
        "artifacts": {
            "fidelity_plot": str(config.FIDELITY_PLOT_PATH),
            "training_curves_plot": str(config.TRAINING_CURVES_PATH),
            "observables_train_plot": str(config.OBSERVABLES_TRAIN_PLOT_PATH),
            "observables_test_plot": str(config.OBSERVABLES_TEST_PLOT_PATH),
            "summary_json": str(config.SUMMARY_PATH),
        },
    }
    with config.SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nMetriche aggregate")
    print(f"  Train | multistep={train_multistep.mean_fidelity:.6f} | rollout={train_rollout.mean_fidelity:.6f}")
    print(f"  Test  | multistep={test_multistep.mean_fidelity:.6f} | rollout={test_rollout.mean_fidelity:.6f}")
    print(f"\nPlot fidelity:  {config.FIDELITY_PLOT_PATH}")
    print(f"Plot training:  {config.TRAINING_CURVES_PATH}")
    print(f"Obs test plot:  {config.OBSERVABLES_TEST_PLOT_PATH}")
    print(f"Summary JSON:   {config.SUMMARY_PATH}")


if __name__ == "__main__":
    main()
