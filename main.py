from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import json
import random
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(1)

import config
from input import (
    build_initial_states,
    build_tfim_hamiltonian,
    compute_evolution_operator,
    evolve_sequences,
    generate_fixed_tfim_dataset,
)
from trainer import (
    TrainingHistory,
    build_model,
    compute_observable_curves,
    evaluate_autoregressive,
    evaluate_teacher_forced,
    exposure_bias_detected,
    plot_observable_curves,
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


def _resolve_three_n1_values(seq_len: int) -> list[int]:
    """
    Seleziona automaticamente 3 valori sensati per N_1 (dimensione del contesto vero).
    N_1 deve stare in [1, seq_len].
    """
    seq_len = int(seq_len)
    if seq_len < 3:
        # Caso piccolo: prendiamo i primi valori unici disponibili.
        return list(range(1, seq_len + 1))[:3]

    candidates = [1, max(1, seq_len // 2), max(1, seq_len - 1)]
    unique: list[int] = []
    seen: set[int] = set()
    for v in candidates:
        if 1 <= v <= seq_len and v not in seen:
            unique.append(v)
            seen.add(v)

    if len(unique) >= 3:
        return unique[:3]

    for v in range(1, seq_len + 1):
        if v not in seen:
            unique.append(v)
            seen.add(v)
        if len(unique) >= 3:
            break
    return unique[:3]


@torch.no_grad()
def _generate_test_states_traditional(
    test_sequences: int,
    evolution_operator: torch.Tensor,
    *,
    seed: int,
    n_qubits: int,
    num_states: int,
) -> tuple[torch.Tensor, list[int]]:
    """
    Test tradizionale:
      - Hamiltoniana identica al training (stesso U fissato)
      - stato iniziale scelto random dalla base computazionale, con rimpiazzo
      - traiettoria: psi(k) = U^k psi(0)
    """
    rng = random.Random(int(seed))
    dim = 2**int(n_qubits)
    initial_state_codes = [rng.randrange(dim) for _ in range(int(test_sequences))]
    initial_states = build_initial_states(
        codes=initial_state_codes,
        family="basis",
        n_qubits=int(n_qubits),
    )
    states = evolve_sequences(
        initial_states=initial_states,
        evolution_operator=evolution_operator,
        num_states=int(num_states),
        device=config.DEVICE,
    )
    return states, initial_state_codes


@torch.no_grad()
def _generate_test_states_with_h_new_tfim(
    test_initial_state_codes: list[int],
    initial_state_family: str,
    *,
    seed: int,
    n_qubits: int,
    num_states: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """
    Crea un test set generando traiettorie con una Hamiltoniana TFIM fissata H_new:
      - campo trasverso uguale a config.FIELD_STRENGTH
      - couplings J_i ~ Normal(mean=1, std=1) (varianza 1)
    """
    n_qubits = int(n_qubits)
    num_states = int(num_states)
    if n_qubits < 1:
        raise ValueError(f"n_qubits deve essere >= 1, ricevuto: {n_qubits}")
    if num_states < 2:
        raise ValueError(f"num_states deve essere >= 2, ricevuto: {num_states}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    if n_qubits == 1:
        couplings = torch.empty((0,), dtype=torch.float32)
    else:
        couplings = torch.normal(
            mean=1.0,
            std=1.0,
            size=(n_qubits - 1,),
            generator=generator,
        ).to(torch.float32)

    hamiltonian = build_tfim_hamiltonian(
        n_qubits=n_qubits,
        couplings=couplings,
        field_strength=float(config.FIELD_STRENGTH),
    )
    evolution_operator, backend = compute_evolution_operator(hamiltonian, float(config.TIME_STEP))

    initial_states = build_initial_states(
        codes=list(test_initial_state_codes),
        family=str(initial_state_family),
        n_qubits=n_qubits,
    )
    states = evolve_sequences(
        initial_states=initial_states,
        evolution_operator=evolution_operator,
        num_states=num_states,
        device=config.DEVICE,
    )

    h_new_payload = {
        "couplings": [float(value) for value in couplings.tolist()],
        "field_strength": float(config.FIELD_STRENGTH),
        "backend": backend,
    }
    return states, h_new_payload


@torch.no_grad()
def _exposure_bias_gap_and_detected(
    teacher_forced_curve: list[float],
    autoregressive_curve: list[float],
) -> tuple[float, bool]:
    """
    Gap alla coda e rilevamento (coerente con la logica di `exposure_bias_detected`):
      gap = mean( teacher_tail - rollout_tail )
      drop = mean( rollout_head - rollout_tail )
    """
    tf = torch.tensor(teacher_forced_curve, dtype=torch.float32)
    ar = torch.tensor(autoregressive_curve, dtype=torch.float32)
    valid = torch.isfinite(ar)
    if valid.sum() < 4:
        return float("nan"), False

    tf_valid = tf[valid]
    ar_valid = ar[valid]
    tail = max(1, int(ar_valid.numel()) // 4)

    gap = float((tf_valid[-tail:] - ar_valid[-tail:]).mean().item())
    drop = float((ar_valid[:tail].mean() - ar_valid[-tail:].mean()).item())
    detected = gap >= float(config.EXPOSURE_BIAS_GAP_THRESHOLD) and drop >= float(
        config.EXPOSURE_BIAS_DROP_THRESHOLD
    )
    return gap, bool(detected)


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

    print("\n[2/5] Valutazione teacher forced")
    train_teacher = evaluate_teacher_forced(model, dataset.train.states)
    traditional_seed = int(config.SEED) + 777
    test_states_traditional, traditional_codes = _generate_test_states_traditional(
        test_sequences=config.TEST_SEQUENCES,
        evolution_operator=dataset.hamiltonian.evolution_operator,
        seed=traditional_seed,
        n_qubits=config.N_QUBITS,
        num_states=config.NUM_STATES,
    )
    test_teacher_traditional = evaluate_teacher_forced(model, test_states_traditional)

    h_new_seed = int(config.SEED) + 999
    test_states_h_new, h_new_payload = _generate_test_states_with_h_new_tfim(
        dataset.test.initial_state_codes,
        dataset.test.initial_state_family,
        seed=h_new_seed,
        n_qubits=config.N_QUBITS,
        num_states=config.NUM_STATES,
    )
    print("\nHamiltoniana H_new (test set, TFIM)")
    print(
        f"  couplings J_i ~ Normal(mean=1, var=1), seed={h_new_seed}. "
        f"field_strength={h_new_payload['field_strength']:.3f}, dt={config.TIME_STEP:.3f}, "
        f"backend={h_new_payload['backend']}"
    )
    print("  J_i: " + ", ".join(f"{value:.4f}" for value in h_new_payload["couplings"]))
    test_teacher_h_new = evaluate_teacher_forced(model, test_states_h_new)
    _flush_device_memory()

    print("\n[3/5] Valutazione autoregressiva")
    rollout_warmup = int(config.ROLLOUT_WARMUP_STATES)
    train_rollout = evaluate_autoregressive(model, dataset.train.states, warmup_states=rollout_warmup)
    test_rollout_traditional = evaluate_autoregressive(
        model,
        test_states_traditional,
        warmup_states=rollout_warmup,
    )
    test_rollout_h_new = evaluate_autoregressive(model, test_states_h_new, warmup_states=rollout_warmup)
    _flush_device_memory()

    print("\n[4/5] Osservabili (esatto vs rollout)")
    train_obs_curves = compute_observable_curves(
        model,
        dataset.train.states,
        warmup_states=rollout_warmup,
    )
    traditional_obs_curves = compute_observable_curves(
        model,
        test_states_traditional,
        warmup_states=rollout_warmup,
    )
    h_new_obs_curves = compute_observable_curves(
        model,
        test_states_h_new,
        warmup_states=rollout_warmup,
    )
    train_obs_path = config.RESULTS_DIR / "observables_train_vs_rollout.png"
    test_traditional_obs_path = config.RESULTS_DIR / "observables_test_traditional_vs_rollout.png"
    test_h_new_obs_path = config.RESULTS_DIR / "observables_test_h_new_vs_rollout.png"
    plot_observable_curves(
        train_obs_curves,
        warmup_states=rollout_warmup,
        output_path=train_obs_path,
        title=f"Osservabili | train | {config.NUM_STATES} stati, warmup={rollout_warmup}",
    )
    plot_observable_curves(
        traditional_obs_curves,
        warmup_states=rollout_warmup,
        output_path=test_traditional_obs_path,
        title=f"Osservabili | test tradizionale (H train) | {config.NUM_STATES} stati, warmup={rollout_warmup}",
    )
    plot_observable_curves(
        h_new_obs_curves,
        warmup_states=rollout_warmup,
        output_path=test_h_new_obs_path,
        title=f"Osservabili | test H_new gaussiana | {config.NUM_STATES} stati, warmup={rollout_warmup}",
    )
    _flush_device_memory()

    train_exposure_bias = exposure_bias_detected(
        train_teacher.fidelity_curve,
        train_rollout.fidelity_curve,
    )
    test_exposure_bias_traditional = exposure_bias_detected(
        test_teacher_traditional.fidelity_curve,
        test_rollout_traditional.fidelity_curve,
    )
    test_exposure_bias_h_new = exposure_bias_detected(
        test_teacher_h_new.fidelity_curve,
        test_rollout_h_new.fidelity_curve,
    )
    add_partial_curves = train_exposure_bias or test_exposure_bias_traditional or test_exposure_bias_h_new

    partial_results_train: dict[int, Any] = {}
    partial_results_test_traditional: dict[int, Any] = {}
    partial_results_test_h_new: dict[int, Any] = {}
    warmup_n1_values: list[int] = []
    if add_partial_curves:
        warmup_n1_values = resolve_partial_warmup_steps(config.SEQ_LEN)
        print("\n[5/5] Exposure bias rilevato, aggiungo warmup parziali N1=" + ", ".join(str(value) for value in warmup_n1_values))
        for warmup_n1 in warmup_n1_values:
            warmup_states = warmup_n1 + 1
            partial_results_train[warmup_n1] = evaluate_autoregressive(
                model,
                dataset.train.states,
                warmup_states=warmup_states,
            )
            partial_results_test_traditional[warmup_n1] = evaluate_autoregressive(
                model,
                test_states_traditional,
                warmup_states=warmup_states,
            )
            partial_results_test_h_new[warmup_n1] = evaluate_autoregressive(
                model,
                test_states_h_new,
                warmup_states=warmup_states,
            )
    else:
        print("\n[5/5] Nessun exposure bias marcato: mantengo solo metodo 1 e 2.")
    _flush_device_memory()

    fig, axes = plt.subplots(1, 3, figsize=(23, 5.5), sharey=True)
    _plot_split_curves(
        axes[0],
        "Train Set",
        train_teacher,
        train_rollout,
        partial_results_train,
    )
    _plot_split_curves(
        axes[1],
        "Test Tradizionale (H train)",
        test_teacher_traditional,
        test_rollout_traditional,
        partial_results_test_traditional,
    )
    _plot_split_curves(
        axes[2],
        "Test H_new (gaussiana)",
        test_teacher_h_new,
        test_rollout_h_new,
        partial_results_test_h_new,
    )
    fig.suptitle("Fidelity vero vs predetto: train, test tradizionale, test H_new", fontsize=14)
    fig.tight_layout()
    fig.savefig(config.FIDELITY_PLOT_PATH, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    print("\nMetriche aggregate")
    print(
        f"  Train | teacher={train_teacher.mean_fidelity:.6f} | rollout={train_rollout.mean_fidelity:.6f}"
    )
    print(
        f"  Test tradizionale | teacher={test_teacher_traditional.mean_fidelity:.6f} | rollout={test_rollout_traditional.mean_fidelity:.6f}"
    )
    print(
        f"  Test H_new        | teacher={test_teacher_h_new.mean_fidelity:.6f} | rollout={test_rollout_h_new.mean_fidelity:.6f}"
    )

    # --- Plot exposure bias vs numero stati veri in input (fase di test) ---
    n1_values = _resolve_three_n1_values(config.SEQ_LEN)
    bias_gaps_traditional: list[float] = []
    detected_flags_traditional: list[bool] = []
    bias_gaps_h_new: list[float] = []
    detected_flags_h_new: list[bool] = []

    print("\nExposure bias vs N_1 (entrambi i test set)")
    for n1 in n1_values:
        test_rollout_n1_traditional = evaluate_autoregressive(
            model,
            test_states_traditional,
            warmup_states=n1,
        )
        gap_traditional, detected_traditional = _exposure_bias_gap_and_detected(
            test_teacher_traditional.fidelity_curve,
            test_rollout_n1_traditional.fidelity_curve,
        )
        test_rollout_n1_h_new = evaluate_autoregressive(model, test_states_h_new, warmup_states=n1)
        gap_h_new, detected_h_new = _exposure_bias_gap_and_detected(
            test_teacher_h_new.fidelity_curve,
            test_rollout_n1_h_new.fidelity_curve,
        )
        bias_gaps_traditional.append(gap_traditional)
        detected_flags_traditional.append(detected_traditional)
        bias_gaps_h_new.append(gap_h_new)
        detected_flags_h_new.append(detected_h_new)
        gap_traditional_str = "nan" if not np.isfinite(gap_traditional) else f"{gap_traditional:.6f}"
        gap_h_new_str = "nan" if not np.isfinite(gap_h_new) else f"{gap_h_new:.6f}"
        print(
            f"  N_1={n1:2d} | trad_gap={gap_traditional_str}, trad_det={detected_traditional} | "
            f"hnew_gap={gap_h_new_str}, hnew_det={detected_h_new}"
        )
        _flush_device_memory()

    exposure_bias_plot_path = config.RESULTS_DIR / "exposure_bias_vs_N1.png"
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.array(n1_values, dtype=np.int32)
    y_trad = np.array(bias_gaps_traditional, dtype=np.float64)
    y_h_new = np.array(bias_gaps_h_new, dtype=np.float64)

    finite_trad = np.isfinite(y_trad)
    finite_h_new = np.isfinite(y_h_new)
    if finite_trad.any():
        ax.plot(
            x[finite_trad],
            y_trad[finite_trad],
            color="#1f618d",
            linewidth=2.0,
            marker="o",
            label="test tradizionale",
        )
    if finite_h_new.any():
        ax.plot(
            x[finite_h_new],
            y_h_new[finite_h_new],
            color="#b03a2e",
            linewidth=2.0,
            marker="s",
            label="test H_new",
        )

    ax.axhline(
        float(config.EXPOSURE_BIAS_GAP_THRESHOLD),
        color="gray",
        linestyle="--",
        linewidth=1.4,
        label="soglia (gap)",
    )
    ax.set_title("Exposure bias vs N_1 (test tradizionale e H_new)")
    ax.set_xlabel("N_1 = numero di stati veri in input (warmup_states)")
    ax.set_ylabel("gap = mean(teacher_tail - rollout_tail)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(exposure_bias_plot_path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot exposure bias: {exposure_bias_plot_path}")

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
            "test_traditional_initial_state_codes": traditional_codes,
        },
        "hamiltonian": {
            "couplings": dataset.hamiltonian.couplings,
            "field_strength": float(dataset.hamiltonian.field_strength),
            "backend": dataset.hamiltonian.backend,
        },
        "test_h_new": {
            "seed": int(h_new_seed),
            "couplings": h_new_payload["couplings"],
            "field_strength": float(h_new_payload["field_strength"]),
            "backend": h_new_payload["backend"],
            "initial_state_family": dataset.test.initial_state_family,
        },
        "training_history": {
            "epochs": history.epochs,
            "train_loss": history.train_loss,
            "train_fidelity": history.train_fidelity,
        },
        "evaluation": {
            "train_teacher_forced": _as_serializable(train_teacher),
            "train_autoregressive": _as_serializable(train_rollout),
            "test_traditional_teacher_forced": _as_serializable(test_teacher_traditional),
            "test_traditional_autoregressive": _as_serializable(test_rollout_traditional),
            "test_h_new_teacher_forced": _as_serializable(test_teacher_h_new),
            "test_h_new_autoregressive": _as_serializable(test_rollout_h_new),
            "exposure_bias": {
                "train": bool(train_exposure_bias),
                "test_traditional": bool(test_exposure_bias_traditional),
                "test_h_new": bool(test_exposure_bias_h_new),
            },
            "partial_warmup_n1_values": warmup_n1_values,
            "train_partial_warmups": {
                str(key): _as_serializable(value) for key, value in partial_results_train.items()
            },
            "test_traditional_partial_warmups": {
                str(key): _as_serializable(value)
                for key, value in partial_results_test_traditional.items()
            },
            "test_h_new_partial_warmups": {
                str(key): _as_serializable(value) for key, value in partial_results_test_h_new.items()
            },
            "exposure_bias_vs_n1": {
                "n1_values": [int(v) for v in n1_values],
                "traditional_gap_values": [
                    float(v) if np.isfinite(v) else None for v in bias_gaps_traditional
                ],
                "traditional_detected_flags": [bool(v) for v in detected_flags_traditional],
                "h_new_gap_values": [float(v) if np.isfinite(v) else None for v in bias_gaps_h_new],
                "h_new_detected_flags": [bool(v) for v in detected_flags_h_new],
                "gap_threshold": float(config.EXPOSURE_BIAS_GAP_THRESHOLD),
                "drop_threshold": float(config.EXPOSURE_BIAS_DROP_THRESHOLD),
            },
        },
        "artifacts": {
            "fidelity_plot": str(config.FIDELITY_PLOT_PATH),
            "training_curves_plot": str(config.TRAINING_CURVES_PATH),
            "observables_train_plot": str(train_obs_path),
            "observables_test_traditional_plot": str(test_traditional_obs_path),
            "observables_test_h_new_plot": str(test_h_new_obs_path),
            "exposure_bias_plot": str(exposure_bias_plot_path),
            "last_checkpoint": str(config.LAST_CHECKPOINT_PATH),
            "checkpoint": str(config.CHECKPOINT_PATH),
        },
        "train_observables": {
            "time_indices": train_obs_curves.time_indices.tolist(),
            "physical_time": train_obs_curves.physical_time.tolist(),
            "mz_exact": train_obs_curves.mz_exact.tolist(),
            "mz_pred": train_obs_curves.mz_pred.tolist(),
            "mx_exact": train_obs_curves.mx_exact.tolist(),
            "mx_pred": train_obs_curves.mx_pred.tolist(),
            "cz_exact": train_obs_curves.cz_exact.tolist(),
            "cz_pred": train_obs_curves.cz_pred.tolist(),
            "rollout_warmup_states": int(rollout_warmup),
        },
        "test_traditional_observables": {
            "time_indices": traditional_obs_curves.time_indices.tolist(),
            "physical_time": traditional_obs_curves.physical_time.tolist(),
            "mz_exact": traditional_obs_curves.mz_exact.tolist(),
            "mz_pred": traditional_obs_curves.mz_pred.tolist(),
            "mx_exact": traditional_obs_curves.mx_exact.tolist(),
            "mx_pred": traditional_obs_curves.mx_pred.tolist(),
            "cz_exact": traditional_obs_curves.cz_exact.tolist(),
            "cz_pred": traditional_obs_curves.cz_pred.tolist(),
            "rollout_warmup_states": int(rollout_warmup),
        },
        "test_h_new_observables": {
            "time_indices": h_new_obs_curves.time_indices.tolist(),
            "physical_time": h_new_obs_curves.physical_time.tolist(),
            "mz_exact": h_new_obs_curves.mz_exact.tolist(),
            "mz_pred": h_new_obs_curves.mz_pred.tolist(),
            "mx_exact": h_new_obs_curves.mx_exact.tolist(),
            "mx_pred": h_new_obs_curves.mx_pred.tolist(),
            "cz_exact": h_new_obs_curves.cz_exact.tolist(),
            "cz_pred": h_new_obs_curves.cz_pred.tolist(),
            "rollout_warmup_states": int(rollout_warmup),
        },
    }

    with config.SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nPlot fidelity:         {config.FIDELITY_PLOT_PATH}")
    print(f"Plot training:         {config.TRAINING_CURVES_PATH}")
    print(f"Plot osservabili train: {train_obs_path}")
    print(f"Plot osservabili trad.: {test_traditional_obs_path}")
    print(f"Plot osservabili H_new: {test_h_new_obs_path}")
    print(f"Summary JSON:          {config.SUMMARY_PATH}")
    if config.SAVE_MODEL:
        print(f"Checkpoint:            {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
