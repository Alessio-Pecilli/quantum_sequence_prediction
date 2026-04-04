import os

os.environ["QSP_N_QUBITS"] = "4"
os.environ["QSP_NUM_STATES"] = "12"
os.environ["QSP_TRAIN_SEQUENCES"] = "16"
os.environ["QSP_TEST_SEQUENCES"] = "16"
os.environ["QSP_D_MODEL"] = "48"
os.environ["QSP_NUM_HEADS"] = "4"
os.environ["QSP_NUM_LAYERS"] = "2"
os.environ["QSP_DIM_FEEDFORWARD"] = "96"
os.environ["QSP_BATCH_SIZE"] = "8"
os.environ["QSP_EPOCHS"] = "4"
os.environ["QSP_NUM_WORKERS"] = "0"
os.environ["QSP_SAVE_MODEL"] = "0"

import config
from input import generate_fixed_tfim_dataset
from trainer import (
    build_model,
    evaluate_autoregressive,
    evaluate_multistep,
    evaluate_teacher_forced,
    set_seed,
    train_model,
)


def main():
    set_seed(config.SEED)
    bundle = generate_fixed_tfim_dataset()
    model = build_model()
    history, adaptive_trace, selection_trace = train_model(
        model,
        bundle.train.states,
        bundle.train.params,
        validation_states=bundle.test.states,
        validation_params=bundle.test.params,
    )
    assert len(history.epochs) == config.EPOCHS
    assert adaptive_trace.initial_horizon == config.MULTISTEP_H_START
    assert adaptive_trace.final_horizon >= adaptive_trace.initial_horizon
    assert adaptive_trace.initial_teacher_steps >= 1
    assert adaptive_trace.final_teacher_steps >= 1
    assert 1 <= selection_trace.best_epoch <= config.EPOCHS

    train_teacher = evaluate_teacher_forced(model, bundle.train.states, bundle.train.params)
    test_teacher = evaluate_teacher_forced(model, bundle.test.states, bundle.test.params)
    train_multistep = evaluate_multistep(model, bundle.train.states, bundle.train.params)
    test_multistep = evaluate_multistep(model, bundle.test.states, bundle.test.params)
    train_rollout = evaluate_autoregressive(model, bundle.train.states, bundle.train.params, warmup_states=1)
    test_rollout = evaluate_autoregressive(model, bundle.test.states, bundle.test.params, warmup_states=1)

    print("MINI EVAL OK")
    print(f"  ultime epoche registrate: {len(history.epochs)}")
    print(
        f"  hybrid split: teacher={config.HYBRID_TEACHER_FORCING_EPOCHS} | "
        f"multistep={config.EPOCHS - config.HYBRID_TEACHER_FORCING_EPOCHS}"
    )
    print(
        f"  H multistep: {adaptive_trace.initial_horizon} | "
        f"teacher steps: {adaptive_trace.initial_teacher_steps}"
    )
    print(f"  best epoch: {selection_trace.best_epoch} | score: {selection_trace.best_objective:.6f}")
    print(f"  train teacher fidelity:    {train_teacher.mean_fidelity:.6f}")
    print(f"  test  teacher fidelity:    {test_teacher.mean_fidelity:.6f}")
    print(f"  train multistep fidelity: {train_multistep.mean_fidelity:.6f}")
    print(f"  test  multistep fidelity: {test_multistep.mean_fidelity:.6f}")
    print(f"  train rollout fidelity:   {train_rollout.mean_fidelity:.6f}")
    print(f"  test  rollout fidelity:   {test_rollout.mean_fidelity:.6f}")


if __name__ == "__main__":
    main()
