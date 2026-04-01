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
    set_seed,
    train_model,
)


def main():
    set_seed(config.SEED)
    bundle = generate_fixed_tfim_dataset()
    model = build_model()
    history, adaptive_trace = train_model(model, bundle.train.states)

    train_multistep = evaluate_multistep(model, bundle.train.states)
    test_multistep = evaluate_multistep(model, bundle.test.states)
    train_rollout = evaluate_autoregressive(model, bundle.train.states, warmup_states=1)
    test_rollout = evaluate_autoregressive(model, bundle.test.states, warmup_states=1)

    print("MINI EVAL OK")
    print(f"  ultime epoche registrate: {len(history.epochs)}")
    print(
        f"  adaptive H: {adaptive_trace.initial_horizon} -> {adaptive_trace.final_horizon} | "
        f"teacher: {adaptive_trace.initial_teacher_steps} -> {adaptive_trace.final_teacher_steps}"
    )
    print(f"  train multistep fidelity: {train_multistep.mean_fidelity:.6f}")
    print(f"  test  multistep fidelity: {test_multistep.mean_fidelity:.6f}")
    print(f"  train rollout fidelity:   {train_rollout.mean_fidelity:.6f}")
    print(f"  test  rollout fidelity:   {test_rollout.mean_fidelity:.6f}")


if __name__ == "__main__":
    main()
