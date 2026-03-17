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
    evaluate_teacher_forced,
    set_seed,
    train_model,
)


def main():
    set_seed(config.SEED)
    bundle = generate_fixed_tfim_dataset()
    model = build_model()
    history = train_model(model, bundle.train.states)

    train_teacher = evaluate_teacher_forced(model, bundle.train.states)
    test_teacher = evaluate_teacher_forced(model, bundle.test.states)
    train_rollout = evaluate_autoregressive(model, bundle.train.states, warmup_states=1)
    test_rollout = evaluate_autoregressive(model, bundle.test.states, warmup_states=1)

    print("MINI EVAL OK")
    print(f"  ultime epoche registrate: {len(history.epochs)}")
    print(f"  train teacher fidelity:   {train_teacher.mean_fidelity:.6f}")
    print(f"  test  teacher fidelity:   {test_teacher.mean_fidelity:.6f}")
    print(f"  train rollout fidelity:   {train_rollout.mean_fidelity:.6f}")
    print(f"  test  rollout fidelity:   {test_rollout.mean_fidelity:.6f}")


if __name__ == "__main__":
    main()
