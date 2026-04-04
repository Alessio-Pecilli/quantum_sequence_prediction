import os

os.environ["QSP_N_QUBITS"] = "3"
os.environ["QSP_NUM_STATES"] = "9"
os.environ["QSP_TRAIN_SEQUENCES"] = "4"
os.environ["QSP_TEST_SEQUENCES"] = "4"
os.environ["QSP_D_MODEL"] = "32"
os.environ["QSP_NUM_HEADS"] = "4"
os.environ["QSP_NUM_LAYERS"] = "2"
os.environ["QSP_DIM_FEEDFORWARD"] = "64"
os.environ["QSP_BATCH_SIZE"] = "4"
os.environ["QSP_EPOCHS"] = "2"
os.environ["QSP_INITIAL_STATE_FAMILY"] = "x_basis"
os.environ["QSP_NUM_WORKERS"] = "0"

import torch

import config
from input import QuantumSequenceDataset, generate_fixed_tfim_dataset
from predictor import ComplexMSELoss, QuantumSequencePredictor, clamp_global_phase, quantum_fidelity
from trainer import evaluate_autoregressive, evaluate_teacher_forced, set_seed


def _assert_phase_clamped(
    states: torch.Tensor,
    *,
    eps: float = 1e-8,
    atol_imag: float = 5e-5,
):
    """
    Verifica che, dopo `clamp_global_phase`, la componente 0 sia:
      - immaginaria ~ 0
      - reale >= 0
    per tutti gli elementi con |c0| > eps.
    """
    clamped = clamp_global_phase(states, eps=eps)
    c0 = clamped[..., 0]
    abs0 = torch.abs(c0)
    valid = abs0 > eps
    if valid.any():
        imag_ok = torch.abs(c0.imag)[valid] <= atol_imag
        real_ok = c0.real[valid] >= -atol_imag
        assert bool(imag_ok.all().item()), "Phase clamp check failed: imag(c0) non ~0"
        assert bool(real_ok.all().item()), "Phase clamp check failed: real(c0) < 0"


set_seed(config.SEED)
bundle = generate_fixed_tfim_dataset()

assert bundle.train.states.shape == (config.TRAIN_SEQUENCES, config.NUM_STATES, config.DIM_2N)
assert bundle.test.states.shape == (config.TEST_SEQUENCES, config.NUM_STATES, config.DIM_2N)
if not config.X_BASIS_SAMPLE_WITH_REPLACEMENT and (
    config.TRAIN_SEQUENCES + config.TEST_SEQUENCES <= 2 ** config.N_QUBITS
):
    assert set(bundle.train.initial_state_codes).isdisjoint(set(bundle.test.initial_state_codes))

train_norms = torch.linalg.vector_norm(bundle.train.states, dim=-1)
test_norms = torch.linalg.vector_norm(bundle.test.states, dim=-1)
assert torch.allclose(train_norms, torch.ones_like(train_norms), atol=1e-5)
assert torch.allclose(test_norms, torch.ones_like(test_norms), atol=1e-5)

# Check: clamping fase globale sui dati (gauge fixing coerente).
_assert_phase_clamped(bundle.train.states)
_assert_phase_clamped(bundle.test.states)

dataset = QuantumSequenceDataset(bundle.train.states, bundle.train.params)
inputs, targets, params = dataset[0]
assert inputs.shape == (config.SEQ_LEN, config.DIM_2N)
assert targets.shape == (config.SEQ_LEN, config.DIM_2N)
assert params.shape == (2,)

model = QuantumSequencePredictor()
predictions = model(bundle.train.inputs[:2], bundle.train.params[:2])
assert predictions.shape == (2, config.SEQ_LEN, config.DIM_2N)

# Check: output del modello già clamped.
_assert_phase_clamped(predictions)

criterion = ComplexMSELoss()
loss, mean_fidelity, fidelity_matrix = criterion(predictions, bundle.train.targets[:2])
assert loss.ndim == 0
assert mean_fidelity.ndim == 0
assert fidelity_matrix.shape == (2, config.SEQ_LEN)

teacher = evaluate_teacher_forced(model, bundle.train.states, bundle.train.params)
rollout = evaluate_autoregressive(model, bundle.train.states, bundle.train.params, warmup_states=1)
assert len(teacher.fidelity_curve) == config.SEQ_LEN
assert len(rollout.fidelity_curve) == config.SEQ_LEN

pairwise_fidelity = quantum_fidelity(bundle.train.targets[:2], bundle.train.targets[:2])
assert torch.allclose(pairwise_fidelity, torch.ones_like(pairwise_fidelity), atol=1e-5)

print("AUDIT OK")
print(f"  famiglia stati iniziali: {bundle.train.initial_state_family}")
print(f"  backend evoluzione:      {bundle.hamiltonian.backend}")
print(f"  teacher fidelity mean:   {teacher.mean_fidelity:.6f}")
print(f"  rollout fidelity mean:   {rollout.mean_fidelity:.6f}")
