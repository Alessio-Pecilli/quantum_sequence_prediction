import os

os.environ["QSP_EMBEDDING_BACKEND"] = "physical_ttn"
os.environ["QSP_N_QUBITS"] = "4"
os.environ["QSP_D_MODEL"] = "32"
os.environ["QSP_NUM_HEADS"] = "4"
os.environ["QSP_NUM_LAYERS"] = "2"
os.environ["QSP_DIM_FEEDFORWARD"] = "64"
os.environ["QSP_BATCH_SIZE"] = "4"
os.environ["QSP_EPOCHS"] = "2"
os.environ["QSP_RESULTS_DIR_NAME"] = "results_test_physical_ttn"

import torch

from input import PARAM_VECTOR_DIM
from physical_ttn import PhysicalQubitTTNDecoder, PhysicalQubitTTNEncoder, TTNTreeSpec
from predictor import QuantumSequencePredictor
from trainer import set_seed


def _random_normalized_complex(*shape: int) -> torch.Tensor:
    real = torch.randn(*shape, dtype=torch.float32)
    imag = torch.randn(*shape, dtype=torch.float32)
    state = torch.complex(real, imag)
    return state / torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp(min=1e-8)


def _assert_finite_nonzero_grads(module: torch.nn.Module, *, prefix: str):
    found = False
    for name, param in module.named_parameters():
        if prefix not in name or param.grad is None:
            continue
        grad = param.grad.detach()
        assert torch.isfinite(grad).all(), f"Non-finite gradient in {name}"
        if float(grad.abs().sum().item()) > 0.0:
            found = True
    assert found, f"No non-zero gradients found for parameters matching {prefix!r}"


def test_encoder_decoder_shapes():
    d_model = 32
    for num_qubits in range(1, 9):
        dim_2n = 2 ** num_qubits
        x = _random_normalized_complex(2, 3, dim_2n)
        encoder = PhysicalQubitTTNEncoder(
            num_qubits=num_qubits,
            d_model=d_model,
            bond_dim=16,
            root_dim=16,
            use_bond_cap=True,
        )
        decoder = PhysicalQubitTTNDecoder(
            num_qubits=num_qubits,
            d_model=d_model,
            bond_dim=16,
            root_dim=16,
            use_bond_cap=True,
        )
        hidden = encoder(x)
        decoded = decoder(hidden)
        assert hidden.shape == (2, 3, d_model), f"Wrong hidden shape for N={num_qubits}: {hidden.shape}"
        assert decoded.shape == (2, 3, dim_2n), f"Wrong decoded shape for N={num_qubits}: {decoded.shape}"
        assert torch.is_complex(decoded), f"Decoder output must be complex for N={num_qubits}"


def test_odd_qubit_tree_specs():
    for num_qubits in (3, 5, 7):
        tree = TTNTreeSpec.build(num_qubits=num_qubits, bond_dim=16, root_dim=16, use_bond_cap=True)
        assert sum(len(level) for level in tree.merges_by_level) == num_qubits - 1
        assert tree.node_specs[tree.root_node_id].qubits == tuple(range(num_qubits))
        assert any(tree.carried_by_level), f"Expected at least one carry node for odd N={num_qubits}"

        x = _random_normalized_complex(2, 3, 2 ** num_qubits)
        encoder = PhysicalQubitTTNEncoder(num_qubits=num_qubits, d_model=24, bond_dim=16, root_dim=16)
        decoder = PhysicalQubitTTNDecoder(num_qubits=num_qubits, d_model=24, bond_dim=16, root_dim=16)
        decoded = decoder(encoder(x))
        assert decoded.shape == x.shape


def test_gradients():
    for num_qubits in (4, 5):
        dim_2n = 2 ** num_qubits
        x = _random_normalized_complex(2, 3, dim_2n)
        target = _random_normalized_complex(2, 3, dim_2n)
        encoder = PhysicalQubitTTNEncoder(num_qubits=num_qubits, d_model=32, bond_dim=16, root_dim=16)
        decoder = PhysicalQubitTTNDecoder(num_qubits=num_qubits, d_model=32, bond_dim=16, root_dim=16)
        decoded = decoder(encoder(x))
        loss = torch.nn.functional.mse_loss(torch.view_as_real(decoded), torch.view_as_real(target))
        loss.backward()
        _assert_finite_nonzero_grads(encoder, prefix="merge_weights")
        _assert_finite_nonzero_grads(decoder, prefix="split_weights")


def test_predictor_smoke():
    for num_qubits in (4, 5):
        dim_2n = 2 ** num_qubits
        model = QuantumSequencePredictor(
            dim_2n=dim_2n,
            d_model=32,
            num_heads=4,
            num_layers=2,
            dim_feedforward=64,
            dropout=0.0,
            max_seq_len=4,
        )
        context_states = _random_normalized_complex(2, 4, dim_2n)
        phys_params = torch.randn(2, PARAM_VECTOR_DIM, dtype=torch.float32)
        output = model(context_states, phys_params)
        norms = torch.linalg.vector_norm(output, dim=-1)
        assert output.shape == (2, 4, dim_2n)
        assert torch.is_complex(output)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)


def main():
    set_seed(7)
    test_encoder_decoder_shapes()
    test_odd_qubit_tree_specs()
    test_gradients()
    test_predictor_smoke()
    print("PHYSICAL TTN TESTS OK")


if __name__ == "__main__":
    main()
