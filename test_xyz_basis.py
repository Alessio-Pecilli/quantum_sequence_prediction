import os

os.environ["QSP_N_QUBITS"] = "2"
os.environ["QSP_NUM_STATES"] = "6"
os.environ["QSP_TRAIN_SEQUENCES"] = "6"
os.environ["QSP_TEST_SEQUENCES"] = "6"
os.environ["QSP_INITIAL_STATE_FAMILY"] = "xyz_basis"
os.environ["QSP_INITIAL_STATE_SAMPLE_WITH_REPLACEMENT"] = "0"
os.environ["QSP_SAVE_MODEL"] = "0"
os.environ["QSP_NUM_WORKERS"] = "0"

import config
from input import generate_fixed_tfim_dataset


bundle = generate_fixed_tfim_dataset()
support_size = 3 * (2 ** config.N_QUBITS)

all_codes = bundle.train.initial_state_codes + bundle.test.initial_state_codes
assert len(all_codes) == support_size
assert len(set(all_codes)) == support_size
assert set(bundle.train.initial_state_codes).isdisjoint(set(bundle.test.initial_state_codes))

basis_indices = sorted({code // (2 ** config.N_QUBITS) for code in all_codes})
assert basis_indices == [0, 1, 2]

print("XYZ BASIS OK")
print(f"  supporto totale:      {support_size}")
print(f"  codici train/test:    {len(bundle.train.initial_state_codes)}/{len(bundle.test.initial_state_codes)}")
print(f"  basi globali viste:   {basis_indices}")
