## Predictor rebuild: X-basis clamped (2N-1)

Questo repo ora usa solo stati in base X con fase globale clampata (gauge fixing) come input/output per training ed evaluation.

### Cosa e' cambiato

- `predictor.py`
  - Modello autoregressivo causale che predice stati complessi, ma con **parametrizzazione reale ridotta**:
    - feature = `[Re(psi_0..psi_{N-1}), Im(psi_1..psi_{N-1})]` (dimensione `2N-1`, con `N=dim=2^n_qubits`).
    - `Im(psi_0)` e' implicitamente 0 (coerente col clamping sulla componente 0).
  - Clamp e normalizzazione applicati:
    - agli input del modello (elimina la liberta' di fase globale),
    - agli output del modello (stato sempre normalizzato e in gauge fissato).
  - Loss: `NegativeLogFidelityLoss` = media su batch+time di `-log F(psi_true, psi_pred)` con `F` fidelity.

- `embedding.py`
  - Proiezione learnable (MLP) dal vettore reale clampato `2N-1` allo spazio latente `d_model`.
  - Utility condivise:
    - `pack_clamped_state_features(psi)` -> feature `2N-1`
    - `unpack_clamped_state_features(feature)` -> stato complesso con `Im(psi_0)=0`

- `input.py`
  - Dataset generato con **stati iniziali solo in base X**:
    - `bit=0 -> |+>`, `bit=1 -> |->`, stato iniziale = prodotto tensoriale.
  - Evoluzione sempre con clamping step-by-step della fase globale sulla componente 0.
  - Audit sintetico (stampato) che mostra bitstring e alcune ampiezze clampate.

- `main.py`
  - Plot richiesti: fidelity vs tempo su training e test per:
    - Metodo 1: teacher forcing (predico usando sempre i veri stati precedenti).
    - Metodo 2: rollout libero (predico usando anche gli stati precedenti predetti).
    - Metodo 3 (opzionale): warmup parziale `N_1` (solo se exposure bias rilevato).
  - Salva `results_paper_logamp_phase/run_summary.json` con curve e configurazione effettiva.

### Metodi di predizione (evaluation)

- Metodo 1 (teacher forcing): per lo stato `i` do in input i veri stati `0..i-1`.
- Metodo 2 (rollout libero): per lo stato `i` do in input lo stato `0` vero e `1..i-1` predetti.
- Metodo 3 (warmup N_1): per lo stato `N_1+1` do in input `0..N_1` veri, poi continuo in rollout libero.

Interpretazione:
- train/test dovrebbero essere simili -> generalizzazione ok.
- se Metodo 2 degrada ma Metodo 1 no -> exposure bias / compounding error.

### Parametri (tutti via env `QSP_*`)

Core:
- `QSP_N_QUBITS`, `QSP_NUM_STATES`, `QSP_TRAIN_SEQUENCES`, `QSP_TEST_SEQUENCES`
- `QSP_INITIAL_STATE_FAMILY` (solo `x_basis`)
- `QSP_X_BASIS_SAMPLE_WITH_REPLACEMENT` (default `true`)

Audit stampa clampata (dataset):
- `QSP_CLAMP_AUDIT_PRINT` (default `true`)
- `QSP_CLAMP_AUDIT_MAX_SEQUENCES` (default `3`)
- `QSP_CLAMP_AUDIT_MAX_STATES` (default `4`)
- `QSP_CLAMP_AUDIT_PRINT_BITSTRINGS` (default `true`)
- `QSP_CLAMP_AUDIT_PRINT_COEFFS` (default `false`)

Rollout / exposure bias:
- `QSP_ROLLOUT_WARMUP_STATES` (warmup per metodo 2/3)
- `QSP_PARTIAL_WARMUP_STEPS`:
  - `auto` (default) oppure lista `1,3,5` (interpretabile come `N_1`)
- `QSP_EXPOSURE_BIAS_GAP_THRESHOLD`, `QSP_EXPOSURE_BIAS_DROP_THRESHOLD`

Plot:
- `QSP_PLOT_DPI`

### Come eseguire

- Mini sanity: `python test_eval.py`
- Audit clamp e coerenza shapes: `python test_audit.py`
- Run completa training+plot: `python main.py`

Output tipici:
- `results_paper_logamp_phase/fidelity_vs_time.png`
- `results_paper_logamp_phase/training_curves.png`
- `results_paper_logamp_phase/run_summary.json`

### Nota su RoPE

Per ora la posizione nel tempo e' gestita con positional encoding sinusoidale additiva.
RoPE si puo' introdurre dopo come modifica isolata (parametrizzabile) per confronto pulito.

