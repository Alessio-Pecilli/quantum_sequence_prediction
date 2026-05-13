# Physical TTN Implementation Report

## Summary

Questa implementazione aggiunge un nuovo backend `physical_ttn` che realizza una pipeline:

`Physical-Qubit TTN Encoder -> Transformer -> Physical-Qubit TTN Decoder`

senza rompere i backend esistenti:

- `dense_legacy`
- `flat_ttn`
- `physical_ttn`

Il nuovo backend lavora sugli assi fisici dei qubit dopo il reshape:

- input complesso: `(..., 2**N)`
- rappresentazione reale/immaginaria: `(..., 2, 2, ..., 2, 2)`

dove gli `N` assi centrali sono i qubit fisici e l'ultimo asse di dimensione `2` rappresenta `(Re, Im)`.

## Cosa e' stato cambiato

- Aggiunto `physical_ttn.py` con:
  - `TTNTreeSpec`
  - `TTNMergeSpec`
  - `PhysicalQubitTTNEncoder`
  - `PhysicalQubitTTNDecoder`
- Aggiornata `config.py` con:
  - `QSP_EMBEDDING_BACKEND`
  - `QSP_TTN_BOND_DIM`
  - `QSP_TTN_ROOT_DIM`
  - `QSP_TTN_USE_BOND_CAP`
  - `QSP_TTN_TREE_PAIRING`
  - helper `bond_dim_for_block(...)`
- Aggiornato `predictor.py` per selezionare:
  - `physical_ttn`
  - `flat_ttn`
  - `dense_legacy`
- Etichettato il TTN precedente come backend flat via alias:
  - `FlatCoefficientTTNEncoder`
  - `FlatCoefficientTTNDecoder`
- Aggiornata `ComplexMSELoss` con allineamento di fase globale verso il target prima della MSE.
- Aggiornate le snapshot config in `trainer.py` e `main.py`.
- Aggiunti:
  - `test_physical_ttn.py`
  - `physical_ttn_autoencoder_sanity.py`
  - `ablation_ttn_backends.py`

## Differenza tra `flat_ttn` e `physical_ttn`

### `flat_ttn`

Il backend `flat_ttn` parte da `view_as_real(x_complex)` con shape `(..., 2**N, 2)` e poi raggruppa coppie adiacenti dei coefficienti flattenati dello statevector.

Questo costruisce un albero sui coefficienti in ordine lineare, non sui qubit fisici.

### `physical_ttn`

Il backend `physical_ttn` fa:

1. `x_complex -> view_as_real`
2. reshape a `(..., 2, 2, ..., 2, 2)`
3. contrazioni TTN solo sugli assi dei qubit fisici
4. l'asse finale `2` di `(Re, Im)` non viene mai trattato come qubit

Quindi il merge TTN avviene tra blocchi fisici adiacenti di qubit, non tra coefficienti adiacenti del vettore flattenato.

## Costruzione della tree spec

La `TTNTreeSpec` costruisce dinamicamente l'albero a partire dalle foglie:

- `q0, q1, ..., q(N-1)`

con pairing attuale:

- `adjacent`

Per ogni livello:

- accoppia `(0,1), (2,3), (4,5), ...`
- se il numero di nodi e' dispari, l'ultimo nodo viene portato al livello successivo come carry

Ogni `TTNMergeSpec` contiene:

- `level`
- `left_node_id`
- `right_node_id`
- `parent_node_id`
- `left_dim`
- `right_dim`
- `parent_dim`
- `qubits` coperti dal nodo padre

Le dimensioni interne sono generalizzate con:

- `TTN_BOND_DIM`
- `TTN_ROOT_DIM`
- `TTN_USE_BOND_CAP`

tramite:

- `bond_dim_for_block(num_physical_qubits_in_block)`

che opzionalmente limita la bond dimension a `min(dim, 2**block_qubits)`.

## Gestione di N pari e dispari

La logica non usa formule fissate su `N=8`.

Esempi:

- `N=8`: merge perfettamente bilanciato per livelli
- `N=5`: merge dei primi quattro qubit, carry del quinto, merge finale al livello successivo
- `N=3,5,7`: i carry nodes sono mantenuti esplicitamente nella `TreeSpec` e ricostruiti dal decoder nel passaggio inverso

Il codice e' stato verificato almeno per:

- `N=1,2,3,4,5,6,7,8`

## Test eseguiti

### 1. Test strutturali TTN fisica

Comando:

- `python test_physical_ttn.py`

Esito:

- `PHYSICAL TTN TESTS OK`

Copertura del test:

- shape encoder/decoder per `N=1..8`
- verifica output complesso del decoder
- verifica `carry nodes` per `N=3,5,7`
- backward/gradient test per `N=4,5`
- smoke test `QuantumSequencePredictor` con backend `physical_ttn`
- controllo che l'output del modello sia circa normalizzato a norma 1

### 2. Sanity autoencoder fisico

Comando:

- `python physical_ttn_autoencoder_sanity.py`

Config eseguita:

- `N=3`
- `d_model=48`
- `bond_dim=16`
- `steps=250`

Risultato:

- initial mean fidelity: `0.122752`
- final mean fidelity: `0.382175`
- improvement: `+0.259422`

Questo conferma che encoder e decoder fisici riescono ad apprendere una ricostruzione migliore del baseline iniziale.

### 3. Compatibilita' backend legacy

Forward smoke eseguiti con:

- `dense_legacy`
- `flat_ttn`

Entrambi hanno prodotto output con shape corretta `(2, 4, 16)` e dtype `torch.complex64`.

### 4. Ablation smoke `dense_legacy` vs `flat_ttn` vs `physical_ttn`

Comando:

- `python ablation_ttn_backends.py`

Config smoke eseguita:

- `N_QUBITS=4`
- `NUM_STATES=5`
- `TRAIN_SEQUENCES=16`
- `TEST_SEQUENCES=8`
- `EPOCHS=1`

Risultati principali:

| backend | train fidelity mean | val fidelity mean | val median | val p10 | val p90 | train loss | val loss | sec/epoch | params | autoencoder fid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_legacy | 0.060120 | 0.066031 | 0.050504 | 0.013982 | 0.102711 | 2.792267 | 2.717807 | 1.387290 | 720287 | 0.773773 |
| flat_ttn | 0.065478 | 0.034581 | 0.019465 | 0.002526 | 0.069839 | 2.910074 | 3.682779 | 1.414364 | 818500 | 0.213817 |
| physical_ttn | 0.054469 | 0.046245 | 0.042912 | 0.005725 | 0.094653 | 2.949287 | 3.157347 | 1.372891 | 688224 | 0.175928 |

Nota:

- questa e' una smoke ablation di correttezza/integrazione, non una conclusione definitiva sulla qualita' finale del backend `physical_ttn`
- i numeri sono ottenuti con una sola epoca e dataset molto piccolo

## Limiti noti

- La TTN fisica e' strutturalmente corretta e generalizzata, ma non e' ancora ottimizzata per massimizzare la validation fidelity.
- Lo smoke autoencoder mostra miglioramento netto, ma non ricostruzione quasi perfetta con la configurazione rapida testata.
- La pairing strategy estesa (`even_odd`, `snake`, `custom`) non e' ancora implementata: attualmente e' supportato solo `adjacent`.
- L'ablation attuale e' volutamente breve per verificare integrazione e stabilita'; per conclusioni prestazionali servono run piu' lunghi.

## File principali toccati

- `physical_ttn.py`
- `config.py`
- `embedding.py`
- `predictor.py`
- `trainer.py`
- `main.py`
- `test_physical_ttn.py`
- `physical_ttn_autoencoder_sanity.py`
- `ablation_ttn_backends.py`
