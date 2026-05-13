# TTN 8-Qubit Handoff Report

Questo documento riassume in modo operativo tutto il lavoro svolto per portare il progetto da una pipeline dense legacy a una pipeline con Tree Tensor Network (TTN), i problemi incontrati e lo stato attuale.

## 1) Obiettivo richiesto

- Usare 8 qubit (`dim=256`) con embedding/decoder TTN.
- Evitare la rappresentazione flat legacy come percorso principale.
- Ottenere training stabile e miglioramento reale di fidelity su validation.

## 2) Modifiche architetturali principali

### 2.1 Config

- Aggiunto `TTN_LATENT_DIM` (configurabile via env).
- `OUTPUT_PARAMETRIZATION` impostata a `direct_complex` (coerente con output Re/Im).
- Aggiunto `EMBEDDING_BACKEND` con due opzioni:
  - `dense_legacy`
  - `ttn`
- Default attuale: `EMBEDDING_BACKEND="ttn"`.

### 2.2 Encoder/Decoder TTN

- `TTNEncoder` in `embedding.py`
  - Input complesso `(B,...,2^N)` -> `view_as_real` -> `(B,...,2^N,2)`.
  - Coarse-graining per coppie adiacenti fino a un singolo nodo.
  - Proiezione finale a `d_model`.

- `TTNDecoder` in `predictor.py`
  - Input `(B,...,d_model)` -> nodo latente root.
  - Fine-graining per split gerarchici fino a nodi fisici.
  - Layer finale -> `(B,...,2^N,2)` -> complesso.

### 2.3 Stabilizzazione TTN (ultima iterazione)

- Aggiunte normalizzazioni interne (`LayerNorm`) nei blocchi TTN encoder/decoder.
- Aggiunti skip/residual locali:
  - Encoder: merge + skip dalla media delle due foglie.
  - Decoder: split + skip dal parent replicato sui due figli.

## 3) Loss/training updates

- Aggiunta `CompositeQuantumLoss` in `predictor.py`:
  - `L = w * NegativeLogFidelity + (1 - w) * ComplexMSE`
- Nuovi parametri config:
  - `LOSS_TYPE` (`fidelity` | `composite`), default `composite`
  - `COMPOSITE_FIDELITY_WEIGHT`, default `0.85`
- `trainer.py` ora costruisce il criterio tramite `build_training_criterion()`.

## 4) Dataset e supporto 8 qubit

Problema trovato:
- Alcuni path dati assumevano setup 4-qubit e causavano `ValueError` con 8 qubit.

Correzioni:
- Rimosso blocco hardcoded `n_qubits == 4` nel path Haar multi-H.
- Selezione Hamiltoniane adattiva per n_qubits:
  - 4 qubit: classi complete.
  - n_qubits != 4: solo classi supportate genericamente (TFIM/XXZ).
- Aggiornata la descrizione `reason` del dataset per riflettere le classi realmente attive.

## 5) Problemi osservati durante i run

### 5.1 Regressione rispetto al legacy

- Con TTN, la training fidelity sale, ma la validation fidelity resta circa piatta vicino a baseline random per dim=256.
- Baseline random attesa circa `1/256 ~= 0.0039`.

### 5.2 Costo fase hybrid

- In diverse configurazioni, il passaggio alla fase hybrid aumenta molto il tempo/epoca.
- Il guadagno su validation non e' stato finora proporzionale al costo.

### 5.3 Batch effettivo

- Era presente un dimezzamento del batch nel loader (`BATCH_SIZE // 2`).
- Corretto a batch pieno (`BATCH_SIZE`) per coerenza tra config e runtime.

## 6) Compatibilita' / rollback controllato

Per ridurre rischio operativo e permettere confronti A/B, e' stato mantenuto anche il backend legacy:
- `ComplexEmbedding`, `pack/unpack_clamped_state_features`, `DenseLegacyDecoder`.
- Se serve confronto:
  - `QSP_EMBEDDING_BACKEND=dense_legacy` per comportamento storico.
  - `QSP_EMBEDDING_BACKEND=ttn` per pipeline TTN.

## 7) Stato attuale consigliato

Attualmente il codice e' impostato per TTN di default:
- `EMBEDDING_BACKEND="ttn"`
- `TTN_LATENT_DIM=96`
- `LOSS_TYPE="composite"`
- `COMPOSITE_FIDELITY_WEIGHT=0.85`

Questa e' la configurazione migliore introdotta finora lato stabilita' per TTN, ma i run condivisi mostrano ancora gap di generalizzazione.

## 8) Punti da verificare dal prossimo agente

1. Confronto rigoroso A/B `dense_legacy` vs `ttn` a parita' di dataset/seed/epoche.
2. Sweep mirato su:
   - `TTN_LATENT_DIM` (96, 128, 160)
   - `COMPOSITE_FIDELITY_WEIGHT` (0.80, 0.90, 0.95)
3. Verifica se il problema principale e':
   - capacita' TTN insufficiente,
   - loss non allineata al criterio di selezione,
   - difficolta' intrinseca del dataset `haar_random` multi-H.
4. Monitorare metriche non arrotondate nel log (evitare effetto "val 0.004" troncata).

## 9) Nota operativa

Le modifiche sono state validate con:
- smoke test shape encoder/decoder TTN livello-per-livello,
- smoke test forward/backward,
- verifica gradiente non nullo nei layer chiave,
- lint sui file modificati.

Nonostante cio', il tema aperto resta la generalizzazione su validation per 8 qubit in setting corrente.
