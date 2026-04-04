# Piano per Training Multi-Step

## Obiettivo
Ridurre l'`exposure bias` senza destabilizzare il modello: il training deve restare inizialmente ancorato al `teacher-forcing`, poi aumentare gradualmente la pressione autoregressiva con una loss multi-step piu' allineata all'inferenza.

## Vincoli empirici da rispettare
La relazione impone alcune guardrail che il piano deve incorporare fin dall'inizio:

- Con `SEQ_LEN = 11`, usare `H = 10` e' possibile ma molto severo.
- Con soli `12` sample di train, un training troppo aggressivo degrada anche il `one-step`.
- Il rapporto `50% teacher-forced / 50% multi-step` e' stato piu' stabile del `30% / 70%`.
- `xyz_basis` rende il problema piu' difficile; conviene mantenere un esperimento di controllo in `x_basis`.
- La metrica critica e' la tenuta `multi-step` su dati held-out, non la sola fidelity teacher-forced sul train.

## Stato attuale
Nel training corrente, la loss principale e' teacher-forced sull'intera sequenza, mentre il rollout autoregressivo e' solo ausiliario:

- In `trainer.py`, `train_model()` usa `predicted = model(inputs)` e confronta direttamente con `targets`.
- Nello stesso file, `_autoregressive_unroll_loss()` implementa gia' una logica di contesto crescente, ma come termine accessorio e con curriculum/scheduled sampling.
- In `config.py` esistono parametri per rollout curriculum e scheduled sampling; non c'e' ancora un orizzonte esplicito `H` progettato come leva principale.
- In `predictor.py`, `QuantumSequencePredictor.forward()` accetta un contesto di lunghezza variabile e produce una previsione causale per ogni posizione, quindi l'architettura e' compatibile con un unroll autoregressivo.

## Strategia proposta
Il training non va "sostituito" con puro multi-step. Va reso ibrido e progressivo:

1. Mantenere una fase iniziale teacher-forced forte per consolidare il `one-step`.
2. Introdurre una loss multi-step con orizzonte limitato e curriculum prudente.
3. Tenere per default un mix bilanciato `50% teacher-forced / 50% multi-step` come base stabile.
4. Aumentare l'aggressivita' solo se la metrica multi-step held-out si stabilizza.

## Modifiche previste
### 1. Configurazione di `H`
- In `config.py`: introdurre `MULTISTEP_H`.
- Non usare `10` come default statico.
- Default consigliato: `6`.
- Supportare opzionalmente un curriculum di unrolling:
  - partire da `H = 2`,
  - aumentare gradualmente fino a `6-8`,
  - sbloccare l'aumento solo quando la loss o fidelity teacher-forced di validazione si stabilizza.
- Validazione: `1 <= MULTISTEP_H < SEQ_LEN`.

### 2. Training ibrido invece di rimozione del teacher-forcing
- In `trainer.py`: preservare la loss teacher-forced come termine principale di ancoraggio iniziale.
- Integrare la loss multi-step come secondo termine, con peso progressivo ma non dominante troppo presto.
- Default operativo raccomandato:
  - prima fase: prevalenza teacher-forced,
  - fase centrale: mix circa `50% / 50%`,
  - evitare configurazioni aggressive tipo `30% / 70%` finche' il dataset resta cosi' piccolo.
- Scheduled sampling e rollout curriculum non vanno rimossi a priori: vanno semplificati o riassorbiti solo se la nuova routine li rende davvero ridondanti.

### 3. Unrolling vettorializzato
- Evitare una doppia iterazione Python su batch e tempi iniziali `t`, perche' penalizza troppo la GPU.
- Invece, costruire finestre sovrapposte che trasformano i tempi iniziali in una dimensione batch aggiuntiva.
- Idea operativa:
  - da una sequenza `[B, T, D]`,
  - estrarre piu' contesti iniziali e target futuri,
  - riformulare il problema come mini-batch esteso di shape concettuale `[B * N_start, H, D]`.
- Se la VRAM non basta:
  - campionare uniformemente solo un sottoinsieme di tempi iniziali `t` per batch,
  - oppure limitare il numero massimo di start positions per step.

### 4. Metriche, checkpoint ed early stopping
- In `trainer.py` e `main.py`: aggiornare reportistica e snapshot config per riflettere `MULTISTEP_H`, curriculum e pesi della loss ibrida.
- Non usare `train_fidelity` o `rollout` puro come unico criterio di selezione.
- Criterio consigliato:
  - early stopping e best checkpoint basati sulla metrica `multistep` su split held-out.
- Se oggi esiste solo `test`, prima di usarlo per early stopping conviene ricavare una vera `validation` separata, per evitare leakage sul test finale.

### 5. Regolarizzazione prudente
- La configurazione meno peggiore osservata ha `BATCH_SIZE = 8`, `WEIGHT_DECAY = 1e-4`, `DROPOUT = 0.0`.
- Questo non implica che `dropout = 0.0` sia ottimale in assoluto; implica solo che run piu' aggressivi sono peggiorati.
- Prossima ablation sensata:
  - provare `dropout = 0.1-0.15`,
  - mantenendo stabile il resto,
  - e giudicare il risultato solo sulla metrica multi-step held-out.

## Punto tecnico chiave
La loss multi-step non deve sforare la sequenza. Per ogni tempo iniziale `t`, il numero reale di passi supervisionati resta:

`min(H, SEQ_LEN - t)`

Questo evita mismatch con causal mask e `max_seq_len=config.SEQ_LEN`.

## Priorita' sperimentali
Prima di rendere il training ancora piu' complesso, conviene seguire questo ordine:

1. Fissare una baseline ibrida stabile con `H` prudente (`2 -> 6` oppure default `6`).
2. Eseguire un controllo in `x_basis` per isolare quanto del problema dipende da `xyz_basis`.
3. Aumentare il numero di traiettorie train prima di spingere ulteriormente su `H` o sulla quota multi-step.
4. Solo dopo valutare se estendere `H` fino a `8` o, con molta prudenza, `10`.

## Limite strutturale principale
La loss migliore non puo' compensare un dataset strutturalmente troppo piccolo:

- con `12` sequenze di train il rischio dominante non e' solo overfitting, ma memorizzazione quasi inevitabile;
- generare piu' traiettorie e' la priorita' piu' importante;
- se la simulazione e' costosa, serve almeno valutare augmentation fisicamente coerenti.

## Impatto atteso
- Vantaggio: training piu' allineato all'inferenza autoregressiva, con minore exposure bias rispetto al puro teacher-forcing.
- Vantaggio pratico: il mix ibrido riduce il rischio di distruggere il `one-step` troppo presto.
- Costo: maggiore complessita' implementativa e possibile pressione su tempo/VRAM.
- Rischio da monitorare: gradiente piu' rumoroso, selezione modello fuorviante e peggioramento artificiale se si usa il test come validation.
