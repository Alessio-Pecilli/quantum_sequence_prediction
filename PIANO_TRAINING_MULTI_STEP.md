# Piano per Training Multi-Step

## Obiettivo
Sostituire il training attuale con una loss autoregressiva multi-step: per ogni tempo `t`, il modello deve prevedere fino a `H` stati futuri riusando le proprie predizioni nel contesto (`t -> t+1`, poi `t,t+1 -> t+2`, ecc.), accumulando una loss a ogni passo.

## Stato attuale
Nel training corrente, la loss principale e' teacher-forced sull'intera sequenza, mentre il rollout autoregressivo e' solo ausiliario:

- In `trainer.py`, `train_model()` usa `predicted = model(inputs)` e confronta direttamente con `targets`.
- Nello stesso file, `_autoregressive_unroll_loss()` implementa gia' la logica di contesto crescente, ma solo come termine accessorio e con curriculum/scheduled sampling.
- In `config.py` esistono oggi solo parametri per rollout curriculum e scheduled sampling; non c'e' ancora un orizzonte esplicito `H`.
- In `predictor.py`, `QuantumSequencePredictor.forward()` accetta un contesto di lunghezza variabile e produce una previsione causale per ogni posizione, quindi l'architettura e' gia' compatibile con un unroll autoregressivo passo-passo.

## Modifiche previste
- In `config.py`: introdurre `H` con default `10` e validazione del tipo `1 <= H <= SEQ_LEN`.
- In `trainer.py`: sostituire l'attuale combinazione teacher-forced + rollout aux con una nuova routine di training che:
  - itera sui tempi iniziali `t` disponibili nella sequenza,
  - costruisce il contesto iniziale fino a `t`,
  - esegue fino a `H` passi autoregressivi o si ferma alla fine della traiettoria,
  - calcola la `NegativeLogFidelityLoss` a ogni passo,
  - aggrega le loss di tutti i tempi e di tutti i passi in una loss finale di batch.
- Sempre in `trainer.py`: adattare metriche e print di training, eliminando o ridimensionando `rollout_steps`, `scheduled_sampling_prob` e la logica di curriculum se non servono piu'.
- In `main.py`: aggiornare eventuale reportistica/config snapshot se dipende dai vecchi parametri di rollout.

## Punto tecnico chiave
La nuova loss non deve sforare la lunghezza massima della sequenza. Per ogni tempo iniziale `t`, il numero reale di passi supervisionati sara' `min(H, SEQ_LEN - t)`. Questo evita mismatch con la causal mask e con `max_seq_len=config.SEQ_LEN` del modello in `predictor.py`.

## Impatto atteso
- Vantaggio: training molto piu' allineato all'inferenza autoregressiva, quindi minore exposure bias.
- Costo: complessita' e tempo di training sensibilmente maggiori, perche' il batch non fa piu' un solo forward teacher-forced ma molti unroll annidati su tutti i tempi `t`.
- Rischio da monitorare: aumento della varianza del gradiente e training meno stabile; conviene verificare tempi, memoria GPU e andamento della fidelity nelle prime epoche.
