# Relazione Su Parametri E Stabilita

## Obiettivo
Vogliamo ottenere buoni risultati nel problema di predizione quantistica con:

- `4` qubit
- `12` stati per traiettoria
- training ibrido: prima `teacher-forced`, poi `multi-step`
- valutazione finale anche in `rollout` libero
- stati iniziali in `xyz_basis` senza rimpiazzo

L'obiettivo vero non e' massimizzare solo la fidelity locale, ma ottenere un modello che:

- sappia fare bene il `one-step`
- regga in `multi-step`
- non collassi in `rollout`

## Riassunto Esecutivo

Le scoperte principali sono queste:

1. Con `12` sequenze train e `12` test il dataset e' troppo piccolo per sostenere scelte di training molto aggressive.
2. `xyz_basis` rende il problema piu' interessante ma anche piu' difficile; con pochi dati amplifica la fragilita' del rollout.
3. Il rapporto `50% teacher-forced / 50% multi-step` e' stato piu' stabile del `30% / 70%`.
4. Spingere troppo la selezione del best model sul rollout non ha risolto il problema; in alcuni run ha peggiorato tutto.
5. `H = 10` e' molto severo su `SEQ_LEN = 11`: si puo' usare, ma solo con parametri prudenti.
6. Il problema residuo piu' importante non e' piu' il one-step, ma l'`exposure bias`.

## Come leggere i risultati

Nel progetto adesso abbiamo tre livelli di valutazione:

- `teacher_forced`: il modello vede sempre il contesto corretto
- `multistep`: dopo alcuni passi corretti usa le proprie predizioni
- `rollout`: dopo il warmup iniziale va completamente "in caduta libera"

Interpretazione:

- se `teacher_forced` e' basso anche su test, il modello non ha imparato bene nemmeno il passo singolo
- se `teacher_forced` e `multistep` sono buoni ma `rollout` crolla, il problema principale e' `exposure bias`
- se train e test divergono molto, c'e' `overfitting`

## Cosa abbiamo osservato nei run

### Run molto aggressivo ma relativamente stabile
Configurazione chiave:

- `EPOCHS = 10000`
- `HYBRID_TEACHER_FORCING_EPOCHS = 5000`
- `BATCH_SIZE = 8`
- `LEARNING_RATE = 5e-5`
- `WEIGHT_DECAY = 1e-4`
- `DROPOUT = 0.0`
- `MULTISTEP_H = 10`
- `teacher_steps = 5`

Risultati migliori osservati:

- `test_teacher_forced ~ 0.836`
- `test_multistep ~ 0.770`
- `test_rollout ~ 0.450`

Interpretazione:

- il modello aveva imparato bene il `one-step`
- il `multi-step` locale era discreto
- il `rollout` restava troppo basso

Conclusione:

- questa configurazione non e' "buona", ma e' stata la meno peggiore
- il problema dominante qui e' `exposure bias`

### Run con piu' peso al multi-step e selezione quasi tutta sul rollout
Configurazione chiave:

- circa `30%` teacher-forced
- circa `70%` multi-step
- `LR` abbassato
- `weight_decay` aumentato
- `dropout` introdotto
- best model pesato molto di piu' sul `rollout`

Risultati osservati:

- `test_teacher_forced ~ 0.19`
- `test_multistep ~ 0.16`
- `test_rollout ~ 0.07`

Interpretazione:

- questa variante non ha solo peggiorato il rollout
- ha distrutto anche la generalizzazione nel `teacher-forced`

Conclusione:

- il training e' diventato troppo aggressivo troppo presto
- con pochi dati questa strategia e' da evitare

## Lezioni principali sulla gestione dei parametri

### 1. Il dataset e' il collo di bottiglia principale
Con `12` sequenze train e `12` test il modello puo' facilmente memorizzare il train invece di imparare una dinamica robusta.

Segnali tipici:

- `train_teacher_forced` molto alto
- `test_teacher_forced` molto piu' basso
- `test_rollout` ancora peggiore

Questo significa che molte scelte di training sembrano "funzionare" sul train ma non generalizzano davvero.

### 2. `xyz_basis` e' piu' difficile di `x_basis`
La famiglia `xyz_basis` aumenta la varieta' degli stati iniziali. E' una scelta interessante e coerente con l'obiettivo fisico, ma aumenta la difficolta'.

Con pochi dati:

- il modello fa piu' fatica a consolidare il `one-step`
- il `multi-step` e' piu' fragile
- il `rollout` soffre di piu' l'accumulo d'errore

Quindi `xyz_basis` va bene, ma richiede maggiore prudenza sui parametri.

### 3. Ridurre troppo la fase teacher-forced peggiora tutto
Abbiamo provato a spingere il training verso:

- circa `30%` teacher-forced
- circa `70%` multi-step

Risultato:

- peggioramento netto anche del `teacher-forced` su test
- peggioramento del `rollout` train e test

Conclusione:

- con dataset piccolo, una fase `teacher-forced` piu' lunga e' utile
- il rapporto `50% / 50%` e' stato molto piu' stabile del `30% / 70%`

### 4. Dare troppo peso al rollout nella selezione del best model non basta
Abbiamo provato a scegliere il best model con una funzione obiettivo molto sbilanciata sul rollout.

Questo non ha risolto il problema:

- se il run e' cattivo nel complesso, il "miglior" rollout resta comunque scarso
- si rischia di selezionare un modello che non generalizza nemmeno nel `teacher-forced`

Conclusione:

- il criterio di selezione aiuta
- ma non sostituisce una dinamica di training sana

### 5. `H = 10` e' severo ma ancora gestibile
Con `SEQ_LEN = 11`, usare `H = 10` significa chiedere un `multi-step` quasi equivalente a un full rollout.

Questo e' utile se vogliamo combattere exposure bias, ma:

- aumenta molto la difficolta' del training
- amplifica il rischio di drift autoregressivo
- richiede una fase `teacher-forced` non troppo corta

Quindi `H = 10` si puo' tenere, ma non assieme a scelte troppo aggressive su tutti gli altri parametri.

### 6. Batch troppo grande su pochi dati rende l'epoca poco informativa
Se il batch effettivo copre quasi tutto il dataset, ogni epoca diventa quasi un unico update globale.

Con `12` sample train:

- batch troppo grandi producono poche correzioni per epoca
- il training e' meno leggibile
- e' piu' difficile capire se il modello sta imparando o solo oscillando

Tenere `BATCH_SIZE = 8` e' stato piu' ragionevole del batch effettivamente troppo grande.

### 7. Piu' epoche da sole non risolvono il problema
Aumentare molto le epoche puo' aiutare il train, ma non risolve automaticamente il rollout.

Infatti abbiamo visto run con:

- `train_fidelity` quasi perfetta
- `test_teacher_forced` buono
- `test_rollout` ancora mediocre

Quindi il problema non e' semplicemente "serve allenare piu' a lungo".

## Configurazione meno peggiore osservata

Questa e' la configurazione che finora ha dato i segnali meno negativi e che conviene usare come base di riferimento:

- `N_QUBITS = 4`
- `NUM_STATES = 12`
- `TRAIN_SEQUENCES = 12`
- `TEST_SEQUENCES = 12`
- `INITIAL_STATE_FAMILY = xyz_basis`
- `BATCH_SIZE = 8`
- `EPOCHS = 10000`
- `LEARNING_RATE = 5e-5`
- `WEIGHT_DECAY = 1e-4`
- `DROPOUT = 0.0`
- `MULTISTEP_H = 10`
- `MULTISTEP_EFFECTIVE_TEACHER_FORCING_STEPS = 5`
- `HYBRID_TEACHER_FORCING_EPOCHS = EPOCHS // 2`
- `ROLLOUT_WARMUP_STATES = 2`

Importante:

- questa non e' una configurazione "buona"
- e' semplicemente la migliore tra quelle testate finora

## Diagnosi attuale

Ad oggi la situazione piu' probabile e':

- il modello impara il `one-step`
- il modello regge discretamente in `multi-step`
- il collo di bottiglia vero resta il `rollout`

Quindi il problema dominante attuale e' ancora `exposure bias`.

## Cosa evitare

Sulla base dei run fatti, conviene evitare:

- ridurre troppo la fase `teacher-forced`
- spingere troppo presto sul `multi-step`
- aumentare contemporaneamente `dropout`, `weight_decay` e pressione sul rollout
- usare una selezione del best model quasi totalmente sbilanciata sul rollout
- interpretare una `train_fidelity` molto alta come segnale di buona generalizzazione

## Strategia consigliata per ottenere risultati migliori

Ordine consigliato degli esperimenti futuri:

1. Tenere fissa la configurazione meno peggiore come baseline.
2. Se il rollout resta basso, aumentare prima il numero di sequenze, non la complessita' del training.
3. Solo dopo valutare se abbassare `H` da `10` a `6-8`.
4. Se serve isolare il ruolo della famiglia iniziale, confrontare `xyz_basis` con `x_basis`.

## Principio guida finale

Con pochi dati, la stabilita' conta piu' dell'aggressivita'.

Meglio un training meno "eroico" ma interpretabile, piuttosto che una strategia molto autoregressiva che sul train sembra forte ma distrugge la generalizzazione.
