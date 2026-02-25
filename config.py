# ===== Configurazione Modello =====

# Dimensione dello spazio di embedding (2^n per n qubit)
D_MODEL = 64

# Numero di teste di attenzione nel Transformer
NUM_HEADS = 4

# Numero di layer del Transformer Encoder
NUM_LAYERS = 2

# ===== Configurazione Dataset =====

# Numero totale di campioni nel dataset
N_TOTALE = 10000

# Lunghezza della sequenza temporale
SEQ_LEN = 10

# ===== Configurazione Training =====

# Percentuale di dati per il training (il resto va al test)
TRAIN_SPLIT = 0.8

# Dimensione del batch
BATCH_SIZE = 32

# Numero di epoche di addestramento
EPOCHS = 50

# Learning rate dell'ottimizzatore
LEARNING_RATE = 1e-3
