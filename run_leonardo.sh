#!/bin/bash
#SBATCH --job-name=qsp_encoding
#SBATCH --output=logs/qsp_%j.out
#SBATCH --error=logs/qsp_%j.err

# ==========================================
# CONFIGURAZIONE RISORSE CINECA LEONARDO (BOOST)
# ==========================================
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala
#SBATCH --time=12:00:00

# ==========================================
# TOPOLOGIA MULTI-GPU E MULTI-NODO
# ==========================================
#SBATCH --nodes=2                # Numero di nodi (2 nodi x 4 GPU = 8 GPU totali)
#SBATCH --ntasks-per-node=4      # CRITICO per DDP: 1 task MPI per ogni GPU A100
#SBATCH --cpus-per-task=8        # Distribuisce i 32 core fisici sulle 4 GPU (8 a testa)
#SBATCH --gres=gpu:4             # Alloca fisicamente le 4 GPU per nodo
#SBATCH --mem=0                  # Richiede tutta la RAM disponibile sul nodo

echo "=== JOB $SLURM_JOB_ID STARTED at $(date) ==="
echo "=== Running on nodes: $SLURM_JOB_NODELIST ==="

# ==========================================
# CARICAMENTO MODULI E AMBIENTE
# ==========================================
module purge
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.7

# Attivazione del tuo virtual environment
source /leonardo_work/IscrC_QuSALa/venv_py311/bin/activate

# ==========================================
# VARIABILI DI AMBIENTE E PATCH UCX (CINECA SPECIFIC)
# ==========================================
# Evita l'overhead dei thread classici a favore del parallelismo di PyTorch
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Patch per l'infrastruttura NVIDIA/Atos di Leonardo: previene crash "Shared memory error" e timeout
export UCX_TLS=self,shm,rc,ud
export UCX_RECONNECT_WAIT=15s
export UCX_CONNECT_TIMEOUT=300s
export UCX_MEMTYPE_CACHE=n

# Disabilita il tokenizer warning parallelo di HuggingFace (se usato indirettamente dal Transformer)
export TOKENIZERS_PARALLELISM=false

# ==========================================
# ESECUZIONE
# ==========================================
# Crea la cartella per i log di SLURM se non esiste
mkdir -p logs

# Lancia il tuo script HPC orchestrato da MPI (PMIX)
# srun assicurerà che ogni rank venga mappato alla GPU corretta (device_id = rank % 4)
srun --mpi=pmix_v3 python3 main_hpc.py

echo "=== JOB $SLURM_JOB_ID COMPLETED at $(date) ==="
