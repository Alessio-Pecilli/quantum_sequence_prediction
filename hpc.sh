#!/bin/bash
#SBATCH --job-name=qsp_hpc
#SBATCH --output=logs/qsp_%j.out
#SBATCH --error=logs/qsp_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

set -euo pipefail

mkdir -p logs

echo "=== JOB ${SLURM_JOB_ID:-local} STARTED at $(date) on $(hostname) ==="

module purge
module load python/3.11.7

source /leonardo_work/IscrC_QuSALa/venv_py311/bin/activate

# Fail fast if launched outside a SLURM allocation (e.g. login node).
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: nessuna allocazione SLURM rilevata." >&2
  echo "Lancia con: sbatch hpc.sh oppure da una shell ottenuta con srun --pty ..." >&2
  exit 2
fi

# Threading controls: avoid CPU oversubscription on HPC.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Distributed flags for this project.
export QSP_HPC_DISTRIBUTED=1
export QSP_HPC_DISTRIBUTED_DATASET=1
export QSP_HPC_DISTRIBUTED_TRAINING=1
export QSP_HPC_DISTRIBUTED_BACKEND="auto"
export QSP_STRICT_BACKEND=1
export QSP_RESULTS_DIR_NAME="${QSP_RESULTS_DIR_NAME:-results_multi_4q_h100_optimal_v2}"
# Profilo "paper run": meno rumore log, piu' stabilita' nel training.
export QSP_TRAIN_DIAGNOSTICS=0
export QSP_TRAIN_DIAG_BATCH_PRINTS=0
export QSP_USE_AMP=0
# Evita warning verbosi di torchrun ereditati dall'ambiente interattivo.
unset TORCH_DISTRIBUTED_DEBUG || true
unset TORCH_NCCL_TRACE_BUFFER_SIZE || true

# 100 evoluzioni per traiettoria: 101 stati totali -> 100 predizioni.
export QSP_N_QUBITS=4
export QSP_NUM_STATES=101
# Profilo "tanti H": curriculum multi-step da orizzonti piccoli a H=100.
export QSP_MULTISTEP_H=100
export QSP_MULTISTEP_H_MAX=100
export QSP_MULTISTEP_H_START=20
export QSP_MULTISTEP_H_PLATEAU_PATIENCE=2
export QSP_MULTISTEP_H_PLATEAU_MIN_DELTA=5e-4

# Tune dataloader workers from allocated CPU cores.
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export QSP_NUM_WORKERS="$(( SLURM_CPUS_PER_TASK >= 4 ? 2 : 0 ))"
else
  export QSP_NUM_WORKERS=2
fi
export QSP_PIN_MEMORY=1

# 4 Hamiltoniane x 200 traiettorie train = 800 train totali.
export QSP_BATCH_SIZE=64
export QSP_EPOCHS=900
export QSP_HYBRID_TEACHER_FORCING_EPOCHS=30
export QSP_LEARNING_RATE=8e-5
export QSP_WEIGHT_DECAY=1e-5
export QSP_GRAD_CLIP_MAX_NORM=0.3
export QSP_TRAIN_SEQUENCES=1600
export QSP_TEST_SEQUENCES=400
export QSP_EARLY_STOPPING_MIN_EPOCHS=300
export QSP_EARLY_STOPPING_PATIENCE=260
export QSP_SCHEDULER_TOTAL_STEPS_CAP=0
export QSP_SCHEDULER_PCT_START=0.05
export QSP_AUTO_RESUME=1
# Al resume: reinizializza optimizer/scheduler con LR cycle sulle epoche rimanenti.
# Dopo un run completato con successo, puoi rimettere 0 per resume "soft".
export QSP_RESET_OPTIMIZER_ON_RESUME=1
export QSP_RESUME_HORIZON_OVERRIDE=0

# Under sbatch, $0 points to a temporary copy in /var/spool/slurmd/... .
# SLURM_SUBMIT_DIR preserves the directory from which the job was submitted.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

if [[ ! -f "main_hpc.py" ]]; then
  echo "ERROR: main_hpc.py non trovato in ${PROJECT_DIR}" >&2
  exit 1
fi

MAIN_SCRIPT="${PROJECT_DIR}/main_hpc.py"

# GPU preflight to avoid accidental CPU-only runs on login/service nodes.
CUDA_COUNT="$(python - <<'PY'
import torch
print(int(torch.cuda.device_count()) if torch.cuda.is_available() else 0)
PY
)"
if [[ "${CUDA_COUNT}" -lt 1 ]]; then
  echo "ERROR: nessuna GPU visibile (torch.cuda.device_count()=${CUDA_COUNT})." >&2
  echo "Host: $(hostname) | SLURM_JOB_ID=${SLURM_JOB_ID:-}" >&2
  exit 3
fi

# Infer processes from visible GPUs unless explicitly set.
if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  PROC_PER_NODE="${NPROC_PER_NODE}"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  PROC_PER_NODE="${SLURM_GPUS_ON_NODE}"
else
  PROC_PER_NODE="${CUDA_COUNT}"
fi
if [[ "${PROC_PER_NODE}" -lt 1 ]]; then
  PROC_PER_NODE=1
fi
if [[ "${PROC_PER_NODE}" -gt "${CUDA_COUNT}" ]]; then
  echo "WARNING: nproc_per_node=${PROC_PER_NODE} > cuda_count=${CUDA_COUNT}; riduco a ${CUDA_COUNT}." >&2
  PROC_PER_NODE="${CUDA_COUNT}"
fi

echo "Working directory: ${PROJECT_DIR}"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Torch CUDA device count: ${CUDA_COUNT}"
echo "Results dir name: ${QSP_RESULTS_DIR_NAME}"
echo "Launching torchrun with nproc_per_node=${PROC_PER_NODE}"
echo "Config: qubits=${QSP_N_QUBITS} num_states=${QSP_NUM_STATES} H=${QSP_MULTISTEP_H} train=${QSP_TRAIN_SEQUENCES} test=${QSP_TEST_SEQUENCES} epochs=${QSP_EPOCHS} batch=${QSP_BATCH_SIZE}"
echo "Main script: ${MAIN_SCRIPT}"

torchrun --standalone --nproc_per_node="${PROC_PER_NODE}" "${MAIN_SCRIPT}"

echo "=== JOB ${SLURM_JOB_ID:-local} FINISHED at $(date) ==="