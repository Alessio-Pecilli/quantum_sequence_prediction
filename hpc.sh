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
#SBATCH --time=03:00:00

set -euo pipefail

mkdir -p logs

echo "=== JOB ${SLURM_JOB_ID:-local} STARTED at $(date) on $(hostname) ==="

module purge
module load python/3.11.7

source /leonardo_work/IscrC_QuSALa/venv_py311/bin/activate

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
export QSP_TRAIN_DIAGNOSTICS=1
export QSP_TRAIN_DIAG_BATCH_PRINTS=2

# 100 evoluzioni per traiettoria: 101 stati totali -> 100 predizioni.
export QSP_N_QUBITS=4
export QSP_NUM_STATES=101
export QSP_MULTISTEP_H=100
export QSP_MULTISTEP_H_MAX=100

# Tune dataloader workers from allocated CPU cores.
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export QSP_NUM_WORKERS="$(( SLURM_CPUS_PER_TASK > 2 ? SLURM_CPUS_PER_TASK - 2 : 0 ))"
else
  export QSP_NUM_WORKERS=4
fi
export QSP_PIN_MEMORY=1

# 4 Hamiltoniane x 200 traiettorie train = 800 train totali.
export QSP_BATCH_SIZE=32
export QSP_EPOCHS=100
export QSP_HYBRID_TEACHER_FORCING_EPOCHS=20
export QSP_TRAIN_SEQUENCES=800
export QSP_TEST_SEQUENCES=200
export QSP_AUTO_RESUME=0

# Under sbatch, $0 points to a temporary copy in /var/spool/slurmd/... .
# SLURM_SUBMIT_DIR preserves the directory from which the job was submitted.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

if [[ ! -f "main_hpc.py" ]]; then
  echo "ERROR: main_hpc.py non trovato in ${PROJECT_DIR}" >&2
  exit 1
fi

MAIN_SCRIPT="${PROJECT_DIR}/main_hpc.py"

# Infer processes from visible GPUs unless explicitly set.
if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  PROC_PER_NODE="${NPROC_PER_NODE}"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  PROC_PER_NODE="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  PROC_PER_NODE="$(python - <<'PY'
import os
v = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
print(len([x for x in v.split(",") if x != ""]) if v else 1)
PY
)"
else
  PROC_PER_NODE=1
fi

echo "Working directory: ${PROJECT_DIR}"
echo "Launching torchrun with nproc_per_node=${PROC_PER_NODE}"
echo "Config: qubits=${QSP_N_QUBITS} num_states=${QSP_NUM_STATES} H=${QSP_MULTISTEP_H} train=${QSP_TRAIN_SEQUENCES} test=${QSP_TEST_SEQUENCES} epochs=${QSP_EPOCHS} batch=${QSP_BATCH_SIZE}"
echo "Main script: ${MAIN_SCRIPT}"

torchrun --standalone --nproc_per_node="${PROC_PER_NODE}" "${MAIN_SCRIPT}"

echo "=== JOB ${SLURM_JOB_ID:-local} FINISHED at $(date) ==="