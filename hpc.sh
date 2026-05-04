#!/bin/bash
#SBATCH --job-name=qsp_hpc
#SBATCH --output=logs/qsp_%j.out
#SBATCH --error=logs/qsp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

set -euo pipefail

mkdir -p logs

echo "=== JOB ${SLURM_JOB_ID:-local} STARTED at $(date) on $(hostname) ==="

# Threading controls: avoid CPU oversubscription on HPC.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Distributed flags for this project.
export QSP_HPC_DISTRIBUTED=1
export QSP_HPC_DISTRIBUTED_DATASET=1
export QSP_HPC_DISTRIBUTED_TRAINING=1
export QSP_HPC_DISTRIBUTED_BACKEND="${QSP_HPC_DISTRIBUTED_BACKEND:-auto}"

# Tune dataloader workers from allocated CPU cores.
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export QSP_NUM_WORKERS="${QSP_NUM_WORKERS:-$(( SLURM_CPUS_PER_TASK > 2 ? SLURM_CPUS_PER_TASK - 2 : 0 ))}"
else
  export QSP_NUM_WORKERS="${QSP_NUM_WORKERS:-4}"
fi
export QSP_PIN_MEMORY="${QSP_PIN_MEMORY:-1}"

# Optional training overrides (examples).
# export QSP_BATCH_SIZE=256
# export QSP_EPOCHS=200

# Activate your environment here if needed:
# source /path/to/venv/bin/activate

cd "$(dirname "$0")"

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

echo "Launching torchrun with nproc_per_node=${PROC_PER_NODE}"

torchrun --standalone --nproc_per_node="${PROC_PER_NODE}" main_hpc.py

echo "=== JOB ${SLURM_JOB_ID:-local} FINISHED at $(date) ==="
