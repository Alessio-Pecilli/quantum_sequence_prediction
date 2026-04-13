#!/bin/bash
#SBATCH --job-name=qsp_leonardo_ddp
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.7

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Workaround richiesti su Leonardo per evitare crash/intermittenze UCX.
export UCX_TLS=self,shm,rc,ud
export UCX_RECONNECT_WAIT=15s
export UCX_CONNECT_TIMEOUT=300s
export UCX_MEMTYPE_CACHE=n

srun --mpi=pmix_v3 python3 main_hpc.py
