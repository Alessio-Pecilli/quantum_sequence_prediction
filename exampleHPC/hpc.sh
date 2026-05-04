#!/bin/bash
#SBATCH --job-name=vqt_train
#SBATCH --output=logs/vqt_%j.out
#SBATCH --error=logs/vqt_%j.err

#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala

#SBATCH --nodes=5
#SBATCH --ntasks-per-node=20     # 5 x 20 = 100 rank (ben sotto il limite 256)
#SBATCH --cpus-per-task=1
#SBATCH --mem=0                  # usa TUTTA la RAM del nodo BOOST
#SBATCH --time=15:00:00

echo "=== JOB $SLURM_JOB_ID STARTED at $(date) on $(hostname) ==="

module purge
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.7

source /leonardo_work/IscrC_QuSALa/venv_py311/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Forza UCX a usare i trasporti corretti (InfiniBand e Shared Memory standard)
export UCX_TLS=self,shm,rc,ud
# Aumenta i timeout per dare tempo ai rank di scambiarsi gli indirizzi
export UCX_RECONNECT_WAIT=15s
export UCX_CONNECT_TIMEOUT=300s
# Risolve spesso il problema "Shared memory error" su architetture NVIDIA/Atos
export UCX_MEMTYPE_CACHE=n

cd /leonardo_work/IscrC_QuSALa/vqt2/variational_quantum_transformer_superposition || exit 1
mkdir -p logs

srun --mpi=pmix_v3 python3 -m vqt.scripts.main_hpc
