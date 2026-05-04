from __future__ import annotations

import os
import socket
import sys
import traceback
from dataclasses import dataclass

# HPC-friendly defaults (can be overridden by env before launch).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import torch
import torch.distributed as dist

torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "1")))

try:
    from mpi4py import MPI  # type: ignore
except Exception:
    MPI = None


@dataclass(frozen=True)
class _MpiInfo:
    rank: int
    world_size: int
    comm: object | None


def _mpi_info() -> _MpiInfo:
    if MPI is None:
        return _MpiInfo(rank=0, world_size=1, comm=None)
    comm = MPI.COMM_WORLD
    return _MpiInfo(rank=int(comm.Get_rank()), world_size=int(comm.Get_size()), comm=comm)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _resolve_local_rank(global_rank: int) -> int:
    for env_name in ("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return int(global_rank)


def _configure_cuda_for_rank(rank: int):
    if not torch.cuda.is_available():
        return
    num_devices = int(torch.cuda.device_count())
    if num_devices <= 0:
        return
    local_rank = _resolve_local_rank(rank)
    device_index = local_rank % num_devices
    torch.cuda.set_device(device_index)


def _ensure_rank_env_from_sources(mpi_rank: int, mpi_world: int):
    if os.getenv("WORLD_SIZE") is None:
        if os.getenv("SLURM_NTASKS"):
            os.environ["WORLD_SIZE"] = os.getenv("SLURM_NTASKS", "1")
        elif mpi_world > 1:
            os.environ["WORLD_SIZE"] = str(int(mpi_world))
    if os.getenv("RANK") is None:
        if os.getenv("SLURM_PROCID"):
            os.environ["RANK"] = os.getenv("SLURM_PROCID", "0")
        else:
            os.environ["RANK"] = str(int(mpi_rank))
    if os.getenv("LOCAL_RANK") is None:
        if os.getenv("SLURM_LOCALID"):
            os.environ["LOCAL_RANK"] = os.getenv("SLURM_LOCALID", "0")
        else:
            os.environ["LOCAL_RANK"] = str(int(_resolve_local_rank(mpi_rank)))


def _resolve_master_addr(mpi: _MpiInfo) -> str:
    if os.getenv("MASTER_ADDR"):
        return str(os.getenv("MASTER_ADDR"))
    if os.getenv("SLURM_LAUNCH_NODE_IPADDR"):
        return str(os.getenv("SLURM_LAUNCH_NODE_IPADDR"))
    if mpi.comm is not None and mpi.world_size > 1:
        local_host = socket.gethostname() if mpi.rank == 0 else None
        bcast = getattr(mpi.comm, "bcast", None)
        if callable(bcast):
            host = bcast(local_host, root=0)
            return str(host)
    return "127.0.0.1"


def _init_torch_distributed(mpi: _MpiInfo) -> tuple[int, int]:
    _ensure_rank_env_from_sources(mpi.rank, mpi.world_size)
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    _configure_cuda_for_rank(rank)

    if world_size <= 1:
        return rank, world_size
    if not dist.is_available():
        raise RuntimeError("torch.distributed non disponibile ma WORLD_SIZE > 1.")

    os.environ.setdefault("MASTER_ADDR", _resolve_master_addr(mpi))
    os.environ.setdefault("MASTER_PORT", os.getenv("QSP_MASTER_PORT", "29500"))
    backend = os.getenv("QSP_HPC_DISTRIBUTED_BACKEND", "auto").strip().lower()
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size


def _print_header(rank: int, world_size: int):
    if rank != 0:
        return
    print("=" * 78)
    print("Quantum Sequence Prediction | main_hpc")
    print("=" * 78)
    print(f"World size:             {world_size}")
    print(f"Rank:                   {rank}")
    print(f"Torch distributed:      {dist.is_initialized()}")
    print(f"OMP_NUM_THREADS:        {os.getenv('OMP_NUM_THREADS', '')}")
    print(f"MKL_NUM_THREADS:        {os.getenv('MKL_NUM_THREADS', '')}")
    print(f"OPENBLAS_NUM_THREADS:   {os.getenv('OPENBLAS_NUM_THREADS', '')}")
    print(f"NUMEXPR_NUM_THREADS:    {os.getenv('NUMEXPR_NUM_THREADS', '')}")
    if torch.cuda.is_available():
        print(f"CUDA device count:      {torch.cuda.device_count()}")
    else:
        print("CUDA device count:      0")
    print("=" * 78)


def _barrier(comm: object | None):
    if comm is None:
        return
    barrier = getattr(comm, "Barrier", None)
    if callable(barrier):
        barrier()


def main():
    mpi = _mpi_info()
    rank = mpi.rank
    world_size = mpi.world_size
    try:
        rank, world_size = _init_torch_distributed(mpi)
    except Exception:
        if mpi.rank == 0:
            print("\nBootstrap distribuito fallito:")
            traceback.print_exc()
        raise

    _print_header(rank, world_size)
    os.environ.setdefault("QSP_HPC_DISTRIBUTED", "1" if world_size > 1 else "0")
    os.environ.setdefault("QSP_HPC_DISTRIBUTED_DATASET", "1")
    os.environ.setdefault("QSP_HPC_DISTRIBUTED_TRAINING", "1")

    exit_code = 0
    try:
        from main import main as base_main

        base_main()
    except Exception:
        exit_code = 1
        print(f"\n[rank {rank}] errore durante l'esecuzione:")
        traceback.print_exc()

    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        finally:
            dist.destroy_process_group()
    _barrier(mpi.comm)

    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
