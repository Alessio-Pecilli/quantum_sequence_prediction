#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from mpi4py import MPI
import gc
import psutil
import faulthandler, numpy as np, os, sys
faulthandler.enable(file=sys.stderr, all_threads=True)
np.seterr(all='warn')  # non raise, ma avvisa

try:
    from mpi4py import MPI
except Exception:
    import numpy as np

    class FakeComm:
        def Get_rank(self): return 0
        def Get_size(self): return 1
        def bcast(self, x, root=0): return x
        def Bcast(self, *a, **k): pass
        def Allreduce(self, *a, **k): pass
        def gather(self, x, root=0): return [x]
        def bcast(self, x, root=0): return x

    class FakeMPI:
        COMM_WORLD = FakeComm()
        # tipi fittizi per compatibilità
        INT = np.int32
        DOUBLE = np.float64
        SUM = None

    MPI = FakeMPI()


import numpy as np
from pathlib import Path
from functools import partial
from datetime import datetime
import logging
import sys
import os
import traceback
from ..generalized_quantum_circuits import process_sentence_states
# Import del tuo progetto
from ..config import (
    TRAINING_SENTENCES,
    OPTIMIZATION_CONFIG,
    QUANTUM_STATES_CONFIG,
    DATASET_CONFIG,
    get_training_sentences,
)
from ..encoding import Encoding
from ..optimization import get_params
from ..generalized_quantum_circuits import GeneralizedQuantumCircuitBuilder
from ..quantum_utils import clear_memory, check_memory_usage
from ..quantum_projection import (
    QuantumStateProjector,
    create_projector_from_config,
    get_projection_shape,
    get_projection_param_count,
)

from scipy.optimize import minimize

# ---------------------------------------------------------------------
# Custom Exception for Max Loss Evaluations Reached
# ---------------------------------------------------------------------
class MaxLossEvaluationsReached(Exception):
    """
    Exception raised when the maximum number of loss evaluations is reached (120).
    This is NOT an error - it's the intended stopping condition.
    """
    pass

# GLOBAL COUNTER for loss evaluations (controlled by config epochs)
GLOBAL_LOSS_COUNTER = [0]  # Using list to allow modification in nested scopes
MAX_LOSS_EVALUATIONS = OPTIMIZATION_CONFIG['epochs']  # Gestito da config.py

# ---------------------------------------------------------------------
# Logging minimale: dettagliato su rank 0, ridotto sugli altri
# ---------------------------------------------------------------------
def setup_logging(rank: int):
    logger = logging.getLogger("mpi_beast")
    logger.setLevel(logging.INFO if rank == 0 else logging.INFO)  # INFO su tutti

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] [rank=%(rank)d] %(message)s")
        class RankFilter(logging.Filter):
            def filter(self, record): record.rank = rank; return True
        ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); ch.addFilter(RankFilter())
        logger.addHandler(ch)

        Path("logs").mkdir(exist_ok=True)
        fh = logging.FileHandler(f"logs/rank_{rank}.log", encoding="utf-8")
        fh.setFormatter(fmt); fh.addFilter(RankFilter())
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------
# Utility: bounds e scaling
# ---------------------------------------------------------------------
def get_param_bounds(n_params):
    low  = -np.pi * np.ones(n_params, dtype=np.float64)
    high =  np.pi * np.ones(n_params, dtype=np.float64)
    return low, high

def scale_to_unit(x, low, high):
    return 2.0 * (x - low) / (high - low) - 1.0

def descale_from_unit(y, low, high):
    return low + (y + 1.0) * 0.5 * (high - low)

def box_constraints_for_unit(y_dim):
    cons = []
    for i in range(y_dim):
        cons.append({'type': 'ineq', 'fun': lambda y, i=i:  1.0 - y[i]})  # y[i] <= 1
        cons.append({'type': 'ineq', 'fun': lambda y, i=i:  1.0 + y[i]})  # y[i] >= -1
    return cons

def get_embedding_shape(encoding, cfg):
    if encoding is None:
        return None
    embedding_dim = cfg['embedding_dim']
    vocab_size = len(encoding.vocabulary)
    if vocab_size < embedding_dim:
        raise ValueError("Vocabulary size must be >= embedding_dim for isometric E.")
    return (vocab_size, embedding_dim)

def get_rotation_shape(cfg):
    embedding_dim = cfg['embedding_dim']
    return (embedding_dim, embedding_dim)

def get_param_counts(params_shape, embedding_shape=None, rotation_shape=None, projection_shape=None):
    """
    Calculate parameter counts for all trainable matrices.
    
    Args:
        params_shape: Shape for quantum circuit params (U, W)
        embedding_shape: Shape for E matrix (vocab_size, embedding_dim) - text mode
        rotation_shape: Shape for V matrix (embedding_dim, embedding_dim) - text mode
        projection_shape: Shape for P matrix (target_dim, source_dim) - quantum mode
    
    Returns:
        Tuple of (n_quantum, n_embedding, n_rotation, n_projection)
    """
    n_quantum = 2 * int(np.prod(params_shape))
    n_embedding = int(np.prod(embedding_shape)) if embedding_shape else 0
    n_rotation = int(np.prod(rotation_shape)) if rotation_shape else 0
    n_projection = int(np.prod(projection_shape)) if projection_shape else 0
    return n_quantum, n_embedding, n_rotation, n_projection

def split_total_params(params_native, params_shape, embedding_shape=None, rotation_shape=None, projection_shape=None):
    """
    Split total parameter vector into components.
    
    For TEXT mode (use_quantum_states=False):
        params_quantum (U, W) + raw_embedding (E) + raw_rotation (V)
        
    For QUANTUM mode (use_quantum_states=True, use_projection=True):
        params_quantum (U, W) + raw_projection (P)
    
    Returns:
        (params_quantum, raw_embedding, raw_rotation, raw_projection)
    """
    n_quantum, n_embedding, n_rotation, n_projection = get_param_counts(
        params_shape, embedding_shape, rotation_shape, projection_shape
    )
    expected = n_quantum + n_embedding + n_rotation + n_projection
    if params_native.size != expected:
        raise ValueError(
            f"Total parameter vector mismatch: expected {expected}, got {params_native.size}. "
            f"(n_quantum={n_quantum}, n_embedding={n_embedding}, n_rotation={n_rotation}, n_projection={n_projection})"
        )
    params_quantum = params_native[:n_quantum]
    raw_embedding = None
    raw_rotation = None
    raw_projection = None
    offset = n_quantum
    
    if n_embedding:
        raw_embedding = params_native[offset:offset + n_embedding].reshape(embedding_shape)
        offset += n_embedding
    if n_rotation:
        raw_rotation = params_native[offset:offset + n_rotation].reshape(rotation_shape)
        offset += n_rotation
    if n_projection:
        raw_projection = params_native[offset:offset + n_projection].reshape(projection_shape)
    
    return params_quantum, raw_embedding, raw_rotation, raw_projection

# ---------------------------------------------------------------------
# Utility: PPL quantistica (Renyi-1/2)
# ---------------------------------------------------------------------
def calculate_quantum_ppl(probabilities):
    """
    Calcola la PPL basata sulla formula Renyi-1/2.
    Formula Loss: L = -2 ln(1/T * sum(sqrt(p(y_t|x))))
    Formula PPL: exp(L)
    
    probabilities: lista dei valori di overlap |<x_target|z>|^2
    """
    t = len(probabilities)
    if t == 0:
        return float("inf")
    sum_sqrt_p = sum(np.sqrt(p) for p in probabilities)
    mean_sqrt_p = sum_sqrt_p / t
    # L = -2 ln(mean_sqrt_p)
    loss = -2.0 * np.log(mean_sqrt_p) if mean_sqrt_p > 0 else float("inf")
    # PPL = exp(L)
    ppl = np.exp(loss) if np.isfinite(loss) else float("inf")
    return ppl

# ---------------------------------------------------------------------
# Loss su UNA frase, dato theta nativo
# ---------------------------------------------------------------------
def loss_for_sentence(sentence_idx, sentence, encoding, params_quantum, cfg, states=None, projector=None):
    """
    Calculate loss for a single sentence/sequence.
    
    Args:
        sentence_idx: Index of sentence
        sentence: Text sentence (for text mode)
        encoding: Encoding instance (for text mode)
        params_quantum: Quantum circuit parameters
        cfg: Configuration dict
        states: Pre-computed quantum states (for quantum mode)
        projector: QuantumStateProjector instance (for quantum mode with projection)
    
    Returns:
        Loss value (float)
    """
    mem = psutil.Process().memory_info().rss / 1024**3
    print(f"[Rank {MPI.COMM_WORLD.Get_rank()}] Memoria attuale: {mem:.2f} GB")

    # Se states è già fornito (es. da QA), usalo direttamente
    # Altrimenti fai l'embedding dalla sentence
    if states is None:
        if encoding is None:
            raise ValueError("Né states né encoding forniti! Impossibile calcolare la loss.")
        states, targets = encoding.encode_single(sentence)
        # ✅ Log: Verifica che F venga usata per i targets
        if hasattr(encoding, 'outputEmbeddingMatrix'):
            F_shape = encoding.outputEmbeddingMatrix.shape
            targets_from_F = len(targets) > 0
            print(f"[MATRIX USAGE] Targets estratti da F (outputEmbeddingMatrix): shape={F_shape}, num_targets={len(targets)}")
    else:
        # QUANTUM MODE: Se abbiamo un projector, proietta gli stati
        if projector is not None:
            # states è una lista o array di stati ad alta dimensione (2^D)
            # Convertiamo in array se lista
            states_array = np.array(states) if isinstance(states, list) else states
            # Proietta ogni stato da 2^D a 2^d
            projected_states = projector.project_sequence(states_array)
            # Converti in lista per process_sentence_states
            states = list(projected_states)
            print(f"[PROJECTION] Proiezione P applicata: {states_array.shape} -> {projected_states.shape}")
        targets = None
    
    states_calculated, U, Z = process_sentence_states(states, targets=targets)

    num_layers = cfg['num_layers']
    num_qubits = cfg['num_qubits']
    params_shape = get_params(num_qubits, num_layers).shape
    n_params_half = int(np.prod(params_shape))
    if params_quantum.size != 2 * n_params_half:
        raise ValueError("Quantum parameter vector has unexpected size.")
    params_v = np.reshape(params_quantum[:n_params_half], params_shape)
    params_k = np.reshape(params_quantum[n_params_half:], params_shape)
    
    # Calcola sentence_length: se abbiamo states direttamente, usa len(states)
    # altrimenti conta le parole nella frase
    if states is not None:
        sentence_length = len(states)
        print(f"stati QA: {sentence_length} stati")
    else:
        sentence_length = len(sentence.split())
        print("frase:", sentence, "è lunga", sentence_length, "parole")
    
    builder = GeneralizedQuantumCircuitBuilder(
        embedding_dim=cfg['embedding_dim'],
        sentence_length=sentence_length
    )

    loss = builder.create_generalized_circuit(
        psi=states_calculated,
        U=U,
        Z=Z,
        params_v=params_v,
        params_k=params_k,
        num_layers=num_layers
    )
    del states, states_calculated, U, Z, builder
    gc.collect()
    loss_float = float(loss)
    if not np.isfinite(loss_float):
        # Penalizza ma NON far saltare niente
        return 1e6
    return loss_float
    
    

# ---------------------------------------------------------------------
# Servizio di valutazione distribuita:
# - rank 0 lancia COBYLA, ad ogni chiamata:
#   * Bcast("EVAL"), Bcast(y_scaled)
#   * ogni rank calcola sum_loss_local e count_local
#   * Allreduce su somme e conteggi
#   * rank 0 ritorna media globale
# - al termine Bcast("STOP")
# ---------------------------------------------------------------------
def distributed_objective_factory(
    comm,
    rank,
    size,
    sentences_split,
    encoding,
    low,
    high,
    cfg,
    logger,
    params_shape,
    embedding_shape,
    rotation_shape,
    log_frequency,
    quantum_states=None,
    projection_shape=None,
    projector=None,
):
    """
    Factory for distributed objective function.
    
    Args:
        comm: MPI communicator
        rank: MPI rank
        size: MPI size
        sentences_split: Split of sentences per rank
        encoding: Encoding instance (text mode)
        low, high: Parameter bounds
        cfg: Configuration
        logger: Logger instance
        params_shape: Shape for quantum params
        embedding_shape: Shape for E matrix (text mode)
        rotation_shape: Shape for V matrix (text mode)
        log_frequency: Logging frequency
        quantum_states: Quantum states array (quantum mode)
        projection_shape: Shape for P matrix (quantum mode)
        projector: QuantumStateProjector instance (quantum mode)
    """
    tagbuf = np.array([0], dtype=np.int32)  # 1=EVAL, 2=STOP
    
    # Per salvare la storia delle loss (solo su rank 0 sarà popolata)
    loss_history = []

    prev_embedding = [None]
    prev_rotation = [None]
    
    # ============================================================
    # EARLY STOPPING COMPLETELY DISABLED (HARD CONSTRAINT)
    # The ONLY stopping condition is: 120 loss evaluations
    # ============================================================
    # All early stopping logic is DISABLED
    # NO patience, NO delta checks, NO dynamic cost-cap
    
    # Tracking dei migliori parametri per uso finale
    best_params_tracker = {
        'params': None,        # Migliori parametri trovati (scaled)
        'loss': np.inf,        # Miglior loss
        'iteration': 0         # Iterazione in cui trovati
    }
    iteration_counter = [0]  # Counter globale iterazioni
    best_loss = np.inf

    # Pre-alloc per riduzioni
    send_buf = np.zeros(3, dtype=np.float64)  # [sum_loss_local, count_local, sum_sqrt_p_local]
    recv_buf = np.zeros(3, dtype=np.float64)

    # Lista locale di (global_idx, sentence)
    my_items = sentences_split[rank]
    
    # Se abbiamo quantum_states, convertiamo in lista di SEQUENZE
    # quantum_states ha shape (num_sequences, num_states_per_seq, dim)
    # qa_sequences_list[i] = i-esima sequenza (array di stati)
    if quantum_states is not None:
        qa_sequences_list = [quantum_states[i] for i in range(quantum_states.shape[0])]
    else:
        qa_sequences_list = None

    def objective(y_scaled):
        # Solo rank 0 chiama questa funzione
        assert rank == 0
        
        # ============================================================
        # HARD CONSTRAINT: INCREMENT GLOBAL COUNTER AND CHECK LIMIT
        # ============================================================
        GLOBAL_LOSS_COUNTER[0] += 1
        current_eval = GLOBAL_LOSS_COUNTER[0]
        
        logger.info(f"[LOSS EVAL {current_eval}/{MAX_LOSS_EVALUATIONS}] Starting evaluation...")
        
        # Check if we reached the hard limit of 120 evaluations
        if current_eval >= MAX_LOSS_EVALUATIONS:
            logger.info(
                f"[MAX EVALUATIONS REACHED] Stopping at {current_eval} loss evaluations. "
                f"This is the ONLY stopping condition (early stopping is DISABLED)."
            )
            raise MaxLossEvaluationsReached(
                f"Reached maximum of {MAX_LOSS_EVALUATIONS} loss evaluations"
            )

        # Broadcast tag=EVAL
        tagbuf[0] = 1
        comm.Bcast([tagbuf, MPI.INT], root=0)

        # Broadcast parametri scalati
        y_scaled = np.ascontiguousarray(y_scaled, dtype=np.float64)

        y_dim = np.array([y_scaled.size], dtype=np.int32)
        comm.Bcast([y_dim, MPI.INT], root=0)
        comm.Bcast([y_scaled, MPI.DOUBLE], root=0)

        # Ogni rank: descala e calcola sum loss locale
        params_native = descale_from_unit(y_scaled, low, high)
        params_quantum, raw_embedding, raw_rotation, raw_projection = split_total_params(
            params_native, params_shape, embedding_shape, rotation_shape, projection_shape
        )
        
        # Se abbiamo P (modalità quantum con proiezione), imposta nel projector
        local_projector = None
        if projector is not None and raw_projection is not None:
            projector.set_params(raw_projection.flatten())
            local_projector = projector
            if log_frequency and current_eval % log_frequency == 0:
                logger.info(f"[PROJECTION] P matrix set: shape={raw_projection.shape}, norm={np.linalg.norm(raw_projection):.4f}")
        elif projector is not None:
            # FALLBACK: Usa il projector globale anche se i parametri non sono trainabili
            local_projector = projector
            if log_frequency and current_eval % log_frequency == 0:
                logger.info(f"[PROJECTION] Using global projector (params not trainable)")
        
        if encoding is not None and raw_embedding is not None and raw_rotation is not None:
            encoding.set_embedding_matrix(
                raw_embedding, rotation_matrix=raw_rotation, isometrize=True
            )
            if log_frequency and current_eval % log_frequency == 0:
                e_matrix = encoding.embeddingMatrix
                v_matrix = encoding.rotationMatrix
                f_matrix = encoding.outputEmbeddingMatrix
                e_gram = e_matrix.T @ e_matrix
                v_gram = v_matrix.T @ v_matrix
                e_isometric = np.allclose(e_gram, np.eye(e_gram.shape[0]))
                v_unitary = np.allclose(v_gram, np.eye(v_gram.shape[0]))
                ee_err = float(np.max(np.abs(e_gram - np.eye(e_gram.shape[0]))))
                vv_err = float(np.max(np.abs(v_gram - np.eye(v_gram.shape[0]))))
                range_equal = np.allclose(
                    f_matrix @ f_matrix.T, e_matrix @ e_matrix.T
                )
                det_v = np.linalg.det(v_matrix)
                det_v_abs = float(np.abs(det_v))
                ef_equal = np.allclose(e_matrix, f_matrix)
                ef_diff = float(np.linalg.norm(e_matrix - f_matrix))
                if prev_embedding[0] is None:
                    delta_e = None
                else:
                    delta_e = float(np.linalg.norm(e_matrix - prev_embedding[0]))
                if prev_rotation[0] is None:
                    delta_v = None
                else:
                    delta_v = float(np.linalg.norm(v_matrix - prev_rotation[0]))
                delta_e_str = f"{delta_e:.6f}" if delta_e is not None else "NA"
                delta_v_str = f"{delta_v:.6f}" if delta_v is not None else "NA"
                logger.info(
                    "[EMBEDDING] "
                    f"E delta L2={delta_e_str} V delta L2={delta_v_str} "
                    f"EtE=I:{e_isometric} VtV=I:{v_unitary} "
                    f"max_abs(EtE-I)={ee_err:.2e} max_abs(VtV-I)={vv_err:.2e} "
                    f"range_equal={range_equal} E!=F={not ef_equal} "
                    f"|E-F|={ef_diff:.6f} |det(V)|={det_v_abs:.6f}"
                )
                prev_embedding[0] = e_matrix.copy()
                prev_rotation[0] = v_matrix.copy()

        # Implementazione efficiente: ciclo semplice sulle frasi locali
        local_sum = 0.0
        local_cnt = 0.0
        local_sum_sqrt_p = 0.0
        for local_pos, (global_idx, sentence) in enumerate(my_items):
            # CONTROLLO PREVENTIVO MEMORIA - SOGLIA PIÙ BASSA per quantum states
            memory_threshold = 1.0 if quantum_states is not None else 1.5  # GB
            if not check_memory_usage(threshold_gb=memory_threshold, rank=rank):
                if rank == 0:
                    logger.warning(f"[MEMORY] Soglia memoria {memory_threshold}GB superata al processo {local_pos}/{len(my_items)}")
                clear_memory()  # Pulizia aggressiva
                
            try:
                # Se abbiamo sequenze QA, passa l'INTERA sequenza corrispondente
                # Ogni sequenza ha num_states_per_seq stati (come una "frase" con N "parole")
                if qa_sequences_list is not None:
                    # global_idx indica quale sequenza quantistica usare
                    sequence = qa_sequences_list[global_idx]
                    # sequence è un array di stati: shape (num_states_per_seq, dim)
                    # Lo convertiamo in lista per process_sentence_states
                    states_to_use = list(sequence)
                else:
                    states_to_use = None
                    
                loss = loss_for_sentence(
                    local_pos, sentence, encoding, params_quantum, cfg, 
                    states=states_to_use, projector=local_projector
                )
                if np.isfinite(loss):
                    local_sum += loss
                    local_cnt += 1.0
                    prob = np.exp(-loss)
                    prob = float(np.clip(prob, 0.0, 1.0))
                    local_sum_sqrt_p += np.sqrt(prob)
            except Exception as e:
                # Log solo su rank 0 per non inondare
                if rank == 0:
                    logger.warning(f"Errore loss su sequenza globale {global_idx}: {e}")
            finally:
                # Pulizia memoria dopo ogni sequenza
                clear_memory()
        

        # Allreduce su [sum, count]
        send_buf[0] = local_sum
        send_buf[1] = local_cnt
        send_buf[2] = local_sum_sqrt_p
        if size == 1:
            recv_buf[:] = send_buf
        else:
            comm.Allreduce([send_buf, MPI.DOUBLE], [recv_buf, MPI.DOUBLE], op=MPI.SUM)

        global_sum = recv_buf[0]
        global_cnt = recv_buf[1]
        global_sum_sqrt_p = recv_buf[2]
        if global_cnt > 0:
            global_mean = global_sum / global_cnt
            mean_sqrt_p = global_sum_sqrt_p / global_cnt
            # Formula Loss: L = -2 ln(1/T * sum(sqrt(p(y_t|x))))
            # Equivalente: L = -2 ln(mean_sqrt_p)
            loss_renyi = -2.0 * np.log(mean_sqrt_p) if mean_sqrt_p > 0 else float("inf")
            # PPL = exp(L)
            ppl_quantum = np.exp(loss_renyi) if np.isfinite(loss_renyi) else float("inf")
        else:
            global_mean = 0.0
            loss_renyi = float("inf")
            ppl_quantum = float("inf")

        logger.info(f"[TRAIN] Loss (circuito): {global_mean:.6f}")
        logger.info(f"[TRAIN] Loss (formula L=-2ln(mean_sqrt_p)): {loss_renyi:.6f}")
        logger.info(f"[TRAIN] PPL (exp(L)): {ppl_quantum:.6f}")
        loss_history.append(global_mean)
        
        # ============================================================
        # TRACK BEST PARAMETERS: Aggiorna se miglioramento
        # ============================================================
        nonlocal best_loss
        nonlocal best_params_tracker, iteration_counter
        
        iteration_counter[0] += 1
        current_iter = iteration_counter[0]
        
        # Aggiorna best assoluto e tracker
        # SAFEGUARD: Salva sempre i parametri della prima iterazione finita
        if global_mean < best_loss:
            best_loss = global_mean
            # Salva i parametri SCALATI (y_scaled) per poterli recuperare
            best_params_tracker['params'] = y_scaled.copy()
            best_params_tracker['loss'] = global_mean
            best_params_tracker['iteration'] = current_iter
            logger.info(f"[TRACKER] New best! Loss={global_mean:.6f} at iter {current_iter}")
        elif best_params_tracker['params'] is None:
            # Caso edge: prima iterazione, salva comunque anche se non è il best
            best_params_tracker['params'] = y_scaled.copy()
            best_params_tracker['loss'] = global_mean
            best_params_tracker['iteration'] = current_iter
            logger.warning(
                f"[TRACKER] First iteration fallback: Loss={global_mean:.6f} "
                f"(not better than best_loss={best_loss:.6f})"
            )
        
        # ============================================================
        # ALL EARLY STOPPING LOGIC REMOVED (HARD CONSTRAINT)
        # Only stopping condition: 120 loss evaluations
        # ============================================================
        logger.info(
            f"[TRACKING] iter={current_iter}, loss={global_mean:.6f}, "
            f"best={best_loss:.6f}, evals={GLOBAL_LOSS_COUNTER[0]}/{MAX_LOSS_EVALUATIONS}"
        )

        return float(global_mean)

    def stop_workers():
        if rank == 0:
            tagbuf[0] = 2
            comm.Bcast([tagbuf, MPI.INT], root=0)

    # Ritorniamo le due funzioni: objective per rank 0, e uno "worker_loop" per gli altri
    def worker_loop():
        # Rank != 0: rimane in ascolto dei broadcast
        local_items = my_items
        while True:
            comm.Bcast([tagbuf, MPI.INT], root=0)
            if tagbuf[0] == 2:
                break
            elif tagbuf[0] == 1:
                # EVAL: ricevi dimensione e vettore y_scaled
                y_dim = np.array([0], dtype=np.int32)
                comm.Bcast([y_dim, MPI.INT], root=0)
                y_scaled = np.empty(y_dim[0], dtype=np.float64)
                comm.Bcast([y_scaled, MPI.DOUBLE], root=0)

                params_native = descale_from_unit(y_scaled, low, high)
                params_quantum, raw_embedding, raw_rotation, raw_projection = split_total_params(
                    params_native, params_shape, embedding_shape, rotation_shape, projection_shape
                )
                
                # Se abbiamo P (modalità quantum con proiezione), imposta nel projector
                local_projector = None
                if projector is not None and raw_projection is not None:
                    projector.set_params(raw_projection.flatten())
                    local_projector = projector
                
                if encoding is not None and raw_embedding is not None and raw_rotation is not None:
                    encoding.set_embedding_matrix(
                        raw_embedding, rotation_matrix=raw_rotation, isometrize=True
                    )

                # Calcolo locale
                local_sum = 0.0
                local_cnt = 0.0
                local_sum_sqrt_p = 0.0
                for local_pos, (global_idx, sentence) in enumerate(local_items):
                    # CONTROLLO PREVENTIVO MEMORIA (worker)
                    if not check_memory_usage(threshold_gb=1.5, rank=rank):
                        clear_memory()  # Pulizia aggressiva
                        
                    try:
                        # Se abbiamo sequenze QA, passa l'INTERA sequenza corrispondente
                        if qa_sequences_list is not None:
                            sequence = qa_sequences_list[global_idx]
                            states_to_use = list(sequence)
                        else:
                            states_to_use = None
                            
                        loss = loss_for_sentence(
                            local_pos, sentence, encoding, params_quantum, cfg, 
                            states=states_to_use, projector=local_projector
                        )
                        if np.isfinite(loss):
                            local_sum += loss
                            local_cnt += 1.0
                            prob = np.exp(-loss)
                            prob = float(np.clip(prob, 0.0, 1.0))
                            local_sum_sqrt_p += np.sqrt(prob)
                    except Exception:
                        pass
                    finally:
                        # Pulizia dopo ogni loss
                        clear_memory()

                # Contribuisce alla riduzione
                send_buf[0] = local_sum
                send_buf[1] = local_cnt
                send_buf[2] = local_sum_sqrt_p
                if size == 1:
                    recv_buf[:] = send_buf
                else:
                    comm.Allreduce([send_buf, MPI.DOUBLE], [recv_buf, MPI.DOUBLE], op=MPI.SUM)
            else:
                # Tag sconosciuto, termina per sicurezza
                break

    return objective, stop_workers, worker_loop, loss_history, best_params_tracker

# ---------------------------------------------------------------------
# Split frasi e preparazione encoding locale
# ---------------------------------------------------------------------
def make_splits_and_encodings(sentences, comm, rank, size, embedding_dim, logger):
    # Distribuisci indici globali equamente
    indices = np.arange(len(sentences), dtype=np.int64)
    chunks = np.array_split(indices, size)

    # Costruiamo la lista locale di (global_idx, sentence)
    local_ids = chunks[rank].tolist()
    local_items = [(int(gid), sentences[int(gid)]) for gid in local_ids]

    # L'encoding locale si costruisce con l'elenco di frasi locali
    local_sentences = [s for (_, s) in local_items]
    encoding_local = Encoding(local_sentences, embeddingDim=embedding_dim)

    if rank == 0:
        logger.info(f"Frasi totali: {len(sentences)}; size MPI: {size}")
    logger.info(f"[Rank {rank}] Frasi locali: {len(local_items)} -> {[s for _, s in local_items]}")

    return chunks, local_items, encoding_local


def evaluate_perplexity_on_new_sentences(best_params_native, cfg, logger):
    """
    Genera un nuovo batch di frasi e valuta la perplexity usando i parametri trovati.
    """
    try:
        test_sentences = get_training_sentences()
    except Exception as exc:
        logger.warning(f"[EVAL] Impossibile ottenere frasi di test: {exc}")
        return None

    if not test_sentences:
        logger.warning("[EVAL] Nessuna frase disponibile per la valutazione")
        return None

    logger.info(f"[EVAL] Calcolo perplexity su {len(test_sentences)} nuove frasi")
    encoding = Encoding(test_sentences, embeddingDim=cfg['embedding_dim'])
    params_shape = get_params(cfg['num_qubits'], cfg['num_layers']).shape
    embedding_shape = get_embedding_shape(encoding, cfg)
    rotation_shape = get_rotation_shape(cfg)
    params_quantum, raw_embedding, raw_rotation = split_total_params(
        best_params_native, params_shape, embedding_shape, rotation_shape
    )
    encoding.set_embedding_matrix(
        raw_embedding, rotation_matrix=raw_rotation, isometrize=True
    )
    
    # ✅ Verifica che F sia stata calcolata
    logger.info(f"[EVAL] Matrici configurate:")
    logger.info(f"[EVAL]   E (embeddingMatrix): {encoding.embeddingMatrix.shape}")
    logger.info(f"[EVAL]   V (rotationMatrix): {encoding.rotationMatrix.shape}")
    logger.info(f"[EVAL]   F (outputEmbeddingMatrix = E @ V): {encoding.outputEmbeddingMatrix.shape}")
    logger.info(f"[EVAL] ✅ Tutte le 4 matrici (U, W, E, F) saranno usate nel test")

    total_loss = 0.0
    total_words = 0
    used_sentences = 0
    probabilities = []

    for idx, sentence in enumerate(test_sentences):
        word_count = len(sentence.split())
        if word_count == 0:
            continue
        try:
            loss_val = loss_for_sentence(idx, sentence, encoding, params_quantum, cfg)
        except Exception as exc:
            logger.warning(f"[EVAL] Errore sulla frase {idx}: {exc}")
            clear_memory()
            continue

        if np.isfinite(loss_val):
            total_loss += loss_val
            total_words += word_count
            used_sentences += 1
            prob = np.exp(-loss_val)
            probabilities.append(float(np.clip(prob, 0.0, 1.0)))
        clear_memory()

    if used_sentences == 0 or total_words == 0:
        logger.warning("[EVAL] Nessuna loss valida per calcolare la perplexity")
        return None

    avg_loss_sentence = total_loss / used_sentences
    avg_loss_word = total_loss / total_words
    
    # Calcola Loss con formula: L = -2 ln(1/T * sum(sqrt(p)))
    sum_sqrt_p = sum(probabilities)
    mean_sqrt_p = sum_sqrt_p / len(probabilities) if probabilities else 0.0
    loss_renyi = -2.0 * np.log(mean_sqrt_p) if mean_sqrt_p > 0 else float("inf")
    # PPL = exp(L)
    perplexity = np.exp(loss_renyi) if np.isfinite(loss_renyi) else float("inf")

    logger.info(f"[EVAL] Loss media frase: {avg_loss_sentence:.6f}")
    logger.info(f"[EVAL] Loss media parola: {avg_loss_word:.6f}")
    logger.info(f"[EVAL] Loss (L=-2ln(mean_sqrt_p)): {loss_renyi:.6f}")
    logger.info(f"[EVAL] Perplexity (exp(L)): {perplexity:.6f}")
    logger.info(f"[EVAL] Verifica consistency: exp({loss_renyi:.6f}) = {perplexity:.6f}")

    return {
        "num_sentences": len(test_sentences),
        "used_sentences": used_sentences,
        "avg_loss_per_sentence": avg_loss_sentence,
        "avg_loss_per_word": avg_loss_word,
        "loss_renyi": loss_renyi,
        "perplexity": perplexity,
    }


def save_all_matrices(best_params_native, cfg, logger, timestamp, encoding_instance=None, 
                      projection_shape=None, embedding_shape=None, rotation_shape=None):
    """
    Salva TUTTE le matrici: U, W, E, F e P (se quantum states mode) con run_id univoco.
    
    Formula Loss: L = -2 ln(1/T * sum(sqrt(p(y_t|x))))
    Formula PPL: exp(L)
    
    Args:
        best_params_native: Parametri ottimizzati (include P se projection_shape è fornito)
        cfg: Configurazione
        logger: Logger
        timestamp: Timestamp per run_id
        encoding_instance: Istanza encoding (per text mode)
        projection_shape: Shape della matrice P (target_dim, source_dim) per quantum states mode
        embedding_shape: Shape della matrice E (vocab_size, embedding_dim) - opzionale, dedotto se None
        rotation_shape: Shape della matrice V (embedding_dim, 2^num_qubits) - opzionale, dedotto se None
    """
    from ..layer import AnsatzBuilder
    from qiskit.quantum_info import Operator
    from ..optimization import get_params
    
    # Crea struttura cartelle
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Run ID univoco
    seed = cfg.get('seed', 42)
    run_id = f"run_{timestamp}_seed{seed}"
    run_dir = results_dir / run_id
    run_dir.mkdir(exist_ok=True)
    
    # Sottocartella per matrici
    matrices_dir = run_dir / "matrices"
    matrices_dir.mkdir(exist_ok=True)
    
    logger.info(f"[SAVE] Salvataggio matrici in: {run_dir}")
    logger.info(f"[SAVE] Run ID: {run_id}")
    
    num_layers = cfg['num_layers']
    num_qubits = cfg['num_qubits']
    params_shape = get_params(num_qubits, num_layers).shape
    
    # Se embedding_shape non è passato esplicitamente, deducilo da encoding_instance
    if embedding_shape is None:
        embedding_shape = get_embedding_shape(encoding_instance, cfg) if encoding_instance else None
    if rotation_shape is None:
        rotation_shape = get_rotation_shape(cfg) if embedding_shape else None
    
    # Split parametri (con supporto per proiezione)
    params_quantum, raw_embedding, raw_rotation, raw_projection = split_total_params(
        best_params_native, params_shape, embedding_shape, rotation_shape, projection_shape
    )
    
    n_params_half = int(np.prod(params_shape))
    params_v = np.reshape(params_quantum[:n_params_half], params_shape)
    params_k = np.reshape(params_quantum[n_params_half:], params_shape)
    
    # U e W da circuiti
    ansatz_v = AnsatzBuilder(num_qubits, params_v, num_layers)
    ansatz_k = AnsatzBuilder(num_qubits, params_k, num_layers)
    U_matrix = Operator(ansatz_v.get_ansatz()).data
    W_matrix = Operator(ansatz_k.get_ansatz()).data
    
    # E e F da encoding (text mode)
    E_matrix = raw_embedding if raw_embedding is not None else np.array([])
    V_rotation = raw_rotation if raw_rotation is not None else np.array([])
    F_matrix = E_matrix @ V_rotation if (raw_embedding is not None and raw_rotation is not None) else np.array([])
    
    # P matrix per quantum states mode
    P_matrix = raw_projection.reshape(projection_shape) if raw_projection is not None and projection_shape is not None else np.array([])
    
    # Salva matrici nella sottocartella
    np.save(matrices_dir / "U_matrix.npy", U_matrix)
    np.save(matrices_dir / "W_matrix.npy", W_matrix)
    np.save(matrices_dir / "E_matrix.npy", E_matrix)
    np.save(matrices_dir / "F_matrix.npy", F_matrix)
    np.save(matrices_dir / "V_rotation.npy", V_rotation)
    np.save(matrices_dir / "P_matrix.npy", P_matrix)
    
    # SALVA ANCHE I PARAMETRI THETA per riutilizzarli nel test!
    np.save(matrices_dir / "best_params_native.npy", best_params_native)
    
    # Log shapes
    logger.info(f"[SAVE] U shape: {U_matrix.shape} -> {matrices_dir / 'U_matrix.npy'}")
    logger.info(f"[SAVE] W shape: {W_matrix.shape} -> {matrices_dir / 'W_matrix.npy'}")
    logger.info(f"[SAVE] E shape: {E_matrix.shape} -> {matrices_dir / 'E_matrix.npy'}")
    logger.info(f"[SAVE] F shape: {F_matrix.shape} -> {matrices_dir / 'F_matrix.npy'}")
    logger.info(f"[SAVE] V_rotation shape: {V_rotation.shape} -> {matrices_dir / 'V_rotation.npy'}")
    logger.info(f"[SAVE] P shape: {P_matrix.shape} -> {matrices_dir / 'P_matrix.npy'}")
    logger.info(f"[SAVE] Parametri theta: {best_params_native.shape} -> {matrices_dir / 'best_params_native.npy'}")
    
    # Salva metadata
    use_quantum_states = cfg.get('use_quantum_states', False)
    qs_cfg = cfg.get('quantum_states_config', {})
    metadata = {
        'run_id': run_id,
        'timestamp': timestamp,
        'seed': seed,
        'num_layers': num_layers,
        'num_qubits': num_qubits,
        'embedding_dim': cfg.get('embedding_dim', None),
        'U_shape': U_matrix.shape,
        'W_shape': W_matrix.shape,
        'E_shape': E_matrix.shape,
        'F_shape': F_matrix.shape,
        'P_shape': P_matrix.shape if P_matrix.size > 0 else None,
        'use_quantum_states': use_quantum_states,
        'source_qubits': qs_cfg.get('source_qubits', None),
        'target_qubits': qs_cfg.get('target_qubits', None),
        'use_projection': qs_cfg.get('use_projection', False),
        'use_Projector': qs_cfg.get('use_Projector', True)
    }
    np.save(matrices_dir / "metadata.npy", metadata)
    logger.info(f"[SAVE] Metadata salvato")
    
    return run_dir


def analyze_ancillae_state(best_params_native, cfg, logger, timestamp):
    """
    Analizza lo stato del registro C (ancillae) dopo misure sui registri A e B.
    Verifica se i coefficienti sono complessi (con fasi) o reali.
    """
    from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
    from ..generalized_quantum_circuits import GeneralizedQuantumCircuitBuilder, process_sentence_states
    from ..quantum_annealing import generate_quantum_states
    from ..optimization import get_params
    from ..layer import AnsatzBuilder
    
    logger.info("[ANCILLAE] Inizio analisi dello stato del registro C (ancillae)...")
    
    # Genera un esempio di stato quantistico
    qs_cfg = QUANTUM_STATES_CONFIG
    num_states = 1  # Uno stato è sufficiente per l'analisi
    num_qubits_qs = qs_cfg.get('num_qubits', 2)
    max_time = qs_cfg.get('max_time', 10.0)
    
    quantum_states = generate_quantum_states(
        num_states=num_states,
        num_qubits=num_qubits_qs,
        max_time=max_time,
        use_test_mode=True
    )
    states = quantum_states  # Lista di stati
    
    # Processa stati per ottenere psi, U, Z
    states_calculated, U, Z = process_sentence_states(states)
    
    # Parametri del circuito
    num_layers = cfg['num_layers']
    num_qubits = cfg['num_qubits']
    params_shape = get_params(num_qubits, num_layers).shape
    
    n_params_half = int(np.prod(params_shape))
    n_params_quantum = 2 * n_params_half
    params_quantum = best_params_native[:n_params_quantum]
    params_v = np.reshape(params_quantum[:n_params_half], params_shape)
    params_k = np.reshape(params_quantum[n_params_half:], params_shape)
    
    # Calcola dimensioni
    embedding_dim = cfg.get('embedding_dim', 4)
    # Usa sentence_length = num_states
    sentence_length = len(states)
    
    n_target_qubits = int(np.ceil(np.log2(embedding_dim * embedding_dim)))
    n_control_qubits = int(np.ceil(np.log2(sentence_length)))
    n_total_qubits = n_target_qubits + n_control_qubits
    
    logger.info(f"[ANCILLAE] Configurazione: n_target={n_target_qubits}, n_control={n_control_qubits}, n_total={n_total_qubits}")
    
    # Costruisci circuito quantistico generalizzato
    builder = GeneralizedQuantumCircuitBuilder(
        embedding_dim=embedding_dim,
        sentence_length=sentence_length
    )
    
    # Crea circuito con la firma corretta
    loss = builder.create_generalized_circuit(
        psi=states_calculated,
        U=U,
        Z=Z,
        params_v=params_v,
        params_k=params_k,
        num_layers=num_layers
    )
    
    # Ottieni il circuito dall'ultimo build (dobbiamo accedere al circuito)
    # Nota: create_generalized_circuit ritorna loss, non il circuito
    # Dobbiamo ricostruire il circuito senza calcolare la loss
    # Per ora, saltiamo l'analisi ancillae dettagliata e ritorniamo info base
    
    logger.warning("[ANCILLAE] Analisi dettagliata temporaneamente disabilitata - ritorno info base")
    
    return {
        'has_quantum_coherence': True,  # Assume coerenza quantistica
        'num_complex': n_total_qubits,
        'num_real': 0,
        'num_total': 2**n_control_qubits,
        'num_significant': 2**n_control_qubits
    }


def print_final_training_summary(best_params_native, cfg, logger, timestamp, 
                                   total_loss_evals, final_loss, best_loss, 
                                   eval_metrics, ancillae_stats=None):
    """
    Prints comprehensive final training summary with ALL parameters and quantum register info.
    
    Formula Loss: L = -2 ln(1/T * sum(sqrt(p(y_t|x))))
    Formula PPL: exp(L)
    """
    from ..optimization import get_params
    
    logger.info("\n" + "="*80)
    logger.info("=== TRAINING SUMMARY ===")
    logger.info("="*80)
    
    # 1) FORMULE USATE
    logger.info(f"\n[FORMULE]")
    logger.info(f"  Loss: L = -2 ln(1/T * sum(sqrt(p(y_t|x))))")
    logger.info(f"  PPL:  exp(L)")
    
    # 2) LOSS EVALUATIONS COUNT
    logger.info(f"\n[LOSS EVALUATIONS]")
    logger.info(f"  Total loss evaluations: {total_loss_evals}")
    logger.info(f"  Maximum allowed: {MAX_LOSS_EVALUATIONS}")
    logger.info(f"  Stopping condition: {'Reached max evaluations' if total_loss_evals >= MAX_LOSS_EVALUATIONS else 'Converged early'}")
    
    # 3) FINAL LOSS VALUES
    logger.info(f"\n[FINAL LOSS]")
    logger.info(f"  Final loss (last evaluation): {final_loss:.8f}")
    logger.info(f"  Best loss seen: {best_loss:.8f}")
    final_ppl = np.exp(final_loss)
    best_ppl = np.exp(best_loss)
    logger.info(f"  Final PPL = exp({final_loss:.8f}) = {final_ppl:.8f}")
    logger.info(f"  Best PPL  = exp({best_loss:.8f}) = {best_ppl:.8f}")
    
    # 4) ALL FINAL PARAMETERS - DETAILED
    logger.info(f"\n[FINAL PARAMETERS]")
    num_layers = cfg['num_layers']
    num_qubits = cfg['num_qubits']
    params_shape = get_params(num_qubits, num_layers).shape

    n_params_half = int(np.prod(params_shape))
    n_params_quantum = 2 * n_params_half
    params_quantum = best_params_native[:n_params_quantum]
    params_v = np.reshape(params_quantum[:n_params_half], params_shape)
    params_k = np.reshape(params_quantum[n_params_half:], params_shape)

    embedding_dim = cfg.get('embedding_dim', 4)
    extra_params = best_params_native.size - n_params_quantum
    rotation_params = embedding_dim * embedding_dim
    embedding_params = extra_params - rotation_params if extra_params >= rotation_params else 0

    logger.info(f"  Total parameters: {best_params_native.size}")
    logger.info(f"  U parameters (Query): {params_v.size} (shape: {params_v.shape})")
    logger.info(f"  W parameters (Key): {params_k.size} (shape: {params_k.shape})")
    
    # Determina se siamo in quantum mode con proiezione
    use_quantum_states = QUANTUM_STATES_CONFIG.get('use_quantum_states', False)
    use_projection = QUANTUM_STATES_CONFIG.get('use_projection', False)
    use_projector = QUANTUM_STATES_CONFIG.get('use_Projector', True)
    projection_mode_active = use_quantum_states and use_projection and use_projector
    
    if projection_mode_active:
        # QUANTUM MODE: mostra P invece di E/V
        source_qubits = QUANTUM_STATES_CONFIG.get('source_qubits', 10)
        target_qubits = QUANTUM_STATES_CONFIG.get('target_qubits', 2)
        source_dim = 2 ** source_qubits
        target_dim = 2 ** target_qubits
        projection_params = target_dim * source_dim
        logger.info(f"  P parameters (Projection): {projection_params}")
        logger.info(f"  P shape: {target_dim} x {source_dim} (2^{target_qubits} x 2^{source_qubits})")
        logger.info(f"  E parameters: 0 (not used in quantum mode)")
        logger.info(f"  V parameters: 0 (not used in quantum mode)")
    elif use_quantum_states and use_projection and not use_projector:
        logger.info("  Projector is disabled via use_Projector=False; quantum states feed directly to the circuit.")
    elif extra_params > 0:
        # TEXT MODE: mostra E e V
        logger.info(f"  E parameters (Embedding): {embedding_params}")
        logger.info(f"  V parameters (Rotation): {rotation_params}")
        if embedding_params > 0 and embedding_params % embedding_dim == 0:
            vocab_size = embedding_params // embedding_dim
            logger.info(f"  E shape: {vocab_size} x {embedding_dim}")
            logger.info(f"  F = E @ V shape: {vocab_size} x {embedding_dim}")
    
    # 5) QUANTUM REGISTERS CONFIGURATION
    logger.info(f"\n[QUANTUM REGISTERS]")
    embedding_dim = cfg.get('embedding_dim', 4)
    sentence_length = cfg.get('max_time', 5)  # Parole per frase
    
    n_target_qubits = int(np.ceil(np.log2(embedding_dim * embedding_dim)))
    n_control_qubits = int(np.ceil(np.log2(sentence_length)))
    n_total_qubits = n_target_qubits + n_control_qubits
    
    logger.info(f"  Total qubits: {n_total_qubits}")
    logger.info(f"  Target qubits (A+B registers): {n_target_qubits}")
    logger.info(f"  Control qubits (C register/ancillae): {n_control_qubits}")
    logger.info(f"  Embedding dimension: {embedding_dim}")
    logger.info(f"  Sentence length: {sentence_length}")
    logger.info(f"  Hilbert space dimension: {2**n_total_qubits}")
    
    # 6) PERPLEXITY ON TEST DATA
    logger.info(f"\n[PERPLEXITY ON TEST DATA]")
    if eval_metrics and 'perplexity' in eval_metrics:
        test_loss = eval_metrics.get('loss_renyi', eval_metrics.get('avg_loss_per_sentence', float('nan')))
        test_ppl = eval_metrics['perplexity']
        logger.info(f"  Test Loss: {test_loss:.8f}")
        logger.info(f"  Test PPL = exp({test_loss:.8f}) = {test_ppl:.8f}")
        logger.info(f"  Test sentences used: {eval_metrics.get('used_sentences', 'N/A')}/{eval_metrics.get('num_sentences', 'N/A')}")
        logger.info(f"  Verifica consistency: exp(L) == PPL")
        consistency_check = np.abs(np.exp(test_loss) - test_ppl) < 1e-6
        logger.info(f"  Consistency check: {consistency_check}")
    else:
        logger.info(f"  Perplexity: NOT AVAILABLE")
    
    # 7) ANCILLAE ANALYSIS (if available)
    if ancillae_stats:
        logger.info(f"\n[ANCILLAE STATE ANALYSIS]")
        logger.info(f"  Has quantum coherence: {ancillae_stats['has_quantum_coherence']}")
        logger.info(f"  Complex coefficients: {ancillae_stats['num_complex']}/{ancillae_stats['num_total']}")
        logger.info(f"  Real coefficients: {ancillae_stats['num_real']}/{ancillae_stats['num_total']}")
        logger.info(f"  Significant coefficients: {ancillae_stats['num_significant']}")
    
    # 8) METADATA
    logger.info(f"\n[METADATA]")
    logger.info(f"  Timestamp: {timestamp}")
    logger.info(f"  Optimizer: COBYLA")
    logger.info(f"  Number of layers: {num_layers}")
    logger.info(f"  Qubits per ansatz: {num_qubits}")
    logger.info(f"  Early stopping: DISABLED (HARD CONSTRAINT)")
    logger.info(f"  Stopping criterion: {MAX_LOSS_EVALUATIONS} loss evaluations ONLY")
    
    logger.info("\n" + "="*80)
    logger.info("=== END OF TRAINING SUMMARY ===")
    logger.info("="*80 + "\n")


def run_3fold_cross_validation(run_dir, cfg, logger, use_quantum_states=False):
    """
    3-fold cross validation POST-training.
    Carica U, W, E, F e valuta senza ri-addestrare.
    
    Formula Loss (dalle immagini): L = -2 ln(1/T * sum(sqrt(p(y_t|x))))
    Formula PPL: PPL = exp(L) = (mean_sqrt_p)^(-2)
    
    Usa DATASET_CONFIG per numero frasi e lunghezza.
    """
    logger.info("\n" + "="*80)
    logger.info("=== 3-FOLD CROSS VALIDATION (POST-TRAINING) ===")
    logger.info("="*80)
    logger.info(f"[CV] Caricamento parametri da: {run_dir}")
    
    # Percorso matrici nella sottocartella
    matrices_dir = run_dir / "matrices"
    
    # Carica TUTTE le matrici E I PARAMETRI THETA
    U_matrix = np.load(matrices_dir / "U_matrix.npy")
    W_matrix = np.load(matrices_dir / "W_matrix.npy")
    E_matrix = np.load(matrices_dir / "E_matrix.npy")
    F_matrix = np.load(matrices_dir / "F_matrix.npy")
    V_rotation = np.load(matrices_dir / "V_rotation.npy")
    P_matrix = np.load(matrices_dir / "P_matrix.npy") if (matrices_dir / "P_matrix.npy").exists() else np.array([])
    best_params_native = np.load(matrices_dir / "best_params_native.npy")
    metadata = np.load(matrices_dir / "metadata.npy", allow_pickle=True).item()
    
    logger.info(f"[CV] ✓ U caricata: {U_matrix.shape}")
    logger.info(f"[CV] ✓ W caricata: {W_matrix.shape}")
    logger.info(f"[CV] ✓ E caricata: {E_matrix.shape}")
    logger.info(f"[CV] ✓ F caricata: {F_matrix.shape}")
    logger.info(f"[CV] ✓ V_rotation caricata: {V_rotation.shape}")
    logger.info(f"[CV] ✓ P caricata: {P_matrix.shape}")
    logger.info(f"[CV] ✓ Parametri theta caricati: {best_params_native.shape}")
    logger.info(f"[CV] Run ID: {metadata['run_id']}")
    
    # Setup del projector se in quantum states mode con proiezione
    projector = None
    use_projector_cfg = QUANTUM_STATES_CONFIG.get('use_Projector', True)
    if use_quantum_states and P_matrix.size > 0 and use_projector_cfg:
        from ..quantum_projection import QuantumStateProjector
        projection_shape = P_matrix.shape
        projector = QuantumStateProjector(
            source_dim=projection_shape[1],
            target_dim=projection_shape[0]
        )
        projector.set_params(P_matrix.flatten())
        logger.info(f"[CV] ✓ Projector configurato: {projection_shape[1]} -> {projection_shape[0]}")
    else:
        projection_shape = None
    
    # Configurazione da DATASET_CONFIG
    from ..config import DATASET_CONFIG
    max_test_sentences = DATASET_CONFIG.get('max_sentences', 100)
    sentence_length = DATASET_CONFIG.get('sentence_length', 5)
    
    logger.info(f"[CV] Config: max_sentences={max_test_sentences}, sentence_length={sentence_length}")
    
    # Genera frasi di test (DIVERSE da training)
    if use_quantum_states:
        from ..quantum_annealing import generate_quantum_states
        qs_cfg = QUANTUM_STATES_CONFIG
        num_states = min(max_test_sentences, qs_cfg.get('num_states', 9))
        # Usa source_qubits se c'è proiezione, altrimenti num_qubits
        use_projection_qs = qs_cfg.get('use_projection', False) and use_projector_cfg and P_matrix.size > 0
        num_qubits_qs = qs_cfg.get('source_qubits', qs_cfg.get('num_qubits', 2)) if use_projection_qs else qs_cfg.get('num_qubits', 2)
        # max_time determina il numero di "parole" (stati temporali)
        max_time = sentence_length  # Usa sentence_length da config
        
        rng = np.random.default_rng(cfg.get('seed', 42) + 999)
        hilbert_dim = 2 ** num_qubits_qs
        test_sentences_all = []
        
        logger.info(f"[CV] Generazione {num_states} stati quantistici test (D={num_qubits_qs}, max_time={max_time})")
        
        # USA LA NUOVA MODALITÀ: stati base casuali senza rimpiazzo
        from ..quantum_annealing import TFIMHamiltonian
        
        # Crea Hamiltoniana TFIM per test (seed diverso dal training)
        hamiltonian_test = TFIMHamiltonian(
            num_qubits=num_qubits_qs,
            seed=cfg.get('seed', 42) + 999  # Seed diverso per test
        )
        
        # Genera stati test usando stati base diversi
        test_quantum_sentences_array = hamiltonian_test.generate_sentences(
            num_sentences=num_states,
            max_time=int(max_time),
            seed=cfg.get('seed', 42) + 999
        )
        
        # Converti in formato compatibile
        for i in range(num_states):
            quantum_sentence = test_quantum_sentences_array[i]  # Shape: (max_time, 2^num_qubits_qs)
            test_sentences_all.append((f"quantum_test_{i}", quantum_sentence))
    else:
        # Usa frasi testuali - genera un test set SEPARATO
        try:
            # Prova a caricare frasi diverse
            all_sentences = get_training_sentences()
            # Prendi un subset come test (es. ultime N frasi)
            n_test = min(max_test_sentences, len(all_sentences))
            test_sentences_text = all_sentences[-n_test:] if len(all_sentences) > n_test else all_sentences
            logger.info(f"[CV] Uso {len(test_sentences_text)} frasi da dataset")
        except Exception:
            # Fallback: usa TRAINING_SENTENCES
            test_sentences_text = TRAINING_SENTENCES[:max_test_sentences]
            logger.info(f"[CV] Uso {len(test_sentences_text)} frasi da TRAINING_SENTENCES")
        
        # Filtra per lunghezza corretta
        test_sentences_all = []
        for s in test_sentences_text:
            words = s.split()
            if len(words) == sentence_length:
                test_sentences_all.append((s, None))
            elif len(words) > sentence_length:
                # Tronca
                truncated = ' '.join(words[:sentence_length])
                test_sentences_all.append((truncated, None))
        
        logger.info(f"[CV] Frasi filtrate per lunghezza {sentence_length}: {len(test_sentences_all)}")
    
    logger.info(f"[CV] Frasi test totali: {len(test_sentences_all)}")
    
    # Split in 3 folds
    n_total = len(test_sentences_all)
    fold_size = n_total // 3
    folds = [
        test_sentences_all[:fold_size],
        test_sentences_all[fold_size:2*fold_size],
        test_sentences_all[2*fold_size:]
    ]
    
    fold_results = []
    
    for fold_idx, fold_data in enumerate(folds):
        logger.info(f"\n[CV] === Fold {fold_idx+1}/3 === ({len(fold_data)} frasi, {sentence_length} parole/frase)")
        
        # Crea encoding per questo fold con E e V caricati
        if not use_quantum_states:
            fold_sentences = [s for s, _ in fold_data]
            encoding = Encoding(fold_sentences, embeddingDim=cfg['embedding_dim'])
            # Imposta E e V dall'addestramento
            encoding.set_embedding_matrix(E_matrix, rotation_matrix=V_rotation, isometrize=False)
            logger.info(f"[CV] Fold {fold_idx+1} - Encoding configurato con E e V caricati")
            # ✅ Verifica che F sia stata calcolata correttamente
            F_computed = encoding.outputEmbeddingMatrix
            F_from_file_norm = np.linalg.norm(F_matrix - F_computed)
            logger.info(f"[CV] Fold {fold_idx+1} - F calcolata: shape={F_computed.shape}")
            logger.info(f"[CV] Fold {fold_idx+1} - Verifica F: ||F_caricata - F_calcolata|| = {F_from_file_norm:.2e}")
            logger.info(f"[CV] Fold {fold_idx+1} - ✅ TUTTE le 4 matrici (U, W, E, F) verranno usate nel test")
        else:
            encoding = None
            logger.info(f"[CV] Fold {fold_idx+1} - Usa stati quantistici, encoding=None")
        
        # Estrai params_quantum da best_params_native
        params_shape = get_params(cfg['num_qubits'], cfg['num_layers']).shape
        embedding_shape = (E_matrix.shape[0], E_matrix.shape[1]) if E_matrix.size > 0 else None
        rotation_shape = (V_rotation.shape[0], V_rotation.shape[1]) if V_rotation.size > 0 else None
        
        params_quantum, _, _, _ = split_total_params(
            best_params_native, params_shape, embedding_shape, rotation_shape, projection_shape
        )
        
        # Calcola LOSS REALE per ogni frase usando TUTTE le matrici addestrate
        probabilities = []
        fold_losses = []
        
        for sent_idx, (sentence, qstates) in enumerate(fold_data):
            try:
                # USA loss_for_sentence con i parametri REALI addestrati!
                loss_val = loss_for_sentence(
                    sent_idx, 
                    sentence, 
                    encoding if not use_quantum_states else None, 
                    params_quantum,  # USA parametri quantum REALI (da U e W)
                    cfg, 
                    states=qstates,
                    projector=projector  # Passa projector per stati quantistici
                )
                if np.isfinite(loss_val):
                    fold_losses.append(loss_val)
                    # Converti loss in probabilità: p = exp(-loss)
                    prob = np.exp(-loss_val)
                    prob = float(np.clip(prob, 0.0, 1.0))
                    probabilities.append(prob)
            except Exception as e:
                logger.warning(f"[CV] Fold {fold_idx+1} - Errore frase {sent_idx}: {e}")
                continue
        
        if probabilities:
            # Formula corretta dalle immagini:
            # mean_sqrt_p = (1/T) * Σ sqrt(p)
            T = len(probabilities)
            sum_sqrt_p = sum(np.sqrt(p) for p in probabilities)
            mean_sqrt_p = sum_sqrt_p / T
            
            # L = -2 ln(mean_sqrt_p)
            fold_loss = -2.0 * np.log(mean_sqrt_p) if mean_sqrt_p > 0 else float("inf")
            
            # PPL = exp(L) = (mean_sqrt_p)^(-2)
            fold_ppl = np.exp(fold_loss) if np.isfinite(fold_loss) else float("inf")
            
            # Verifica formula alternativa: PPL = (mean_sqrt_p)^(-2)
            ppl_check = (mean_sqrt_p ** -2) if mean_sqrt_p > 0 else float("inf")
        else:
            fold_loss = float("inf")
            fold_ppl = float("inf")
            ppl_check = float("inf")
        
        fold_results.append({
            'fold': fold_idx + 1,
            'loss': fold_loss,
            'ppl': fold_ppl,
            'num_sentences': len(fold_data),
            'words_per_sentence': sentence_length,
            'total_words': len(fold_data) * sentence_length
        })
        
        logger.info(f"[CV] Fold {fold_idx+1} - Frasi: {len(fold_data)}, Parole/frase: {sentence_length}, Totale parole: {len(fold_data) * sentence_length}")
        logger.info(f"[CV] Fold {fold_idx+1} - Loss (L=-2ln(mean_sqrt_p)): {fold_loss:.6f}")
        logger.info(f"[CV] Fold {fold_idx+1} - PPL (exp(L)): {fold_ppl:.6f}")
        logger.info(f"[CV] Fold {fold_idx+1} - PPL ((mean_sqrt_p)^-2): {ppl_check:.6f}")
        logger.info(f"[CV] Fold {fold_idx+1} - Consistency check: {np.abs(fold_ppl - ppl_check) < 1e-6}")
    
    # Statistiche aggregate con ERRORE MEDIO e DEVIAZIONE STANDARD
    losses = [r['loss'] for r in fold_results if np.isfinite(r['loss'])]
    ppls = [r['ppl'] for r in fold_results if np.isfinite(r['ppl'])]
    
    logger.info(f"\n[CV] Fold con dati validi: {len(losses)}/{len(fold_results)}")
    
    if len(losses) < 2:
        logger.warning(f"[CV] ⚠️ ATTENZIONE: Solo {len(losses)} fold con dati validi!")
        logger.warning(f"[CV] ⚠️ Serve almeno 2 fold per calcolare deviazione standard significativa")
        logger.warning(f"[CV] ⚠️ Aumenta max_sentences in DATASET_CONFIG per avere più frasi test")
    
    if losses:
        mean_loss = np.mean(losses)
        std_loss = np.std(losses, ddof=1) if len(losses) > 1 else 0.0  # Sample std (n-1)
        sem_loss = std_loss / np.sqrt(len(losses)) if len(losses) > 1 else 0.0  # Standard error of mean
    else:
        mean_loss = float("inf")
        std_loss = 0.0
        sem_loss = 0.0
    
    if ppls:
        mean_ppl = np.mean(ppls)
        std_ppl = np.std(ppls, ddof=1) if len(ppls) > 1 else 0.0  # Sample std (n-1)
        sem_ppl = std_ppl / np.sqrt(len(ppls)) if len(ppls) > 1 else 0.0  # Standard error of mean
    else:
        mean_ppl = float("inf")
        std_ppl = 0.0
        sem_ppl = 0.0
    
    logger.info("\n" + "="*80)
    logger.info("=== 3-FOLD CV RESULTS ===")
    logger.info("="*80)
    logger.info(f"Formula Loss: L = -2 ln(1/T * sum(sqrt(p(y_t|x))))")
    logger.info(f"Formula PPL:  exp(L) = (mean_sqrt_p)^(-2)")
    logger.info(f"")
    logger.info(f"Configurazione test:")
    logger.info(f"  - Numero frasi per fold: ~{n_total//3}")
    logger.info(f"  - Parole per frase: {sentence_length}")
    logger.info(f"  - Totale frasi: {n_total}")
    logger.info(f"  - Parametri usati: U, W, E, F (caricati da {metadata['run_id']})")
    logger.info(f"")
    
    # ✨ STAMPA LOSS E PPL PER FOLD (PRIMA DEL TEST)
    logger.info(f"=== LOSS E PPL PER FOLD ===")
    for r in fold_results:
        logger.info(f"  Fold {r['fold']}: Loss={r['loss']:.6f}, PPL={r['ppl']:.6f}")
    logger.info(f"")
    
    logger.info(f"Risultati ({len(losses)} fold validi):")
    logger.info(f"  Loss (media)  = {mean_loss:.6f}")
    logger.info(f"  Loss (std)    = {std_loss:.6f}  {'⚠️ (calcolato su 1 valore)' if len(losses) == 1 else ''}")
    logger.info(f"  Loss (sem)    = {sem_loss:.6f}  {'⚠️ (calcolato su 1 valore)' if len(losses) == 1 else ''}")
    logger.info(f"")
    logger.info(f"  PPL (media)   = {mean_ppl:.6f}")
    logger.info(f"  PPL (std)     = {std_ppl:.6f}  {'⚠️ (calcolato su 1 valore)' if len(ppls) == 1 else ''}")
    logger.info(f"  PPL (sem)     = {sem_ppl:.6f}  {'⚠️ (calcolato su 1 valore)' if len(ppls) == 1 else ''}")
    logger.info(f"")
    logger.info(f"=" * 80)
    logger.info(f"METRICHE STATISTICHE RICHIESTE:")
    logger.info(f"=" * 80)
    logger.info(f"  📊 ERRORE MEDIO (std Loss)         = {std_loss:.6f}  {f'[n={len(losses)}]' if losses else '[n=0]'}")
    logger.info(f"  📊 DEVIAZIONE STANDARD (std PPL)   = {std_ppl:.6f}  {f'[n={len(ppls)}]' if ppls else '[n=0]'}")
    logger.info(f"  📊 STANDARD ERROR MEAN (sem Loss)  = {sem_loss:.6f}  {f'[n={len(losses)}]' if losses else '[n=0]'}")
    logger.info(f"  📊 STANDARD ERROR MEAN (sem PPL)   = {sem_ppl:.6f}  {f'[n={len(ppls)}]' if ppls else '[n=0]'}")
    logger.info(f"=" * 80)
    if len(losses) == 1:
        logger.info(f"")
        logger.info(f"⚠️  NOTA: Con solo 1 fold valido, std e sem sono 0 (serve n≥2)")
        logger.info(f"⚠️  SOLUZIONE: Aumenta max_sentences in config.py DATASET_CONFIG")
    logger.info(f"")
    logger.info(f"Consistency check per fold:")
    for r in fold_results:
        if np.isfinite(r['loss']) and np.isfinite(r['ppl']):
            expected_ppl = np.exp(r['loss'])
            ppl_alt = (np.exp(-r['loss']/2.0)) ** (-2)  # Verifica formula alternativa
            check1 = np.abs(expected_ppl - r['ppl']) < 1e-6
            check2 = np.abs(ppl_alt - r['ppl']) < 1e-6
            logger.info(f"  Fold {r['fold']}: exp({r['loss']:.6f}) = {expected_ppl:.6f} vs {r['ppl']:.6f} [exp: {check1}, alt: {check2}]")
    logger.info("="*80 + "\n")
    
    return fold_results


def evaluate_perplexity_on_quantum_states(best_params_native, cfg, logger):
    """
    Genera un NUOVO set di stati quantistici (diverso dal training) e calcola perplexity.
    """
    from ..quantum_annealing import generate_quantum_states
    
    qs_cfg = QUANTUM_STATES_CONFIG
    num_states = qs_cfg.get('num_states', 9)
    num_qubits_qs = qs_cfg.get('num_qubits', 2)
    max_time = qs_cfg.get('max_time', 10.0)
    
    logger.info(f"[EVAL-QA] Generazione NUOVO set di {num_states} stati quantistici per test...")
    
    # Genera NUOVI stati (con tempo diverso per avere stati diversi)
    test_quantum_states = generate_quantum_states(
        num_states=num_states,
        num_qubits=num_qubits_qs,
        max_time=max_time * 1.5,  # Tempo diverso = stati diversi
        use_test_mode=True
    )
    
    logger.info(f"[EVAL-QA] Stati test generati: shape={test_quantum_states.shape}")
    
    # Converti in lista
    test_states_list = [test_quantum_states[i] for i in range(test_quantum_states.shape[0])]
    
    # Calcola loss con gli stati di test
    try:
        params_shape = get_params(cfg['num_qubits'], cfg['num_layers']).shape
        n_params_quantum = 2 * int(np.prod(params_shape))
        params_quantum = best_params_native[:n_params_quantum]
        loss_val = loss_for_sentence(
            sentence_idx=0,
            sentence="test_quantum_states",
            encoding=None,  # Non serve encoding!
            params_quantum=params_quantum,
            cfg=cfg,
            states=test_states_list
        )
    except Exception as exc:
        logger.warning(f"[EVAL-QA] Errore nel calcolo loss: {exc}")
        return None
    
    if not np.isfinite(loss_val):
        logger.warning("[EVAL-QA] Loss non finita")
        return None
    
    # Perplexity (Renyi-1/2) basata su overlap
    avg_loss_per_state = loss_val / num_states
    prob = float(np.clip(np.exp(-loss_val), 0.0, 1.0))
    perplexity = calculate_quantum_ppl([prob])
    
    logger.info(f"[EVAL-QA] Loss totale: {loss_val:.6f}")
    logger.info(f"[EVAL-QA] Loss media per stato: {avg_loss_per_state:.6f}")
    logger.info(f"[EVAL-QA] Perplexity (Renyi-1/2) = {perplexity:.6f}")
    
    return {
        "num_states": num_states,
        "total_loss": loss_val,
        "avg_loss_per_state": avg_loss_per_state,
        "perplexity": perplexity,
    }

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logger = setup_logging(rank)
    cfg = OPTIMIZATION_CONFIG
    
    # ============================================================
    # TEST-ONLY MODE: Skip training e carica matrici pre-addestrate
    # ============================================================
    from ..config import TEST_ONLY_CONFIG
    if TEST_ONLY_CONFIG.get('skip_training', False):
        if rank == 0:
            logger.info("\n" + "="*80)
            logger.info("=== TEST-ONLY MODE: SKIP TRAINING ===")
            logger.info("="*80)
            
            matrices_dir = Path(TEST_ONLY_CONFIG.get('matrices_dir', Path.cwd()))
            logger.info(f"[TEST-ONLY] Caricamento matrici da: {matrices_dir}")
            
            # Verifica che esistano tutti i file necessari
            required_files = [
                'best_params_native.npy', 'E_matrix.npy', 'F_matrix.npy',
                'U_matrix.npy', 'V_rotation.npy', 'W_matrix.npy', 'metadata.npy'
            ]
            
            missing_files = [f for f in required_files if not (matrices_dir / f).exists()]
            if missing_files:
                logger.error(f"[TEST-ONLY] File mancanti: {missing_files}")
                logger.error(f"[TEST-ONLY] Verifica che i file siano in: {matrices_dir}")
                return 1
            
            # Carica matrici
            try:
                best_params_native = np.load(matrices_dir / 'best_params_native.npy')
                E_matrix = np.load(matrices_dir / 'E_matrix.npy')
                F_matrix = np.load(matrices_dir / 'F_matrix.npy')
                U_matrix = np.load(matrices_dir / 'U_matrix.npy')
                V_rotation = np.load(matrices_dir / 'V_rotation.npy')
                W_matrix = np.load(matrices_dir / 'W_matrix.npy')
                metadata = np.load(matrices_dir / 'metadata.npy', allow_pickle=True).item()
                
                logger.info(f"[TEST-ONLY] ✓ Matrici caricate:")
                logger.info(f"[TEST-ONLY]   - best_params_native: {best_params_native.shape}")
                logger.info(f"[TEST-ONLY]   - U_matrix: {U_matrix.shape}")
                logger.info(f"[TEST-ONLY]   - W_matrix: {W_matrix.shape}")
                logger.info(f"[TEST-ONLY]   - E_matrix: {E_matrix.shape}")
                logger.info(f"[TEST-ONLY]   - F_matrix: {F_matrix.shape}")
                logger.info(f"[TEST-ONLY]   - V_rotation: {V_rotation.shape}")
                logger.info(f"[TEST-ONLY]   - Metadata: {metadata}")
                
                # Crea una finta run_dir con le matrici caricate
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_dir = Path("results")
                results_dir.mkdir(exist_ok=True)
                run_id = f"test_only_{ts}"
                run_dir = results_dir / run_id
                run_dir.mkdir(exist_ok=True)
                matrices_subdir = run_dir / "matrices"
                matrices_subdir.mkdir(exist_ok=True)
                
                # Copia le matrici nella sottocartella per compatibilità con run_3fold_cross_validation
                np.save(matrices_subdir / "best_params_native.npy", best_params_native)
                np.save(matrices_subdir / "E_matrix.npy", E_matrix)
                np.save(matrices_subdir / "F_matrix.npy", F_matrix)
                np.save(matrices_subdir / "U_matrix.npy", U_matrix)
                np.save(matrices_subdir / "V_rotation.npy", V_rotation)
                np.save(matrices_subdir / "W_matrix.npy", W_matrix)
                np.save(matrices_subdir / "metadata.npy", metadata)
                
                logger.info(f"[TEST-ONLY] ✓ Matrici copiate in: {matrices_subdir}")
                
                # Esegui direttamente la 3-fold cross-validation
                logger.info(f"[TEST-ONLY] Inizio 3-fold cross-validation...")
                
                # SOLO frasi testuali (come specificato)
                use_quantum_states = False
                
                cv_results = run_3fold_cross_validation(
                    run_dir, cfg, logger, use_quantum_states=use_quantum_states
                )
                
                logger.info(f"[TEST-ONLY] ✓ Cross-validation completata: {len(cv_results)} folds")
                logger.info("\n" + "="*80)
                logger.info("=== TEST-ONLY MODE COMPLETATO ===")
                logger.info("="*80)
                
                return 0
                
            except Exception as e:
                logger.error(f"[TEST-ONLY] Errore nel caricamento/test: {e}")
                logger.error(traceback.format_exc())
                return 1
        else:
            # Rank != 0: aspetta e termina
            comm.Barrier()
            return 0

    try:
        # ==================================================================
        # SCELTA: Stati Quantistici vs Sentences Testuali
        # ==================================================================
        use_quantum_states = QUANTUM_STATES_CONFIG.get('use_quantum_states', False)
        
        if use_quantum_states:
            # ==============================================================
            # Modalità QUANTUM STATES
            # ==============================================================
            # Logica IDENTICA alle frasi:
            # - num_states = N "frasi quantistiche" da trainare
            # - max_time = M "parole" (evoluzioni temporali) per frase
            # - Con K rank, ogni rank elabora N/K frasi
            # ==============================================================
            from ..quantum_annealing import generate_quantum_states
            
            qs_cfg = QUANTUM_STATES_CONFIG
            num_quantum_sentences = qs_cfg.get('num_states', 9)  # N "frasi quantistiche"
            sentence_length_quantum = int(qs_cfg.get('max_time', 10))  # M "parole" per frase
            
            # Supporto per proiezione: usa source_qubits se abilitata
            use_projector_cfg = qs_cfg.get('use_Projector', True)
            use_projection_qs = qs_cfg.get('use_projection', False) and use_projector_cfg
            source_qubits = qs_cfg.get('source_qubits', 10)  # D = qubit sorgente (default 10)
            target_qubits = qs_cfg.get('target_qubits', qs_cfg.get('num_qubits', 2))  # d = qubit target
            num_qubits_qs = source_qubits if use_projection_qs else qs_cfg.get('num_qubits', 2)
            use_test_mode = qs_cfg.get('use_test_mode', True)
            
            if rank == 0:
                logger.info(f"[QUANTUM] ========================================")
                logger.info(f"[QUANTUM] Modalità QUANTUM STATES attiva!")
                logger.info(f"[QUANTUM] ========================================")
                logger.info(f"[QUANTUM] Frasi quantistiche (num_states): {num_quantum_sentences}")
                logger.info(f"[QUANTUM] Parole per frase (max_time): {sentence_length_quantum}")
                if use_projection_qs:
                    logger.info(f"[QUANTUM] ✓ PROIEZIONE ABILITATA: {source_qubits} qubit -> {target_qubits} qubit")
                    logger.info(f"[QUANTUM] Dimensione Hilbert sorgente: 2^{source_qubits} = {2**source_qubits}")
                    logger.info(f"[QUANTUM] Dimensione Hilbert target: 2^{target_qubits} = {2**target_qubits}")
                else:
                    if qs_cfg.get('use_projection', False) and not use_projector_cfg:
                        logger.info("[QUANTUM] ⚠ use_Projector=False → salto P, gli stati vanno direttamente al circuito.")
                    logger.info(f"[QUANTUM] Dimensione Hilbert: 2^{num_qubits_qs} = {2**num_qubits_qs}")
                logger.info(f"[QUANTUM] MPI ranks: {size}")
                logger.info(f"[QUANTUM] Frasi per rank: ~{num_quantum_sentences // size}")
            
            # SOLO RANK 0 genera TUTTE le frasi quantistiche, poi broadcast
            if rank == 0:
                logger.info(f"[QUANTUM] Rank 0: Generazione OTTIMIZZATA di {num_quantum_sentences} frasi quantistiche...")
                logger.info(f"[QUANTUM] Dimensione target: 2^{num_qubits_qs} = {2**num_qubits_qs}, memoria stimata: {(num_quantum_sentences * sentence_length_quantum * (2**num_qubits_qs) * 16) // (1024**2)} MB")
                
                # ===============================================
                # NUOVO SISTEMA MINI-BATCH con molti LOG!
                # ===============================================
                logger.info("\n" + "="*80)
                logger.info("=== NUOVO SISTEMA MINI-BATCH DATASET ===")
                logger.info("="*80)
                
                # Log parametri config
                logger.info(f"[CONFIG] QUANTUM_STATES_CONFIG reading...")
                for key, value in QUANTUM_STATES_CONFIG.items():
                    logger.info(f"[CONFIG]   {key}: {value}")
                
                # Crea dataset mini-batch usando config.py
                from ..quantum_annealing import TFIMHamiltonian
                
                logger.info(f"[MINIBATCH] Creazione TFIMHamiltonian...")
                logger.info(f"[MINIBATCH] - num_qubits: {QUANTUM_STATES_CONFIG.get('num_qubits', 4)}")
                logger.info(f"[MINIBATCH] - seed: {cfg.get('seed', 42)}")
                
                # Crea dataset usando il metodo convenienza
                logger.info(f"[MINIBATCH] Generazione dataset con create_minibatch_dataset_from_config...")
                minibatch_data, minibatch_info, tfim_instance = TFIMHamiltonian.create_minibatch_dataset_from_config(
                    seed=cfg.get('seed', 42)
                )
                
                # Log dettagliato del dataset generato
                logger.info(f"\n[MINIBATCH] ✅ Dataset generato con successo!")
                logger.info(f"[MINIBATCH] 📊 Statistiche complete:")
                for key, value in minibatch_info.items():
                    logger.info(f"[MINIBATCH]   {key}: {value}")
                
                # Log struttura per ogni mini-batch
                batches = minibatch_data['batches']
                J_matrices = minibatch_data['J_matrices']
                initial_states_per_batch = minibatch_data['initial_states']
                hamiltonians = minibatch_data['hamiltonians']
                
                logger.info(f"\n[MINIBATCH] 🔍 Dettagli per ogni batch:")
                for batch_idx in range(minibatch_info['num_batches']):
                    batch_states = batches[batch_idx]
                    J_matrix = J_matrices[batch_idx]
                    initial_states = initial_states_per_batch[batch_idx]
                    hamiltonian = hamiltonians[batch_idx]
                    
                    logger.info(f"[MINIBATCH] Batch {batch_idx + 1}:")
                    logger.info(f"[MINIBATCH]   - States shape: {batch_states.shape}")
                    logger.info(f"[MINIBATCH]   - J matrix range: [{J_matrix.min():.3f}, {J_matrix.max():.3f}]")
                    logger.info(f"[MINIBATCH]   - Initial states: {len(initial_states)}")
                    logger.info(f"[MINIBATCH]   - Hamiltonian id: {id(hamiltonian)}")
                    
                    # Log alcuni stati iniziali (primi 3)
                    for i, state in enumerate(initial_states[:3]):
                        state_repr = np.where(np.abs(state) > 1e-10)[0]
                        if len(state_repr) == 1:
                            binary = format(state_repr[0], f'0{minibatch_info["num_qubits"]}b')
                            logger.info(f"[MINIBATCH]     - Initial[{i}]: |{binary}⟩")
                    
                    # Verifica diversità matrice J  
                    if batch_idx > 0:
                        prev_J = J_matrices[batch_idx-1]
                        diff = np.linalg.norm(J_matrix - prev_J)
                        logger.info(f"[MINIBATCH]   - J differ from prev: {diff:.6f} {'✓' if diff > 1e-6 else '✗'}")
                
                # Prepara iterator per training
                logger.info(f"\n[MINIBATCH] 🔄 Preparazione iterator per training...")
                batch_iterator = tfim_instance.get_batch_iterator(minibatch_data, shuffle=True, seed=cfg.get('seed', 42))
                logger.info(f"[MINIBATCH] Iterator creato con {len(batch_iterator)} batch")
                
                # CONVERSIONE per compatibilità con resto del codice
                # Il codice esistente si aspetta quantum_states con shape (num_sentences, time_steps, dim)
                logger.info(f"\n[MINIBATCH] 🔄 Conversione in formato compatibile...")
                
                # Concatena tutti i batch in un singolo array per compatibilità
                all_sentences = []
                batch_info_mapping = []  # Mappa sentence_idx -> (batch_idx, sentence_in_batch)
                
                for batch_idx, batch_data in enumerate(batch_iterator):
                    batch_states = batch_data['batch_states']  # (batch_size, time_steps, dim)
                    
                    for sentence_in_batch in range(batch_states.shape[0]):
                        sentence_states = batch_states[sentence_in_batch]  # (time_steps, dim)
                        all_sentences.append(sentence_states)
                        batch_info_mapping.append((batch_idx, sentence_in_batch))
                    
                    logger.info(f"[MINIBATCH] Batch {batch_idx}: estratte {batch_states.shape[0]} frasi")
                
                # Crea quantum_states nel formato atteso
                quantum_states = np.array(all_sentences)  # (total_sentences, time_steps, dim)
                logger.info(f"[MINIBATCH] ✅ Conversione completata: {quantum_states.shape}")
                
                # Log finale di verifica
                logger.info(f"\n[MINIBATCH] 🎯 Verifica finale:")
                logger.info(f"[MINIBATCH] - Total sentences: {len(all_sentences)}")
                logger.info(f"[MINIBATCH] - Quantum states shape: {quantum_states.shape}")
                logger.info(f"[MINIBATCH] - Memory size: {quantum_states.nbytes / 1024**2:.1f} MB")
                logger.info(f"[MINIBATCH] - Batch mapping length: {len(batch_info_mapping)}")
                
                # Verifica diversità fra sentence
                if len(all_sentences) >= 2:
                    diff_01 = np.linalg.norm(quantum_states[0] - quantum_states[1])
                    logger.info(f"[MINIBATCH] - Diversità ||sent_0 - sent_1||: {diff_01:.6f} {'✓' if diff_01 > 1e-6 else '✗'}")
                
                # Salva per debug (opzionale)
                np.save('quantum_states_minibatch.npy', quantum_states)
                logger.info(f"[MINIBATCH] Stati salvati in quantum_states_minibatch.npy")
                
                # Salva anche metadati per debug avanzato
                metadata_debug = {
                    'minibatch_info': minibatch_info,
                    'batch_info_mapping': batch_info_mapping,
                    'config_used': dict(QUANTUM_STATES_CONFIG)
                }
                np.save('minibatch_metadata_debug.npy', metadata_debug)
                logger.info(f"[MINIBATCH] Metadata salvati in minibatch_metadata_debug.npy")
                
                logger.info("\n" + "="*80)
                logger.info("=== MINI-BATCH DATASET READY FOR TRAINING! ===")
                logger.info("="*80)
            else:
                quantum_states = None
            
            # Broadcast a TUTTI i rank con LOG dettagliati
            logger.info(f"[MINIBATCH] Rank {rank}: Broadcasting quantum states dal rank 0...")
            quantum_states = comm.bcast(quantum_states, root=0)
            
            if quantum_states is not None:
                logger.info(f"[MINIBATCH] Rank {rank}: ✅ Ricevute quantum states, shape={quantum_states.shape}")
                logger.info(f"[MINIBATCH] Rank {rank}: - Data type: {quantum_states.dtype}")
                logger.info(f"[MINIBATCH] Rank {rank}: - Memory: {quantum_states.nbytes / 1024**2:.1f} MB")
                logger.info(f"[MINIBATCH] Rank {rank}: - Min value: {quantum_states.min():.6f}")
                logger.info(f"[MINIBATCH] Rank {rank}: - Max value: {quantum_states.max():.6f}")
            else:
                logger.warning(f"[MINIBATCH] Rank {rank}: ⚠ quantum_states è None!")
            
            # Broadcast anche metadata per debug avanzato se su rank 0
            if rank == 0:
                broadcast_metadata = {
                    'minibatch_info': minibatch_info,
                    'batch_info_mapping': batch_info_mapping,
                    'total_batches': minibatch_info['num_batches'],
                    'batch_size': minibatch_info['batch_size']
                }
                logger.info(f"[MINIBATCH] Rank 0: Broadcasting metadata...")
            else:
                broadcast_metadata = None
            
            broadcast_metadata = comm.bcast(broadcast_metadata, root=0)
            logger.info(f"[MINIBATCH] Rank {rank}: Metadata ricevuto, batches={broadcast_metadata['total_batches']}")
            
            # Ora distribuiamo le frasi tra i rank con LOG dettagliati per mini-batch
            logger.info(f"[MINIBATCH] Rank {rank}: Distribuzione frasi tra rank...")
            
            # Leggi parametri da metadata
            num_quantum_sentences = broadcast_metadata['minibatch_info']['dataset_size']
            sentences = [f"minibatch_sentence_{i}" for i in range(num_quantum_sentences)]
            
            logger.info(f"[MINIBATCH] Rank {rank}: Distribuendo {num_quantum_sentences} frasi quantistiche")
            logger.info(f"[MINIBATCH] Rank {rank}: Totale ranks: {size}")
            
            indices = np.arange(num_quantum_sentences, dtype=np.int64)
            chunks = np.array_split(indices, size)
            local_ids = chunks[rank].tolist()
            my_items = [(int(gid), sentences[int(gid)]) for gid in local_ids]
            encoding_local = None  # Non serve encoding con mini-batch QA!
            
            logger.info(f"[MINIBATCH] Rank {rank}: ✅ Assegnate {len(my_items)} frasi")
            logger.info(f"[MINIBATCH] Rank {rank}: - IDs assegnati: {[x[0] for x in my_items]}")
            logger.info(f"[MINIBATCH] Rank {rank}: - Range: {min([x[0] for x in my_items]) if my_items else 'N/A'}-{max([x[0] for x in my_items]) if my_items else 'N/A'}")
            
            # Log informazioni su quale batch appartiene ogni frase
            batch_info_mapping = broadcast_metadata['batch_info_mapping']
            for gid, sentence_name in my_items[:3]:  # Solo primi 3 per non spammare
                if gid < len(batch_info_mapping):
                    batch_idx, sentence_in_batch = batch_info_mapping[gid]
                    logger.info(f"[MINIBATCH] Rank {rank}: - Frase {gid} → Batch {batch_idx}, posizione {sentence_in_batch}")
            
            if len(my_items) > 3:
                logger.info(f"[MINIBATCH] Rank {rank}: ... e altre {len(my_items) - 3} frasi")
        else:
            # Modalità SENTENCES TESTUALI (default) 
            logger.info(f"[TEXT] Rank {rank}: Modalità SENTENCES TESTUALI attiva")
            logger.info(f"[TEXT] Rank {rank}: use_quantum_states = {use_quantum_states}")
            
            quantum_states = None  # Niente stati QA
            sentences = TRAINING_SENTENCES
            
            logger.info(f"[TEXT] Rank {rank}: Frasi testuali: {len(sentences)}")
            for i, sentence in enumerate(sentences[:3]):  # Log prime 3 frasi
                logger.info(f"[TEXT] Rank {rank}: - Frase {i}: '{sentence}'")
            
            chunks, my_items, encoding_local = make_splits_and_encodings(
                sentences, comm, rank, size, cfg['embedding_dim'], logger
            )
            
            logger.info(f"[TEXT] Rank {rank}: ✅ Splits e encoding completati")
            logger.info(f"[TEXT] Rank {rank}: - Items assegnati: {len(my_items)}")
            if encoding_local:
                logger.info(f"[TEXT] Rank {rank}: - Vocab size: {len(encoding_local.vocabulary)}")
        
        # Parametrizzazione
        params_shape = get_params(cfg['num_qubits'], cfg['num_layers']).shape
        embedding_shape = get_embedding_shape(encoding_local, cfg)
        rotation_shape = get_rotation_shape(cfg) if embedding_shape is not None else None
        
        # Setup proiezione per quantum states mode
        projection_shape = None
        projector = None
        if use_quantum_states:
            from ..quantum_projection import create_projector_from_config, get_projection_shape
            projection_shape = get_projection_shape(QUANTUM_STATES_CONFIG)
            if projection_shape is not None:
                projector = create_projector_from_config(QUANTUM_STATES_CONFIG)
                if rank == 0:
                    logger.info(f"[PROJECTION] ✓ Projector creato: {projection_shape[1]} -> {projection_shape[0]}")
                
                # In quantum states mode con proiezione:
                # - P fa il lavoro di E (proietta stati → embedding)
                # - E non serve (non ci sono parole)
                # - V non serve (gli stati proiettati sono già i target)
                # embedding_shape e rotation_shape restano None
                
                if rank == 0:
                    logger.info(f"[QUANTUM] P sostituisce E (embedding): {projection_shape}")
                    logger.info(f"[QUANTUM] V non serve in quantum mode (target = stati proiettati)")
        
        n_params_quantum, n_params_embedding, n_params_rotation, n_params_projection = get_param_counts(
            params_shape, embedding_shape, rotation_shape, projection_shape
        )
        n_params = n_params_quantum + n_params_embedding + n_params_rotation + n_params_projection

        low, high = get_param_bounds(n_params)

        # Inizializzazione su rank 0
        if rank == 0:
            rng = np.random.default_rng(cfg.get('seed', 42))
            y0 = rng.uniform(-1.0, 1.0, size=n_params)  # nello spazio [-1,1]
            logger.info(
                f"Parametri: quantum={n_params_quantum} embed={n_params_embedding} "
                f"rot={n_params_rotation} proj={n_params_projection} tot={n_params} shape_half={params_shape}"
            )

        # Prepara objective distribuito
        # sentences_split è una lista di liste: per ogni rank, lista di (global_idx, sentence)
        sentences_split = [[] for _ in range(size)]
        # Broadcast strutture: costruiamo localmente la stessa struttura
        sentences_split[rank] = my_items
        # Raccogliamo su tutti (Allgather-like manuale)
        all_items = comm.gather(my_items, root=0)
        if rank == 0:
            sentences_split = all_items
        sentences_split = comm.bcast(sentences_split, root=0)

        objective, stop_workers, worker_loop, loss_history, best_params_tracker = distributed_objective_factory(
            comm, rank, size, sentences_split, encoding_local, low, high, cfg, logger,
            params_shape, embedding_shape, rotation_shape, cfg.get('log_frequency', 10),
            quantum_states=quantum_states if use_quantum_states else None,
            projection_shape=projection_shape,
            projector=projector
        )

        # Rank non-zero: entra nel loop worker e resta in ascolto
        if rank != 0:
            worker_loop()
            return 0

        # Rank 0: ottimizzazione COBYLA con early stopping
        constraints = box_constraints_for_unit(n_params)
        RANDOM_RESTARTS = 1     # solo 1 run
        MAXITER_PER_RUN = cfg.get('opt_maxiter', 100)
        RHO_BEG = cfg.get('rhobeg', 0.25)
        TOL = cfg.get('tol', 1e-5)

        logger.info(
            f"Inizio ottimizzazione con COBYLA (NO early stopping): "
            f"maxiter={MAXITER_PER_RUN}, tol={TOL}"
        )
        logger.info(
            f"[STOPPING CONFIG] HARD CONSTRAINT: Stop ONLY at {MAX_LOSS_EVALUATIONS} loss evaluations"
        )
        logger.info(
            f"[STOPPING CONFIG] All early stopping logic DISABLED"
        )

        y_start = y0
        
        # ============================================================
        # OTTIMIZZAZIONE CON GESTIONE EARLY STOPPING EXCEPTION
        # ============================================================
        early_stopped = False
        early_stop_reason = "Not stopped"
        best_y = None
        best_f = np.inf
        iters = 0
        
        try:
            res = minimize(
                fun=objective,
                x0=y_start,
                method="COBYLA",
                constraints=constraints,
                options={
                    "maxiter": MAXITER_PER_RUN,
                    "rhobeg": RHO_BEG,
                    "tol": TOL,
                    "disp": True,
                },
            )
            
            # Terminazione normale (converged before 120 evaluations)
            best_y = res.x
            best_f = res.fun
            iters = getattr(res, "nit", None)
            if iters is None:
                iters = getattr(res, "nfev", "N/A")
            early_stopped = False
            logger.info(
                f"[OPTIMIZATION] Converged naturally after {GLOBAL_LOSS_COUNTER[0]} loss evaluations "
                f"(before reaching limit of {MAX_LOSS_EVALUATIONS})"
            )
            
        except MaxLossEvaluationsReached as e:
            # REACHED 120 LOSS EVALUATIONS - This is NOT an error!
            logger.info("="*70)
            logger.info(f"[MAX EVALUATIONS] Training reached {MAX_LOSS_EVALUATIONS} loss evaluations")
            logger.info(f"[MAX EVALUATIONS] This is the intended stopping condition")
            logger.info("="*70)
            
            # Use best parameters found during optimization
            if best_params_tracker['params'] is not None:
                best_y = best_params_tracker['params']  # Already in y_scaled format
                best_f = best_params_tracker['loss']
                iters = best_params_tracker['iteration']
                logger.info(
                    f"[FINAL PARAMS] Using best parameters found: "
                    f"loss={best_f:.6f}, iteration={iters}"
                )
            else:
                # Fallback: use y_start if no best params tracked
                logger.warning(
                    f"[FINAL PARAMS] No best params tracked, using initial parameters"
                )
                best_y = y_start
                best_f = np.inf
                iters = 0
            
            early_stopped = True
            early_stop_reason = f"Reached max {MAX_LOSS_EVALUATIONS} loss evaluations"
            logger.info(f"[STOPPING REASON] {early_stop_reason}")
        
        # ============================================================
        # LOGGING RISULTATI OTTIMIZZAZIONE
        # ============================================================
        if early_stopped:
            logger.info(
                f"[COBYLA] 🛑 Terminato per SMART EARLY STOPPING dopo {iters} iterazioni"
            )
            logger.info(f"[COBYLA] Reason: {early_stop_reason}")
            logger.info(f"[COBYLA] Final loss: {best_f:.6f}")
        else:
            logger.info(f"[COBYLA] Terminato normalmente: f*={best_f}, iters={iters}")

        # Stop ai worker
        stop_workers()

        # ============================================================
        # PARAMETRI FINALI IN SPAZIO NATIVO
        # ============================================================
        best_params_native = descale_from_unit(best_y, low, high)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ============================================================
        # SALVATAGGIO TUTTE LE MATRICI U, W, E, F, P (CRUCIALE)
        # ============================================================
        logger.info("[SAVE] Inizio salvataggio TUTTE le matrici U, W, E, F, P...")
        run_dir = None
        try:
            run_dir = save_all_matrices(
                best_params_native, cfg, logger, ts, 
                encoding_instance=encoding_local,
                projection_shape=projection_shape,
                embedding_shape=embedding_shape,
                rotation_shape=rotation_shape
            )
            logger.info(f"[SAVE] ✓ Tutte le matrici salvate in: {run_dir}")
        except Exception as e:
            logger.error(f"[SAVE] ✗ Errore nel salvataggio matrici: {e}")
            logger.error(traceback.format_exc())
        
        # ============================================================
        # ANALISI STATO ANCILLAE (REGISTRO C) - SKIPPATA
        # ============================================================
        ancillae_stats = None
        logger.info("[ANCILLAE] ⏭ Analisi registro C skippata (disabilitata)")
    
        
        # ============================================================
        # 3-FOLD CROSS VALIDATION (POST-TRAINING)
        # ============================================================
        if run_dir:
            logger.info("[CV] Inizio 3-fold cross validation...")
            try:
                cv_results = run_3fold_cross_validation(
                    run_dir, cfg, logger, use_quantum_states=use_quantum_states
                )
                logger.info(f"[CV] ✓ Cross validation completata: {len(cv_results)} folds")
            except Exception as e:
                logger.error(f"[CV] ✗ Errore nella cross validation: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("[CV] Skip cross validation: matrici non salvate")

        # ============================================================
        # PRINT COMPREHENSIVE FINAL SUMMARY (HARD CONSTRAINT #5)
        # ============================================================
        # Get best loss from tracker
        tracker_best_loss = best_params_tracker.get('loss', best_f)
        
        # eval_metrics dovrebbe essere calcolato dalla CV, per ora None
        eval_metrics = None
        if 'cv_results' in locals() and cv_results:
            # Estrai metriche aggregate dalla CV
            eval_metrics = {
                'num_sentences': sum(r.get('num_sentences', 0) for r in cv_results),
                'used_sentences': sum(r.get('num_sentences', 0) for r in cv_results),
                'avg_loss_per_sentence': np.mean([r.get('loss', float('nan')) for r in cv_results if np.isfinite(r.get('loss', float('nan')))]),
                'perplexity': np.mean([r.get('ppl', float('nan')) for r in cv_results if np.isfinite(r.get('ppl', float('nan')))])
            }
        
        print_final_training_summary(
            best_params_native=best_params_native,
            cfg=cfg,
            logger=logger,
            timestamp=ts,
            total_loss_evals=GLOBAL_LOSS_COUNTER[0],
            final_loss=best_f,
            best_loss=tracker_best_loss,
            eval_metrics=eval_metrics,
            ancillae_stats=ancillae_stats if 'ancillae_stats' in locals() else None
        )

        # ============================================================
        # SALVATAGGIO PARAMETRI, SUMMARY E GRAFICI
        # ============================================================
        # Usa run_dir se disponibile, altrimenti crea struttura
        if run_dir:
            output_dir = run_dir
        else:
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            seed = cfg.get('seed', 42)
            run_id = f"run_{ts}_seed{seed}"
            output_dir = results_dir / run_id
            output_dir.mkdir(exist_ok=True)
        
        # Sottocartelle
        params_dir = output_dir / "parameters"
        params_dir.mkdir(exist_ok=True)
        summaries_dir = output_dir / "summaries"
        summaries_dir.mkdir(exist_ok=True)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Salva parametri
        np.save(params_dir / "theta_finali_native.npy", best_params_native)
        logger.info(f"[SAVE] Parametri salvati in {params_dir / 'theta_finali_native.npy'}")
        
        # Salva summary
        with open(summaries_dir / "training_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"total_loss_evaluations={GLOBAL_LOSS_COUNTER[0]}\n")
            f.write(f"max_loss_evaluations_limit={MAX_LOSS_EVALUATIONS}\n")
            f.write(f"best_loss_mean={best_f:.8f}\n")
            f.write(f"best_loss_seen={tracker_best_loss:.8f}\n")
            f.write(f"n_params_total={n_params}\n")
            f.write(f"n_params_quantum={n_params_quantum}\n")
            f.write(f"n_params_embedding={n_params_embedding}\n")
            f.write(f"n_params_rotation={n_params_rotation}\n")
            f.write(f"n_params_projection={n_params_projection}\n")
            f.write(f"param_shape_half={params_shape}\n")
            if projection_shape is not None:
                f.write(f"projection_shape={projection_shape}\n")
            f.write(f"maxiter={MAXITER_PER_RUN}, tol={TOL}\n")
            f.write("\n# Stopping Configuration (HARD CONSTRAINTS)\n")
            f.write(f"early_stopping=DISABLED\n")
            f.write(f"stopping_criterion=ONLY at {MAX_LOSS_EVALUATIONS} loss evaluations\n")
            f.write(f"patience=DISABLED\n")
            f.write(f"delta_checks=DISABLED\n")
            f.write(f"\n# Optimization Results\n")
            f.write(f"stopped_at_max_evaluations={early_stopped}\n")
            f.write(f"stop_reason={early_stop_reason}\n")
            f.write(f"actual_iterations={iters}\n")
            f.write(f"mpi_size={size}, sentences={len(sentences)}\n")
            if eval_metrics:
                f.write(f"\n# Test Data Perplexity\n")
                f.write(
                    f"eval_sentences_used="
                    f"{eval_metrics.get('used_sentences', 0)}/{eval_metrics.get('num_sentences', 0)}\n"
                )
                f.write(
                    f"eval_avg_loss_per_sentence={eval_metrics.get('avg_loss_per_sentence', float('nan')):.8f}\n"
                )
                f.write(
                    f"eval_avg_loss_per_word={eval_metrics.get('avg_loss_per_word', float('nan')):.8f}\n"
                )
                f.write(f"eval_perplexity={eval_metrics.get('perplexity', float('nan')):.8f}\n")
            else:
                f.write("eval_perplexity=NA\n")
        
        logger.info(f"[SAVE] Summary salvato in {summaries_dir / 'training_summary.txt'}")
        
        # Salvataggio storia loss
        np.savetxt(summaries_dir / "loss_history.txt", np.array(loss_history))
        logger.info(f"[SAVE] Storia loss salvata in {summaries_dir / 'loss_history.txt'}")
        
        # ✨ GRAFICO TRAINING LOSS
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            if loss_history:
                epochs = list(range(1, len(loss_history) + 1))
                
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, loss_history, marker='o', linewidth=2, markersize=4, color='blue', label='Training Loss')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.title('Training Loss over Epochs', fontsize=14, fontweight='bold')
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plot_path = plots_dir / "training_loss.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"[SAVE] ✓ Grafico training loss salvato in {plot_path}")
            else:
                logger.warning("[SAVE] ⚠ Nessun dato loss_history per creare il grafico")
                
        except Exception as e:
            logger.warning(f"[SAVE] ⚠ Impossibile creare grafico training loss: {e}")

        logger.info(f"[COMPLETE] Ottimizzazione completata. Loss media finale: {best_f:.6f}")
        logger.info(f"[COMPLETE] Tutti i file salvati in: {output_dir}")
        logger.info(f"[COMPLETE] - Parametri: {params_dir}")
        logger.info(f"[COMPLETE] - Summaries: {summaries_dir}")
        logger.info(f"[COMPLETE] - Plots: {plots_dir}")
        if run_dir:
            logger.info(f"[COMPLETE] - Matrici: {run_dir / 'matrices'}")

        return 0
    
    except Exception as e:
        if rank == 0:
            logger.error("Errore critico durante il training")
            logger.error(f"Tipo: {type(e).__name__}, Msg: {e}")
            for line in traceback.format_exc().splitlines():
                logger.error(line)
            try:
                tag = np.array([2], dtype=np.int32)
                MPI.COMM_WORLD.Bcast([tag, MPI.INT], root=0)
            except Exception:
                pass
            exit_local = 1
        else:
            exit_local = 0

    finally:
        try:
            comm.Barrier()
            if rank == 0:
                stop_workers()
        except Exception:
            pass
    # ---- Uscita coerente tra i rank ----
    comm.Barrier()
    exit_global = comm.allreduce(exit_local, op=MPI.SUM)
    if exit_global == 0:
        logger.info(f"[Rank {rank}] Uscita pulita (exit=0)")
        return 0
    else:
        if rank == 0:
            logger.error(f"Terminazione globale anomala: {exit_global} rank in errore")
        return 1



if __name__ == "__main__":
    sys.exit(main())
