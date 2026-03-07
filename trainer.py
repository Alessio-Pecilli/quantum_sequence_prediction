"""
Advanced Training System per Quantum Sequence Prediction.

Include:
  - Early Stopping configurabile (su loss o fidelity)
  - LR Scheduling: Warmup lineare + Cosine Annealing + ReduceLROnPlateau
  - Gradient Clipping (max norm)
  - Gradient Accumulation (simulazione batch più grandi)
  - Mixed Precision Training (AMP) su GPU
  - EMA (Exponential Moving Average) dei pesi
  - Checkpointing automatico del miglior modello
  - Diagnostica LR: analisi e suggerimenti su quando modificare il learning rate
"""

import math
import time
import copy
import os
import gc
import random
from collections import defaultdict

import torch
import torch.nn as nn

import config


# ============================================================
#  EARLY STOPPING
# ============================================================
class EarlyStopping:
    """
    Ferma il training quando la metrica monitorata smette di migliorare.

    Supporta sia metriche da minimizzare (loss) sia da massimizzare (fidelity).
    Salva automaticamente il miglior stato del modello.
    """

    def __init__(
        self,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        min_delta: float = config.EARLY_STOPPING_MIN_DELTA,
        metric: str = config.EARLY_STOPPING_METRIC,
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.verbose = verbose

        # Determina se la metrica va massimizzata o minimizzata
        self.mode = "max" if "fidelity" in metric else "min"
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score: float, epoch: int, model: nn.Module) -> bool:
        """
        Aggiorna lo stato. Ritorna True se il training deve fermarsi.
        """
        if self.mode == "min":
            improved = (
                self.best_score is None
                or score < self.best_score - self.min_delta
            )
        else:
            improved = (
                self.best_score is None
                or score > self.best_score + self.min_delta
            )

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            # Deep copy dello stato del modello (solo pesi, non grafi computazionali)
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(
                        f"  >> EarlyStopping dopo {epoch} ep. "
                        f"(best {self.metric}={self.best_score:.6f} @ ep.{self.best_epoch})"
                    )

        return self.early_stop

    def restore_best_model(self, model: nn.Module):
        """Ripristina i pesi del miglior modello trovato."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# ============================================================
#  EMA (Exponential Moving Average)
# ============================================================
class ModelEMA:
    """
    Mantiene una copia EMA (media mobile esponenziale) dei pesi del modello.
    Durante la valutazione, i pesi EMA tendono a produrre risultati più stabili
    e performanti rispetto ai pesi istantanei.

    ema_weights = decay * ema_weights + (1 - decay) * model_weights
    """

    def __init__(self, model: nn.Module, decay: float = config.EMA_DECAY):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Aggiorna i pesi EMA dopo ogni step di ottimizzazione."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module):
        """Sostituisce i pesi del modello con quelli EMA (per valutazione)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Ripristina i pesi originali dopo la valutazione."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ============================================================
#  LR SCHEDULER COMPOSITO (Warmup + Cosine/Plateau)
# ============================================================
class WarmupCosineScheduler:
    """
    Learning Rate Scheduler composito:
      1. Fase di Warmup: rampa lineare da ~0 a lr_max in warmup_epochs
      2. Fase di Cosine Annealing: decadimento cosinusoidale fino a lr_min
      3. (Opzionale) ReduceLROnPlateau come override reattivo

    Questo pattern è lo standard per i Transformer (Vaswani et al., Loshchilov & Hutter).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = config.LR_WARMUP_EPOCHS,
        total_epochs: int = config.EPOCHS,
        lr_max: float = config.LEARNING_RATE,
        lr_min: float = config.LR_MIN,
        scheduler_type: str = config.LR_SCHEDULER_TYPE,
        plateau_patience: int = config.LR_PLATEAU_PATIENCE,
        plateau_factor: float = config.LR_PLATEAU_FACTOR,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.scheduler_type = scheduler_type
        self.current_epoch = 0

        # Storia del LR per analisi
        self.lr_history = []

        # ReduceLROnPlateau (usato come fallback reattivo)
        self._plateau_scheduler = None
        if "plateau" in scheduler_type:
            self._plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=plateau_factor,
                patience=plateau_patience,
                min_lr=lr_min,
            )

    def _cosine_lr(self, epoch: int) -> float:
        """Calcola il LR con cosine annealing dopo il warmup."""
        if epoch < self.warmup_epochs:
            # Warmup lineare
            return self.lr_min + (self.lr_max - self.lr_min) * (epoch / max(1, self.warmup_epochs))
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            progress = min(progress, 1.0)
            return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1.0 + math.cos(math.pi * progress)
            )

    def step(self, epoch: int, val_loss: float = None):
        """
        Aggiorna il LR. Chiamare alla fine di ogni epoca.
        val_loss è necessario solo se si usa ReduceLROnPlateau.
        """
        self.current_epoch = epoch

        if self.scheduler_type == "cosine":
            new_lr = self._cosine_lr(epoch)
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr

        elif self.scheduler_type == "plateau":
            # Solo warmup + plateau
            if epoch < self.warmup_epochs:
                new_lr = self.lr_min + (self.lr_max - self.lr_min) * (
                    epoch / max(1, self.warmup_epochs)
                )
                for pg in self.optimizer.param_groups:
                    pg["lr"] = new_lr
            elif val_loss is not None and self._plateau_scheduler is not None:
                self._plateau_scheduler.step(val_loss)

        elif self.scheduler_type == "cosine+plateau":
            # Cosine come base, poi plateau può ridurre ulteriormente
            cosine_lr = self._cosine_lr(epoch)
            for pg in self.optimizer.param_groups:
                pg["lr"] = cosine_lr
            # Il plateau può solo ridurre sotto il livello del cosine
            if (
                val_loss is not None
                and self._plateau_scheduler is not None
                and epoch >= self.warmup_epochs
            ):
                self._plateau_scheduler.step(val_loss)
                # Prendi il minimo tra cosine e plateau
                plateau_lr = self.optimizer.param_groups[0]["lr"]
                final_lr = min(cosine_lr, plateau_lr)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = final_lr

        current_lr = self.optimizer.param_groups[0]["lr"]
        self.lr_history.append(current_lr)
        return current_lr

    def get_last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ============================================================
#  LR DIAGNOSTICA E ANALISI
# ============================================================
class LRDiagnostics:
    """
    Analizza la dinamica del Learning Rate durante il training e genera
    suggerimenti actionable su quando e come modificarlo.
    """

    def __init__(self):
        self.loss_history = []
        self.lr_history = []
        self.fid_history = []
        self.suggestions = []

    def update(self, epoch: int, lr: float, train_loss: float, val_loss: float, val_fid: float):
        self.lr_history.append(lr)
        self.loss_history.append(val_loss)
        self.fid_history.append(val_fid)

    def analyze(self) -> list[str]:
        """
        Esegue l'analisi completa e ritorna una lista di suggerimenti.
        """
        self.suggestions = []
        n = len(self.loss_history)
        if n < 10:
            return self.suggestions

        # 1. Rileva plateau nella loss (varianza molto bassa nelle ultime N epoche)
        window = min(15, n // 3)
        recent_loss = self.loss_history[-window:]
        loss_var = self._variance(recent_loss)
        loss_range = max(recent_loss) - min(recent_loss)

        if loss_range < 1e-4 and n > 20:
            self.suggestions.append(
                f"PLATEAU RILEVATO: la val_loss è stagnante nelle ultime {window} epoche "
                f"(range={loss_range:.2e}). Considerare un LR più aggressivo o early stop."
            )

        # 2. Rileva oscillazioni (loss che sale e scende frequentemente)
        if n > 10:
            sign_changes = 0
            diffs = [self.loss_history[i+1] - self.loss_history[i] for i in range(n - 1)]
            for i in range(len(diffs) - 1):
                if diffs[i] * diffs[i + 1] < 0:
                    sign_changes += 1
            oscillation_rate = sign_changes / max(1, len(diffs) - 1)
            if oscillation_rate > 0.7:
                self.suggestions.append(
                    f"OSCILLAZIONI: la val_loss oscilla frequentemente (rate={oscillation_rate:.0%}). "
                    f"Il LR potrebbe essere troppo alto. Ridurre il LR o aumentare la patience del plateau."
                )

        # 3. Rileva divergenza (loss in crescita costante)
        if n > 5:
            recent_5 = self.loss_history[-5:]
            if all(recent_5[i] > recent_5[i-1] for i in range(1, 5)):
                self.suggestions.append(
                    "DIVERGENZA: la val_loss è in aumento costante nelle ultime 5 epoche. "
                    "Il LR è probabilmente troppo alto. Ridurre immediatamente."
                )

        # 4. Analisi velocità di miglioramento
        if n > 20:
            first_half_improvement = self.fid_history[n//2] - self.fid_history[0]
            second_half_improvement = self.fid_history[-1] - self.fid_history[n//2]
            if first_half_improvement > 0 and second_half_improvement > 0:
                slowdown_ratio = second_half_improvement / max(1e-8, first_half_improvement)
                if slowdown_ratio < 0.1:
                    self.suggestions.append(
                        f"RALLENTAMENTO: il miglioramento nella 2a meta' ({second_half_improvement:.4f}) "
                        f"e' solo il {slowdown_ratio:.1%} rispetto alla 1a meta' ({first_half_improvement:.4f}). "
                        f"Il modello potrebbe aver raggiunto la sua capacità massima con questa configurazione."
                    )

        # 5. LR troppo basso fin dall'inizio
        if n > 10:
            total_improvement = self.fid_history[-1] - self.fid_history[0]
            if total_improvement < 0.01 and self.lr_history[0] < 1e-4:
                self.suggestions.append(
                    "LR TROPPO BASSO: dopo molte epoche il miglioramento è minimo. "
                    "Provare un LR iniziale più alto (es. 3e-3 o 1e-2)."
                )

        return self.suggestions

    @staticmethod
    def _variance(values):
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / (len(values) - 1)

    def print_report(self):
        """Stampa il report diagnostico del LR."""
        suggestions = self.analyze()
        if not suggestions:
            print("    [OK] Nessun problema rilevato nella dinamica del LR.")
            return

        print(f"    {'-' * 50}")
        print(f"    DIAGNOSTICA LEARNING RATE ({len(suggestions)} suggerimenti)")
        print(f"    {'-' * 50}")
        for i, s in enumerate(suggestions, 1):
            print(f"    {i}. {s}")


# ============================================================
#  TRAINER AVANZATO
# ============================================================
class AdvancedTrainer:
    """
    Training loop avanzato che integra tutte le best practice:
      - Early Stopping
      - LR Scheduling (Warmup + Cosine + Plateau)
      - Gradient Clipping
      - Gradient Accumulation
      - Mixed Precision (AMP)
      - EMA dei pesi
      - Checkpointing del miglior modello
      - Diagnostica LR
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader,
        test_loader,
        device: str = config.DEVICE,
    ):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        # --- torch.compile per velocizzare (PyTorch 2.0+) ---
        # In MEMORY_SAFE_MODE lo disattiviamo per evitare picchi memoria in compilazione.
        compile_enabled = config.TORCH_COMPILE and not config.MEMORY_SAFE_MODE and device == "cuda"
        if compile_enabled and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(
                    self.model,
                    backend=config.TORCH_COMPILE_BACKEND,
                    fullgraph=False,
                )
                self._compiled = True
            except Exception as e:
                print(f"  [WARN] torch.compile fallito ({e}), proseguo senza compilazione")
                self._compiled = False
        else:
            if config.TORCH_COMPILE and config.MEMORY_SAFE_MODE:
                print("  [MEM] MEMORY_SAFE_MODE attivo: torch.compile disabilitato.")
            if config.TORCH_COMPILE and device != "cuda":
                print("  [INFO] torch.compile abilitato in config ma device!=cuda: compilazione disabilitata.")
            self._compiled = False

        # --- Ottimizzatore: AdamW (Adam con weight decay decoupled) ---
        # fused=True e' piu' veloce ma disponibile solo su CUDA
        optimizer_kwargs = dict(
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        if device == "cuda":
            optimizer_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)

        # --- LR Scheduler ---
        self.scheduler = WarmupCosineScheduler(self.optimizer)

        # --- Early Stopping ---
        self.early_stopper = None
        if config.EARLY_STOPPING_ENABLED:
            self.early_stopper = EarlyStopping()

        # --- EMA ---
        self.ema = None
        if config.EMA_ENABLED:
            self.ema = ModelEMA(self.model)

        # --- Mixed Precision (solo su CUDA, su CPU e' overhead inutile) ---
        self.use_amp = config.AMP_ENABLED and device == "cuda"
        self.scaler = torch.amp.GradScaler(device, enabled=self.use_amp) if self.use_amp else None

        # --- Gradient Accumulation ---
        self.grad_accum_steps = config.GRAD_ACCUMULATION_STEPS

        # --- Gradient Clipping ---
        self.grad_clip = config.GRAD_CLIP_MAX_NORM

        # --- LR Diagnostica ---
        self.lr_diagnostics = LRDiagnostics()

        # --- History ---
        self.history = defaultdict(list)
        self.total_train_time = 0.0
        self.actual_epochs = 0
        self.best_test_fid = 0.0
        self.best_test_epoch = 0

        # --- Memory management ---
        if config.MICRO_BATCH_SIZE <= 0:
            self.micro_batch_size = config.BATCH_SIZE
        else:
            self.micro_batch_size = min(config.MICRO_BATCH_SIZE, config.BATCH_SIZE)

        self.pin_memory = bool(config.PIN_MEMORY and device == "cuda")
        self.non_blocking = self.pin_memory
        self.gc_every_n_steps = max(0, int(config.GC_COLLECT_EVERY_N_STEPS))
        self.cuda_cache_every_n_steps = max(0, int(config.CUDA_EMPTY_CACHE_EVERY_N_STEPS))
        self._checkpoint_warned = False
        self._current_epoch = 0

    def _iter_micro_batches(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        Suddivide un batch in micro-batch per ridurre il picco memoria.
        """
        batch_size = x_batch.shape[0]
        micro_bs = max(1, min(self.micro_batch_size, batch_size))

        for start in range(0, batch_size, micro_bs):
            end = min(start + micro_bs, batch_size)
            x_micro = x_batch[start:end]
            y_micro = y_batch[start:end]
            weight = x_micro.shape[0] / batch_size
            yield x_micro, y_micro, weight

    def _periodic_memory_cleanup(self, step: int):
        """Cleanup periodico per evitare crescita della memoria durante loop lunghi."""
        do_gc = self.gc_every_n_steps > 0 and step % self.gc_every_n_steps == 0
        do_cuda = (
            self.device == "cuda"
            and self.cuda_cache_every_n_steps > 0
            and step % self.cuda_cache_every_n_steps == 0
        )

        if do_gc:
            gc.collect()
        if do_cuda:
            torch.cuda.empty_cache()

    def _force_memory_cleanup(self):
        """Cleanup completo a fine epoca/fase."""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    @staticmethod
    def _atomic_torch_save(obj: dict, path: str):
        """
        Salvataggio atomico per evitare checkpoint corrotti.
        Scrive su file temporaneo e poi fa replace atomico.
        """
        tmp_path = f"{path}.tmp"
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)

    @staticmethod
    def _collect_rng_state() -> dict:
        """
        Snapshot degli RNG states per resume riproducibile.
        """
        state = {
            "python_random_state": random.getstate(),
            "torch_random_state": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_random_state_all"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict | None):
        """Ripristina RNG states se disponibili nel checkpoint."""
        if not state:
            return

        py_state = state.get("python_random_state")
        if py_state is not None:
            random.setstate(py_state)

        torch_state = state.get("torch_random_state")
        if torch_state is not None:
            torch.random.set_rng_state(torch_state)

        cuda_state = state.get("cuda_random_state_all")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)

    def _build_checkpoint_payload(self, epoch: int, reason: str | None = None) -> dict:
        """Costruisce payload completo del checkpoint (ultimo/best/emergency)."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state": {
                "current_epoch": self.scheduler.current_epoch,
                "lr_history": self.scheduler.lr_history,
            },
            "history": dict(self.history),
            "best_test_fid": self.best_test_fid,
            "best_test_epoch": self.best_test_epoch,
            "architecture_signature": self._get_architecture_signature(),
            "rng_state": self._collect_rng_state(),
            "saved_at_unix": time.time(),
        }
        if reason is not None:
            checkpoint["save_reason"] = reason

        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        if self.early_stopper is not None:
            checkpoint["early_stopping"] = {
                "best_score": self.early_stopper.best_score,
                "counter": self.early_stopper.counter,
                "best_epoch": self.early_stopper.best_epoch,
            }

        return checkpoint

    def _save_emergency_checkpoint(self, epoch: int, reason: str):
        """
        Salva un checkpoint di emergenza in caso di interruzione/errore runtime.
        """
        try:
            checkpoint = self._build_checkpoint_payload(epoch=epoch, reason=reason)
            os.makedirs("results", exist_ok=True)
            path = os.path.join("results", f"emergency_{reason}.pt")
            self._atomic_torch_save(checkpoint, path)
            print(f"  [SAFE] Checkpoint emergenza salvato: {path}")
        except Exception as e:
            print(f"  [WARN] Impossibile salvare checkpoint emergenza ({reason}): {e}")

    def _train_one_epoch(self) -> tuple[float, float]:
        """
        Esegue una singola epoca di training.
        Su CPU: path diretto senza AMP/GradScaler (zero overhead).
        Su CUDA: AMP + GradScaler per massima velocita'.

        Ritorna: (avg_loss, avg_fidelity)
        """
        self.model.train()
        loss_acc, fid_acc = 0.0, 0.0
        self.optimizer.zero_grad(set_to_none=True)  # set_to_none=True e' piu' veloce

        n_batches = len(self.train_loader)

        for step, (x_batch_cpu, y_batch_cpu) in enumerate(self.train_loader, start=1):
            batch_loss, batch_fid = 0.0, 0.0

            for x_micro_cpu, y_micro_cpu, weight in self._iter_micro_batches(x_batch_cpu, y_batch_cpu):
                x_micro = x_micro_cpu.to(self.device, non_blocking=self.non_blocking)
                y_micro = y_micro_cpu.to(self.device, non_blocking=self.non_blocking)

                if self.use_amp and self.scaler is not None:
                    with torch.amp.autocast(self.device):
                        pred = self.model(x_micro)
                        loss, fid = self.criterion(pred, y_micro)
                        loss_scaled = (loss * weight) / self.grad_accum_steps
                    self.scaler.scale(loss_scaled).backward()
                else:
                    pred = self.model(x_micro)
                    loss, fid = self.criterion(pred, y_micro)
                    loss_scaled = (loss * weight) / self.grad_accum_steps
                    loss_scaled.backward()

                batch_loss += loss.item() * weight
                batch_fid += fid.item() * weight

                del x_micro, y_micro, pred, loss, fid, loss_scaled

            if step % self.grad_accum_steps == 0 or step == n_batches:
                if self.grad_clip > 0:
                    if self.use_amp and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

                if self.use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)

            loss_acc += batch_loss
            fid_acc += batch_fid

            del x_batch_cpu, y_batch_cpu
            self._periodic_memory_cleanup(step)

        self._force_memory_cleanup()

        avg_loss = loss_acc / n_batches
        avg_fid = fid_acc / n_batches
        return avg_loss, avg_fid

    @torch.no_grad()
    def _evaluate(self, use_ema: bool = True) -> tuple[float, float]:
        """
        Valutazione sul test set.
        Se use_ema=True e EMA è abilitato, usa i pesi EMA per la valutazione.

        Ritorna: (avg_loss, avg_fidelity)
        """
        # Applica pesi EMA per valutazione
        if use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)

        self.model.eval()
        loss_acc, fid_acc = 0.0, 0.0

        for step, (x_batch_cpu, y_batch_cpu) in enumerate(self.test_loader, start=1):
            batch_loss, batch_fid = 0.0, 0.0

            for x_micro_cpu, y_micro_cpu, weight in self._iter_micro_batches(x_batch_cpu, y_batch_cpu):
                x_micro = x_micro_cpu.to(self.device, non_blocking=self.non_blocking)
                y_micro = y_micro_cpu.to(self.device, non_blocking=self.non_blocking)

                pred = self.model(x_micro)
                loss, fid = self.criterion(pred, y_micro)

                batch_loss += loss.item() * weight
                batch_fid += fid.item() * weight

                del x_micro, y_micro, pred, loss, fid

            loss_acc += batch_loss
            fid_acc += batch_fid

            del x_batch_cpu, y_batch_cpu
            self._periodic_memory_cleanup(step)

        # Ripristina pesi originali
        if use_ema and self.ema is not None:
            self.ema.restore(self.model)

        self._force_memory_cleanup()

        avg_loss = loss_acc / len(self.test_loader)
        avg_fid = fid_acc / len(self.test_loader)
        return avg_loss, avg_fid

    @staticmethod
    def _get_architecture_signature() -> dict:
        """
        Raccoglie tutti i parametri di configurazione che definiscono l'architettura
        del modello. Se uno qualsiasi di questi cambia, i pesi salvati NON sono
        compatibili e bisogna ripartire da zero.
        """
        return {
            "D_MODEL": config.D_MODEL,
            "NUM_HEADS": config.NUM_HEADS,
            "NUM_LAYERS": config.NUM_LAYERS,
            "DIM_FEEDFORWARD": config.DIM_FEEDFORWARD,
            "N_QUBITS": config.N_QUBITS,
            "DIM_2N": config.DIM_2N,
            "SEQ_LEN": config.SEQ_LEN,
            "DROPOUT": config.DROPOUT,
        }

    @staticmethod
    def check_checkpoint_compatibility(checkpoint_path: str) -> tuple[bool, str, dict | None]:
        """
        Verifica se un checkpoint salvato è compatibile con la configurazione corrente.

        Ritorna:
            (compatibile, messaggio, checkpoint_data)
            - compatibile: True se i pesi possono essere riutilizzati
            - messaggio: spiegazione leggibile del risultato
            - checkpoint_data: il checkpoint caricato (None se non esiste o non valido)
        """
        if not os.path.exists(checkpoint_path):
            return False, f"Nessun checkpoint trovato in '{checkpoint_path}'", None

        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        except Exception as e:
            return False, f"Checkpoint corrotto o illeggibile: {e}", None

        # Verifica che il checkpoint contenga i campi necessari
        required_keys = {"epoch", "model_state_dict", "optimizer_state_dict", "history"}
        if not required_keys.issubset(checkpoint.keys()):
            missing = required_keys - checkpoint.keys()
            return False, f"Checkpoint incompleto, campi mancanti: {missing}", None

        # Verifica compatibilità architettura
        saved_arch = checkpoint.get("architecture_signature")
        if saved_arch is None:
            return False, "Checkpoint senza firma architettura (versione precedente), incompatibile", None

        current_arch = AdvancedTrainer._get_architecture_signature()
        mismatches = []
        for key in current_arch:
            saved_val = saved_arch.get(key)
            current_val = current_arch[key]
            if saved_val != current_val:
                mismatches.append(f"  {key}: salvato={saved_val} → attuale={current_val}")

        if mismatches:
            detail = "\n".join(mismatches)
            n_saved = sum(p.numel() for p in checkpoint["model_state_dict"].values())
            msg = (
                f"Architettura INCOMPATIBILE ({len(mismatches)} differenze):\n{detail}\n"
                f"  Parametri salvati: {n_saved:,} — Si riparte da zero."
            )
            return False, msg, None

        saved_epoch = checkpoint["epoch"]
        saved_fid = checkpoint.get("best_test_fid", 0.0)
        msg = (
            f"Checkpoint COMPATIBILE trovato (epoca {saved_epoch}, "
            f"best fidelity={saved_fid:.6f}). Riprendo il training."
        )
        return True, msg, checkpoint

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Salva un checkpoint del modello."""
        checkpoint = self._build_checkpoint_payload(epoch=epoch, reason="epoch")

        # Salva SEMPRE l'ultimo checkpoint (fallback per crash/spegnimento)
        try:
            os.makedirs(os.path.dirname(config.LAST_CHECKPOINT_PATH) or ".", exist_ok=True)
            self._atomic_torch_save(checkpoint, config.LAST_CHECKPOINT_PATH)

            if is_best and config.SAVE_BEST_MODEL:
                os.makedirs(os.path.dirname(config.BEST_MODEL_PATH) or ".", exist_ok=True)
                best_checkpoint = dict(checkpoint)
                best_checkpoint["save_reason"] = "best"
                self._atomic_torch_save(best_checkpoint, config.BEST_MODEL_PATH)

            if config.CHECKPOINT_EVERY_N_EPOCHS > 0 and epoch % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
                path = f"results/checkpoint_epoch_{epoch}.pt"
                os.makedirs("results", exist_ok=True)
                periodic_checkpoint = dict(checkpoint)
                periodic_checkpoint["save_reason"] = "periodic"
                self._atomic_torch_save(periodic_checkpoint, path)
        except Exception as e:
            if not self._checkpoint_warned:
                print(f"  [WARN] Checkpoint non salvato: {e}")
                self._checkpoint_warned = True

    def resume_from_checkpoint(self, checkpoint: dict):
        """
        Ripristina lo stato completo del trainer da un checkpoint compatibile.
        Deve essere chiamato PRIMA di train().
        """
        # Modello
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Ottimizzatore
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Scheduler
        if "scheduler_state" in checkpoint:
            self.scheduler.current_epoch = checkpoint["scheduler_state"]["current_epoch"]
            self.scheduler.lr_history = checkpoint["scheduler_state"]["lr_history"]

        # History
        saved_history = checkpoint.get("history", {})
        for key, values in saved_history.items():
            self.history[key] = list(values)

        # Best tracking
        self.best_test_fid = checkpoint.get("best_test_fid", 0.0)
        self.best_test_epoch = checkpoint.get("best_test_epoch", 0)

        # EMA
        if self.ema is not None and "ema_shadow" in checkpoint:
            self.ema.shadow = checkpoint["ema_shadow"]
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Early Stopping
        if self.early_stopper is not None and "early_stopping" in checkpoint:
            es_data = checkpoint["early_stopping"]
            self.early_stopper.best_score = es_data.get("best_score")
            self.early_stopper.counter = es_data.get("counter", 0)
            self.early_stopper.best_epoch = es_data.get("best_epoch", 0)

        # RNG states (se presenti)
        self._restore_rng_state(checkpoint.get("rng_state"))

        # Epoca di partenza
        self._resume_epoch = checkpoint["epoch"]

    def train(self, epochs: int = config.EPOCHS, verbose: bool = True) -> dict:
        """
        Esegue il training loop completo.

        Args:
            epochs: Numero massimo di epoche
            verbose: Se stampare il progresso

        Ritorna: Dizionario con la history completa del training
        """
        start_epoch = getattr(self, "_resume_epoch", 0) + 1

        if start_epoch > 1 and verbose:
            print(f"  >> Ripresa dal checkpoint: epoca {start_epoch}/{epochs}")
            print(f"     Best fidelity finora: {self.best_test_fid:.6f} (ep.{self.best_test_epoch})")
            print()

        if start_epoch > epochs:
            print(f"  >> Il checkpoint (epoca {start_epoch - 1}) ha già completato {epochs} epoche. Nulla da fare.")
            return dict(self.history)

        try:
            for epoch in range(start_epoch, epochs + 1):
                self._current_epoch = epoch
                t_epoch = time.time()

                # --- Train ---
                train_loss, train_fid = self._train_one_epoch()
                train_ppl = math.exp(min(train_loss, 20))  # clamp per evitare overflow

                # --- Evaluate ---
                test_loss, test_fid = self._evaluate(use_ema=config.EMA_ENABLED)
                test_ppl = math.exp(min(test_loss, 20))

                epoch_time = time.time() - t_epoch
                self.total_train_time += epoch_time

                # --- LR Scheduling ---
                current_lr = self.scheduler.step(epoch, val_loss=test_loss)

                # --- History ---
                self.history["train_loss"].append(train_loss)
                self.history["train_fidelity"].append(train_fid)
                self.history["train_ppl"].append(train_ppl)
                self.history["test_loss"].append(test_loss)
                self.history["test_fidelity"].append(test_fid)
                self.history["test_ppl"].append(test_ppl)
                self.history["lr"].append(current_lr)
                self.history["epoch_time"].append(epoch_time)

                # --- Best model tracking ---
                is_best = test_fid > self.best_test_fid
                if is_best:
                    self.best_test_fid = test_fid
                    self.best_test_epoch = epoch

                # --- Checkpointing ---
                self._save_checkpoint(epoch, is_best=is_best)

                # --- LR Diagnostics ---
                self.lr_diagnostics.update(epoch, current_lr, train_loss, test_loss, test_fid)

                # --- Stampa progresso ---
                if verbose:
                    marker = " *" if is_best else ""
                    print(
                        f"  Ep {epoch:3d}/{epochs}  "
                        f"Train [Loss {train_loss:.4f}  PPL {train_ppl:.4f}]  "
                        f"Test [Loss {test_loss:.4f}  PPL {test_ppl:.4f}]{marker}"
                    )

                # --- Early Stopping ---
                if self.early_stopper is not None:
                    es_metric = (
                        test_fid
                        if config.EARLY_STOPPING_METRIC == "test_fidelity"
                        else test_loss
                    )
                    if self.early_stopper(es_metric, epoch, self.model):
                        break

                self.actual_epochs = epoch
        except KeyboardInterrupt:
            print("\n  [INTERRUPT] Training interrotto dall'utente.")
            self._save_emergency_checkpoint(
                epoch=max(start_epoch, self._current_epoch),
                reason="keyboard_interrupt",
            )
            raise
        except Exception as e:
            print(f"\n  [ERROR] Eccezione durante il training: {e}")
            self._save_emergency_checkpoint(
                epoch=max(start_epoch, self._current_epoch),
                reason="runtime_error",
            )
            raise

        # --- Post-training ---
        # Ripristina il miglior modello
        if self.early_stopper is not None and self.early_stopper.best_model_state is not None:
            self.early_stopper.restore_best_model(self.model)
        elif config.SAVE_BEST_MODEL and os.path.exists(config.BEST_MODEL_PATH):
            checkpoint = torch.load(config.BEST_MODEL_PATH, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        return dict(self.history)

    def print_training_summary(self):
        """Stampa un riepilogo conciso del training."""
        h = self.history
        n = len(h["train_loss"])
        if n == 0 or self.best_test_epoch <= 0:
            print(f"\n  Training interrotto senza epoche complete salvate (tempo: {self.total_train_time:.1f}s).")
            return

        print(f"\n  Training completato: {n}/{config.EPOCHS} ep. in {self.total_train_time:.1f}s")
        print(f"  Best test: Loss={h['test_loss'][self.best_test_epoch-1]:.4f}  PPL={h['test_ppl'][self.best_test_epoch-1]:.4f}  (ep.{self.best_test_epoch})")
