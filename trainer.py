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
            if self.verbose and self.counter > 0:
                print(
                    f"    [!] EarlyStopping: nessun miglioramento da {self.counter}/{self.patience} epoche "
                    f"(best {self.metric}={self.best_score:.6f} @ epoca {self.best_epoch})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(
                        f"    [X] EarlyStopping ATTIVATO: training fermato dopo {epoch} epoche. "
                        f"Miglior {self.metric}={self.best_score:.6f} @ epoca {self.best_epoch}"
                    )

        return self.early_stop

    def restore_best_model(self, model: nn.Module):
        """Ripristina i pesi del miglior modello trovato."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print(f"    [OK] Modello ripristinato ai pesi dell'epoca {self.best_epoch}")


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
        if config.TORCH_COMPILE and hasattr(torch, 'compile'):
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
        self.optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

        # --- LR Scheduler ---
        self.scheduler = WarmupCosineScheduler(self.optimizer)

        # --- Early Stopping ---
        self.early_stopper = None
        if config.EARLY_STOPPING_ENABLED:
            self.early_stopper = EarlyStopping()

        # --- EMA ---
        self.ema = None
        if config.EMA_ENABLED:
            self.ema = ModelEMA(model)

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

        if self.use_amp and self.scaler is not None:
            # === Path CUDA con AMP ===
            for step, (x_batch, y_batch) in enumerate(self.train_loader):
                with torch.amp.autocast(self.device):
                    pred = self.model(x_batch)
                    loss, fid = self.criterion(pred, y_batch)
                    loss_scaled = loss / self.grad_accum_steps

                self.scaler.scale(loss_scaled).backward()

                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == n_batches:
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.ema is not None:
                        self.ema.update(self.model)

                loss_acc += loss.item()
                fid_acc += fid.item()
        else:
            # === Path CPU: niente AMP/Scaler, minimo overhead ===
            for step, (x_batch, y_batch) in enumerate(self.train_loader):
                pred = self.model(x_batch)
                loss, fid = self.criterion(pred, y_batch)

                if self.grad_accum_steps > 1:
                    (loss / self.grad_accum_steps).backward()
                else:
                    loss.backward()

                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == n_batches:
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.ema is not None:
                        self.ema.update(self.model)

                loss_acc += loss.item()
                fid_acc += fid.item()

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

        for x_batch, y_batch in self.test_loader:
            pred = self.model(x_batch)
            loss, fid = self.criterion(pred, y_batch)
            loss_acc += loss.item()
            fid_acc += fid.item()

        # Ripristina pesi originali
        if use_ema and self.ema is not None:
            self.ema.restore(self.model)

        avg_loss = loss_acc / len(self.test_loader)
        avg_fid = fid_acc / len(self.test_loader)
        return avg_loss, avg_fid

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Salva un checkpoint del modello."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": dict(self.history),
            "best_test_fid": self.best_test_fid,
        }
        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow

        if is_best and config.SAVE_BEST_MODEL:
            os.makedirs(os.path.dirname(config.BEST_MODEL_PATH), exist_ok=True)
            torch.save(checkpoint, config.BEST_MODEL_PATH)

        if config.CHECKPOINT_EVERY_N_EPOCHS > 0 and epoch % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
            path = f"results/checkpoint_epoch_{epoch}.pt"
            os.makedirs("results", exist_ok=True)
            torch.save(checkpoint, path)

    def train(self, epochs: int = config.EPOCHS, verbose: bool = True) -> dict:
        """
        Esegue il training loop completo.

        Args:
            epochs: Numero massimo di epoche
            verbose: Se stampare il progresso

        Ritorna: Dizionario con la history completa del training
        """
        if verbose:
            features = []
            if config.EARLY_STOPPING_ENABLED:
                features.append(f"EarlyStop(patience={config.EARLY_STOPPING_PATIENCE})")
            features.append(f"LR={config.LR_SCHEDULER_TYPE}(warmup={config.LR_WARMUP_EPOCHS})")
            if config.GRAD_CLIP_MAX_NORM > 0:
                features.append(f"GradClip({config.GRAD_CLIP_MAX_NORM})")
            if config.EMA_ENABLED:
                features.append(f"EMA({config.EMA_DECAY})")
            if self.use_amp:
                features.append("AMP")
            if self.grad_accum_steps > 1:
                features.append(f"GradAccum(x{self.grad_accum_steps})")
            features.append(f"WeightDecay={config.WEIGHT_DECAY}")
            if self._compiled:
                features.append("torch.compile")
            print(f"  Features: {', '.join(features)}\n")

        for epoch in range(1, epochs + 1):
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
                lr_str = f"{current_lr:.2e}"
                marker = " *" if is_best else ""
                print(
                    f"  Epoca {epoch:3d}/{epochs}  "
                    f"| Train L:{train_loss:.4f} F:{train_fid:.4f}  "
                    f"| Test L:{test_loss:.4f} F:{test_fid:.4f}  "
                    f"| LR:{lr_str}  "
                    f"| {epoch_time:.1f}s{marker}"
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

        # --- Post-training ---
        # Ripristina il miglior modello
        if self.early_stopper is not None and self.early_stopper.best_model_state is not None:
            self.early_stopper.restore_best_model(self.model)
        elif config.SAVE_BEST_MODEL and os.path.exists(config.BEST_MODEL_PATH):
            checkpoint = torch.load(config.BEST_MODEL_PATH, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if verbose:
                print(f"    [OK] Modello ripristinato dal checkpoint (epoca {checkpoint['epoch']})")

        return dict(self.history)

    def print_training_summary(self):
        """Stampa un riepilogo dettagliato del training."""
        h = self.history
        n = len(h["train_loss"])

        print(f"\n  {'-' * 56}")
        print(f"  RIEPILOGO TRAINING AVANZATO")
        print(f"  {'-' * 56}")
        print(f"    Epoche completate:     {n}/{config.EPOCHS}")
        if n < config.EPOCHS and config.EARLY_STOPPING_ENABLED:
            print(f"    Early stopping:        SÌ (patience={config.EARLY_STOPPING_PATIENCE})")
        print(f"    Tempo totale:          {self.total_train_time:.1f}s")
        print(f"    Tempo medio/epoca:     {self.total_train_time / max(1, n):.2f}s")
        print()

        # Fidelity
        print(f"    Train Fidelity finale: {h['train_fidelity'][-1]:.6f}")
        print(f"    Test  Fidelity finale: {h['test_fidelity'][-1]:.6f}")
        print(f"    Miglior Test Fidelity: {self.best_test_fid:.6f} (epoca {self.best_test_epoch})")
        miglioramento = h["train_fidelity"][-1] - h["train_fidelity"][0]
        print(f"    Miglioramento totale:  {miglioramento:+.6f}")
        print()

        # Learning Rate
        print(f"    LR iniziale:           {h['lr'][0]:.2e}")
        print(f"    LR finale:             {h['lr'][-1]:.2e}")
        print(f"    LR minimo raggiunto:   {min(h['lr']):.2e}")
        print(f"    LR massimo raggiunto:  {max(h['lr']):.2e}")

        # Numero di riduzioni LR (cambi significativi)
        lr_changes = 0
        for i in range(1, len(h["lr"])):
            if abs(h["lr"][i] - h["lr"][i - 1]) / max(h["lr"][i - 1], 1e-10) > 0.01:
                lr_changes += 1
        print(f"    Cambi LR significativi: {lr_changes}")
        print()

        # Generalization gap
        gap = h["train_fidelity"][-1] - h["test_fidelity"][-1]
        print(f"    Generalization gap:    {gap:+.6f}")
        if gap < 0.02:
            print(f"    Verdetto:              Eccellente generalizzazione")
        elif gap < 0.05:
            print(f"    Verdetto:              Buona generalizzazione")
        elif gap < 0.15:
            print(f"    Verdetto:              Gap moderato — valutare regolarizzazione")
        else:
            print(f"    Verdetto:              Overfitting significativo")

        # Diagnostica LR
        print()
        self.lr_diagnostics.print_report()
