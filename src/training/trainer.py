"""Base trainer with W&B logging, mixed precision, and checkpoint resume.

Provides the training loop infrastructure shared across all three
training phases (segmentation, classification, multi-task).

Features:
- Mixed precision training via torch.cuda.amp
- Gradient checkpointing support
- Checkpoint save/resume every N epochs
- W&B logging with per-task loss tracking
- Early stopping with configurable patience
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.training.scheduler import CosineAnnealingWarmup, build_scheduler
from monai.inferers import sliding_window_inference
from src.utils.checkpoint import save_checkpoint, load_checkpoint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT_DIR = Path("checkpoints")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Trainer
# ---------------------------------------------------------------------------


class BaseTrainer:
    """Base training loop with AMP, checkpointing, and W&B logging.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    criterion : nn.Module
        Loss function.
    config : dict
        Full configuration dictionary.
    device : torch.device
        Training device.
    wandb_run : Optional[Any]
        W&B run object for logging. None for offline mode.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: dict,
        device: torch.device | None = None,
        wandb_run: Optional[Any] = None,
    ) -> None:
        self.config = config
        training_cfg = config.get("training", {})

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.wandb_run = wandb_run

        # Training settings
        self.max_epochs = training_cfg.get("max_epochs", 300)
        self.save_every = training_cfg.get("save_every_n_epochs", 5)
        self.patience = training_cfg.get("early_stopping_patience", 30)
        self.gradient_clip = training_cfg.get("gradient_clip", 1.0)
        self.use_amp = training_cfg.get("mixed_precision", True)

        # Build scheduler
        self.scheduler = build_scheduler(optimizer, config)

        # Mixed precision
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Checkpoint directory
        self.checkpoint_dir = DEFAULT_CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping state
        self.best_metric = -float("inf")
        self.epochs_without_improvement = 0
        self.start_epoch = 0
        self.global_step = 0

        logger.info(
            "Trainer initialized: device=%s, max_epochs=%d, AMP=%s, patience=%d",
            self.device, self.max_epochs, self.use_amp, self.patience,
        )

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Resume training from a saved checkpoint.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to the checkpoint file.
        """
        checkpoint = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler
        )
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_metric = checkpoint.get("best_metric", -float("inf"))
        self.global_step = checkpoint.get("global_step", 0)

        logger.info(
            "Resumed from checkpoint: epoch=%d, best_metric=%.4f",
            self.start_epoch, self.best_metric,
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Run a single training epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        epoch : int
            Current epoch index.

        Returns
        -------
        dict[str, float]
            Dictionary of training metrics for this epoch.
        """
        self.model.train()
        epoch_losses = []
        epoch_metrics = {}

        for batch_idx, batch_data in enumerate(train_loader):
            self.global_step += 1

            loss_dict = self._train_step(batch_data)
            epoch_losses.append(loss_dict.get("total_loss", 0.0))

            # Accumulate per-key losses
            for key, value in loss_dict.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)

            # Log to W&B
            if self.wandb_run and self.global_step % self.config.get("wandb", {}).get("log_every_n_steps", 10) == 0:
                log_dict = {f"train/{k}": v for k, v in loss_dict.items()}
                log_dict["train/lr"] = self.optimizer.param_groups[0]["lr"]
                self.wandb_run.log(log_dict, step=self.global_step)

        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        avg_metrics["epoch"] = epoch

        return avg_metrics

    def _train_step(self, batch_data: dict) -> dict[str, float]:
        """Execute a single training step.

        Override in subclasses for task-specific training logic.

        Parameters
        ----------
        batch_data : dict
            Batch dictionary from the data loader.

        Returns
        -------
        dict[str, float]
            Loss values for this step.
        """
        images = batch_data["image"].to(self.device)
        labels = batch_data["label"].to(self.device)

        self.optimizer.zero_grad()

        with autocast("cuda", enabled=self.use_amp):
            outputs = self.model(images)
            if isinstance(outputs, dict):
                seg_logits = outputs.get("seg_logits", outputs)
            else:
                seg_logits = outputs

            if isinstance(self.criterion, nn.Module) and hasattr(self.criterion, "forward"):
                # For UncertaintyWeightedLoss
                if hasattr(self.criterion, "log_var_seg"):
                    loss_dict = self.criterion(
                        seg_logits=seg_logits,
                        seg_targets=labels,
                        cls_logits=outputs.get("binary_logits") if isinstance(outputs, dict) else None,
                        cls_targets=batch_data.get("malignancy", torch.tensor([-1])).to(self.device) if "malignancy" in batch_data else None,
                        staging_logits=outputs.get("stage_logits") if isinstance(outputs, dict) else None,
                        staging_targets=batch_data.get("bclc_stage", torch.tensor([-1])).to(self.device) if "bclc_stage" in batch_data else None,
                    )
                    loss = loss_dict["total_loss"]
                else:
                    loss = self.criterion(seg_logits, labels)
                    loss_dict = {"total_loss": loss.item()}
            else:
                loss = self.criterion(seg_logits, labels)
                loss_dict = {"total_loss": loss.item()}

        self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.gradient_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Convert tensor values to float
        result = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.item()
            else:
                result[k] = float(v)

        return result

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Run validation using sliding window inference."""
        self.model.eval()
        val_losses = []
        patch_size = self.config.get("data", {}).get("patch_size", [96, 96, 96])

        for batch_data in val_loader:
            images = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)

            with autocast("cuda", enabled=self.use_amp):
                # Use sliding window inference — handles any volume size
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=patch_size,
                    sw_batch_size=2,
                    predictor=self.model,
                    overlap=0.5,
                )

                if isinstance(outputs, dict):
                    seg_logits = outputs.get("seg_logits", outputs)
                else:
                    seg_logits = outputs

                if hasattr(self.criterion, "log_var_seg"):
                    loss_dict = self.criterion(seg_logits=seg_logits, seg_targets=labels)
                    val_losses.append(loss_dict["total_loss"].item())
                elif isinstance(self.criterion, nn.Module):
                    loss = self.criterion(seg_logits, labels)
                    val_losses.append(loss.item())

        avg_loss = np.mean(val_losses) if val_losses else 0.0
        metrics = {"val_loss": avg_loss, "epoch": epoch}

        if self.wandb_run:
            self.wandb_run.log({"val/loss": avg_loss}, step=self.global_step)

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_prefix: str = "model",
    ) -> dict[str, Any]:
        """Full training loop with validation, checkpointing, and early stopping.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader
            Validation data loader.
        checkpoint_prefix : str
            Prefix for checkpoint filenames.

        Returns
        -------
        dict[str, Any]
            Final training summary.
        """
        logger.info("Starting training: epochs %d → %d", self.start_epoch, self.max_epochs)
        training_start = time.time()

        for epoch in range(self.start_epoch, self.max_epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader, epoch)

            # Step scheduler
            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, lr=%.2e, time=%.1fs",
                epoch + 1, self.max_epochs,
                train_metrics.get("total_loss", 0.0),
                val_metrics.get("val_loss", 0.0),
                current_lr, epoch_time,
            )

            # Check for improvement (using negative val_loss as metric)
            current_metric = -val_metrics.get("val_loss", float("inf"))

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.epochs_without_improvement = 0

                # Save best checkpoint
                save_checkpoint(
                    path=self.checkpoint_dir / f"{checkpoint_prefix}_best.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_metric=self.best_metric,
                    global_step=self.global_step,
                    config=self.config,
                )
                logger.info("New best model saved (metric=%.4f)", self.best_metric)
            else:
                self.epochs_without_improvement += 1

            # Periodic checkpoint
            if (epoch + 1) % self.save_every == 0:
                save_checkpoint(
                    path=self.checkpoint_dir / f"{checkpoint_prefix}_epoch{epoch + 1}.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_metric=self.best_metric,
                    global_step=self.global_step,
                    config=self.config,
                )

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch + 1, self.patience,
                )
                break

        total_time = time.time() - training_start
        logger.info("Training completed in %.1f minutes", total_time / 60)

        return {
            "best_metric": self.best_metric,
            "final_epoch": epoch + 1,
            "total_time_minutes": total_time / 60,
        }
