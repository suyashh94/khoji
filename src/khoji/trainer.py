"""Training loop for fine-tuning embedding models with LoRA."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from khoji.data import TripletDataset
from khoji.device import get_device
from khoji.lora import LoRASettings, apply_lora
from khoji.loss import triplet_margin_loss
from khoji.model import _detect_pooling, _pool, _resolve_dtype


@dataclass
class TrainingConfig:
    """Training hyperparameters.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size (actual micro-batch sent to GPU).
        grad_accum_steps: Number of steps to accumulate gradients before
            updating. Effective batch size = batch_size * grad_accum_steps.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Number of linear warmup steps.
        max_grad_norm: Maximum gradient norm for clipping. None = no clipping.
        max_length: Max token length for tokenization.
        mixed_precision: "fp16", "bf16", or None (disabled).
        loss_fn: Loss function. Takes (query_emb, pos_emb, neg_emb) -> scalar.
            Defaults to triplet_margin_loss.
        lora: LoRA configuration. None = no LoRA (train full model).
        save_dir: Directory to save the trained LoRA adapter. None = don't save.
        save_every_n_steps: Save checkpoint every N optimizer steps. None = disabled.
        keep_all_checkpoints: If True, keep all checkpoints. If False, keep only latest.
    """

    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    max_length: int = 512
    mixed_precision: str | None = None
    loss_fn: Callable[..., torch.Tensor] = triplet_margin_loss
    lora: LoRASettings | None = None
    save_dir: str | None = None
    overfit_batches: int | None = None  # Set to 1 (or N) to overfit on N batches for debugging
    sanity_check_samples: int = 10  # Number of training samples to check before/after training
    save_every_n_steps: int | None = None
    keep_all_checkpoints: bool = False
    dtype: str | None = None  # Load base model weights in this precision ("fp16", "bf16", or None)


@dataclass
class TrainHistory:
    """Training history with per-step metrics."""

    step_loss: list[float] = field(default_factory=list)
    step_lr: list[float] = field(default_factory=list)
    step_grad_norm: list[float] = field(default_factory=list)
    epoch_loss: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "step_loss": self.step_loss,
            "step_lr": self.step_lr,
            "step_grad_norm": self.step_grad_norm,
            "epoch_loss": self.epoch_loss,
        }

    def save(self, path: str) -> None:
        """Save training history to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Training history saved to {path}")


class Trainer:
    """Fine-tunes an embedding model on triplet data using LoRA.

    **HuggingFace models** — pass ``model_name``:

        Trainer("BAAI/bge-base-en-v1.5", config)

    **Custom PyTorch models** — pass ``model``, ``tokenizer``, and ``pooling``:

        Trainer(model=my_encoder, tokenizer=my_tokenizer, pooling="mean", config=config)

    When using a custom model, LoRA is still applied via ``config.lora`` if set.
    Set ``config.lora = None`` to skip LoRA and train the full model.
    """

    def __init__(
        self,
        model_name: str | None = None,
        config: TrainingConfig | None = None,
        model: torch.nn.Module | None = None,
        tokenizer: object | None = None,
        pooling: str = "cls",
        adapter_path: str | None = None,
    ):
        self.model_name = model_name or "custom"
        self.config = config or TrainingConfig()
        self.device = get_device()

        if model is not None:
            # Custom model path
            if tokenizer is None:
                raise ValueError("Must provide tokenizer when passing a custom model.")
            self.tokenizer = tokenizer
            self.pooling_mode = pooling
            base_model = model.to(self.device)
        elif model_name is not None:
            # HuggingFace model path
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            load_kwargs = {}
            torch_dtype = _resolve_dtype(self.config.dtype)
            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype
            base_model = AutoModel.from_pretrained(model_name, **load_kwargs).to(self.device)
            self.pooling_mode = _detect_pooling(model_name)
        else:
            raise ValueError("Provide either model_name or model.")

        # Apply LoRA (skip if lora config is None)
        if self.config.lora is not None:
            if adapter_path is not None:
                # Warm-start from a previously trained adapter
                self.model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                print(f"LoRA warm-start from {adapter_path} | "
                      f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
            else:
                self.model = apply_lora(base_model, self.config.lora)
        else:
            self.model = base_model

        # Mixed precision setup
        self.amp_dtype = None
        self.scaler = None
        if self.config.mixed_precision is not None:
            if self.config.mixed_precision == "fp16":
                self.amp_dtype = torch.float16
                # GradScaler only needed for fp16, not bf16
                self.scaler = torch.amp.GradScaler(device=self.device.type)
            elif self.config.mixed_precision == "bf16":
                self.amp_dtype = torch.bfloat16
            else:
                raise ValueError(
                    f"Unknown mixed_precision: {self.config.mixed_precision}. "
                    "Use 'fp16', 'bf16', or null."
                )

        effective_bs = self.config.batch_size * self.config.grad_accum_steps
        amp_str = f" | AMP: {self.config.mixed_precision}" if self.config.mixed_precision else ""
        print(f"Effective batch size: {effective_bs} "
              f"(micro={self.config.batch_size} x accum={self.config.grad_accum_steps})"
              f"{amp_str}")

    def train(self, dataset: TripletDataset) -> TrainHistory:
        """Run the training loop.

        Args:
            dataset: TripletDataset of (query, positive, negative) triplets.

        Returns:
            TrainHistory with per-step loss, lr, grad norm, and per-epoch loss.
        """
        history = TrainHistory()

        # Overfit mode: take only N batches and replay them every epoch
        overfit_samples = None
        if self.config.overfit_batches is not None:
            n = self.config.overfit_batches
            subset = [dataset[i] for i in range(min(n * self.config.batch_size, len(dataset)))]
            from khoji.data import Triplet
            dataset = TripletDataset([Triplet(*s) for s in subset])
            overfit_samples = subset
            print(f"\n[OVERFIT MODE] Training on {len(dataset)} samples for {self.config.epochs} epochs")
            self._overfit_report("BEFORE training", overfit_samples)
        else:
            print(f"\nTraining on {len(dataset)} triplets for {self.config.epochs} epochs")
            # Sanity check: sample N triplets and report before/after training
            if self.config.sanity_check_samples > 0:
                import random
                n = min(self.config.sanity_check_samples, len(dataset))
                indices = random.sample(range(len(dataset)), n)
                overfit_samples = [dataset[i] for i in indices]
                self._overfit_report("BEFORE training", overfit_samples)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.overfit_batches is None,
        )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Total optimizer steps (accounting for gradient accumulation)
        steps_per_epoch = (len(dataloader) + self.config.grad_accum_steps - 1) // self.config.grad_accum_steps
        total_opt_steps = steps_per_epoch * self.config.epochs
        total_batches = len(dataloader) * self.config.epochs
        scheduler = self._build_scheduler(optimizer, total_opt_steps)

        print(f"Total batches: {total_batches} | "
              f"Optimizer steps: {total_opt_steps} | "
              f"Grad clipping: {self.config.max_grad_norm}\n")

        self.model.train()
        global_batch = 0
        global_opt_step = 0
        accum_loss = 0.0

        pbar = tqdm(total=total_batches, desc="Training", unit="batch")

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            optimizer.zero_grad()

            for batch_idx, (queries, positives, negatives) in enumerate(dataloader):
                # Encode all three (with optional AMP)
                if self.amp_dtype is not None:
                    with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        query_emb = self._encode_batch(queries)
                        pos_emb = self._encode_batch(positives)
                        neg_emb = self._encode_batch(negatives)
                        loss = self.config.loss_fn(query_emb, pos_emb, neg_emb)
                else:
                    query_emb = self._encode_batch(queries)
                    pos_emb = self._encode_batch(positives)
                    neg_emb = self._encode_batch(negatives)
                    loss = self.config.loss_fn(query_emb, pos_emb, neg_emb)

                scaled_loss = loss / self.config.grad_accum_steps

                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                batch_loss = loss.item()
                accum_loss += batch_loss
                epoch_loss += batch_loss
                epoch_batches += 1
                global_batch += 1

                # Step optimizer every grad_accum_steps or at end of epoch
                is_accum_step = (batch_idx + 1) % self.config.grad_accum_steps == 0
                is_last_batch = (batch_idx + 1) == len(dataloader)

                if is_accum_step or is_last_batch:
                    if self.scaler is not None:
                        # fp16: unscale before clipping, then step via scaler
                        self.scaler.unscale_(optimizer)

                    # Gradient clipping
                    grad_norm = 0.0
                    if self.config.max_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        ).item()
                    else:
                        # Still compute grad norm for logging
                        grad_norm = self._compute_grad_norm()

                    if self.scaler is not None:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    global_opt_step += 1

                    # Log per optimizer step
                    n_accum = (batch_idx % self.config.grad_accum_steps) + 1
                    step_loss = accum_loss / n_accum
                    step_lr = scheduler.get_last_lr()[0]

                    history.step_loss.append(step_loss)
                    history.step_lr.append(step_lr)
                    history.step_grad_norm.append(grad_norm)

                    accum_loss = 0.0

                    # Checkpoint saving
                    if (self.config.save_every_n_steps is not None
                            and self.config.save_dir is not None
                            and global_opt_step % self.config.save_every_n_steps == 0):
                        self._save_checkpoint(global_opt_step)

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(
                    epoch=f"{epoch+1}/{self.config.epochs}",
                    loss=f"{batch_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            epoch_avg = epoch_loss / max(epoch_batches, 1)
            history.epoch_loss.append(epoch_avg)
            tqdm.write(f"  Epoch {epoch+1}/{self.config.epochs} complete | "
                       f"Avg Loss: {epoch_avg:.4f}")

        pbar.close()

        # Show after-training metrics (both overfit and sanity check modes)
        if overfit_samples is not None:
            self._overfit_report("AFTER training", overfit_samples)

        # Save if requested
        if self.config.save_dir:
            self.save(self.config.save_dir)

        return history

    def save(self, path: str) -> None:
        """Save the model weights (LoRA adapter or full model)."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        label = "LoRA adapter" if self.config.lora is not None else "Full model"
        print(f"{label} saved to {save_path}")

    def _save_checkpoint(self, step: int) -> None:
        """Save a checkpoint during training."""
        base = Path(self.config.save_dir)
        if self.config.keep_all_checkpoints:
            ckpt_dir = base / f"checkpoint-{step}"
        else:
            ckpt_dir = base / "checkpoint-latest"
            # Remove old checkpoint before saving new one
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        tqdm.write(f"  Checkpoint saved to {ckpt_dir}")

    @torch.no_grad()
    def _overfit_report(self, label: str, samples: list[tuple[str, str, str]]) -> None:
        """Print similarity metrics for overfit samples."""
        self.model.eval()

        queries = [s[0] for s in samples]
        positives = [s[1] for s in samples]
        negatives = [s[2] for s in samples]

        q_emb = self._encode_batch(queries)
        p_emb = self._encode_batch(positives)
        n_emb = self._encode_batch(negatives)

        pos_sim = torch.nn.functional.cosine_similarity(q_emb, p_emb)
        neg_sim = torch.nn.functional.cosine_similarity(q_emb, n_emb)
        margin = pos_sim - neg_sim

        print(f"\n  [{label}] Overfit metrics ({len(samples)} samples):")
        print(f"    Avg cos_sim(query, positive):  {pos_sim.mean().item():.4f}")
        print(f"    Avg cos_sim(query, negative):  {neg_sim.mean().item():.4f}")
        print(f"    Avg margin (pos - neg):        {margin.mean().item():.4f}")
        print(f"    Samples where pos > neg:       {(margin > 0).sum().item()}/{len(samples)}")

        # Per-sample details if small enough
        if len(samples) <= 10:
            print(f"    {'Sample':<8} {'pos_sim':>8} {'neg_sim':>8} {'margin':>8} {'correct':>8}")
            print(f"    {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
            for i in range(len(samples)):
                correct = "yes" if margin[i].item() > 0 else "NO"
                print(f"    {i:<8} {pos_sim[i].item():>8.4f} {neg_sim[i].item():>8.4f} "
                      f"{margin[i].item():>8.4f} {correct:>8}")
        print()

        self.model.train()

    def _encode_batch(self, texts: list[str] | tuple[str, ...]) -> torch.Tensor:
        """Tokenize and encode a batch of texts, returning L2-normalized embeddings."""
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        embeddings = _pool(
            outputs.last_hidden_state,
            encoded["attention_mask"],
            self.pooling_mode,
        )
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def _compute_grad_norm(self) -> float:
        """Compute total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Linear warmup then linear decay schedule."""
        warmup = self.config.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            remaining = total_steps - step
            total_decay = total_steps - warmup
            return max(0.0, remaining / max(total_decay, 1))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
