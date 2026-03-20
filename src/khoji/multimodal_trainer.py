"""Training loop for multimodal (text-to-image) fine-tuning."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from khoji.device import get_device
from khoji.image_utils import build_image_processor, load_images_batch
from khoji.lora import LoRASettings, apply_lora
from khoji.loss import triplet_margin_loss
from khoji.model import _resolve_dtype
from khoji.multimodal_data import MultimodalTripletDataset
from khoji.trainer import TrainHistory


@dataclass
class MultimodalTrainingConfig:
    """Training hyperparameters for multimodal fine-tuning."""

    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    max_length: int = 77
    mixed_precision: str | None = None
    loss_fn: Callable[..., torch.Tensor] = triplet_margin_loss
    lora: LoRASettings | None = None
    lora_target: str = "both"  # "vision", "text", or "both"
    save_dir: str | None = None
    overfit_batches: int | None = None
    sanity_check_samples: int = 10
    save_every_n_steps: int | None = None
    keep_all_checkpoints: bool = False
    dtype: str | None = None
    cache_dir: str | None = None
    base_dir: str | None = None


class MultimodalTrainer:
    """Fine-tunes a multimodal model on text-image triplets.

    **HuggingFace CLIP/SigLIP** (auto-wires encode functions):

        trainer = MultimodalTrainer("openai/clip-vit-base-patch32", config)

    **Custom model** — provide model + encode functions:

        trainer = MultimodalTrainer(
            model=my_clip_model,
            encode_text_fn=my_text_fn,    # (texts: list[str]) -> Tensor
            encode_image_fn=my_image_fn,  # (sources: list[str]) -> Tensor
            config=config,
        )

    Encode functions are called with gradients enabled during training.
    Library handles L2 normalization.

    Args:
        model_name: HuggingFace model ID (CLIP/SigLIP).
        config: Training configuration.
        model: Custom nn.Module for parameters/LoRA/save.
        encode_text_fn: Custom text encoding function.
            ``(texts: list[str]) -> Tensor (batch, dim)``.
        encode_image_fn: Custom image encoding function.
            ``(sources: list[str]) -> Tensor (batch, dim)``.
            Receives image source paths/URLs, not PIL images.
        preprocess_overrides: Dict with image_size, mean, std overrides.
        adapter_path: Path to saved LoRA adapter for warm-starting.
    """

    def __init__(
        self,
        model_name: str | None = None,
        config: MultimodalTrainingConfig | None = None,
        model: torch.nn.Module | None = None,
        encode_text_fn: (
            Callable[[list[str]], torch.Tensor] | None
        ) = None,
        encode_image_fn: (
            Callable[[list[str]], torch.Tensor] | None
        ) = None,
        preprocess_overrides: dict | None = None,
        adapter_path: str | None = None,
    ):
        self.model_name = model_name or "custom"
        self.config = config or MultimodalTrainingConfig()
        self.device = get_device()

        if model is not None and encode_text_fn is not None and encode_image_fn is not None:
            # Custom model + encode functions
            self.model = model.to(self.device)
            self._encode_text_fn = encode_text_fn
            self._encode_image_fn = encode_image_fn

        elif model_name is not None:
            # HuggingFace convenience path
            self._load_hf_model(model_name, preprocess_overrides)

        else:
            raise ValueError(
                "Provide either model_name or "
                "(model + encode_text_fn + encode_image_fn)."
            )

        # Apply LoRA
        self._apply_lora(adapter_path=adapter_path)

        # Mixed precision
        self.amp_dtype = None
        self.scaler = None
        if self.config.mixed_precision is not None:
            if self.config.mixed_precision == "fp16":
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler(
                    device=self.device.type
                )
            elif self.config.mixed_precision == "bf16":
                self.amp_dtype = torch.bfloat16
            else:
                raise ValueError(
                    f"Unknown mixed_precision: "
                    f"{self.config.mixed_precision}"
                )

        effective_bs = (
            self.config.batch_size * self.config.grad_accum_steps
        )
        amp_str = (
            f" | AMP: {self.config.mixed_precision}"
            if self.config.mixed_precision else ""
        )
        print(
            f"Effective batch size: {effective_bs} "
            f"(micro={self.config.batch_size} "
            f"x accum={self.config.grad_accum_steps}){amp_str}"
        )

    def _load_hf_model(
        self, model_name: str, preprocess_overrides: dict | None
    ) -> None:
        """Load CLIP/SigLIP and wire up encode functions."""
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(model_name)
        model_type = getattr(model_config, "model_type", "unknown")

        load_kwargs = {}
        torch_dtype = _resolve_dtype(self.config.dtype)
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        if model_type == "clip":
            from transformers import CLIPModel
            full_model = CLIPModel.from_pretrained(
                model_name, **load_kwargs
            ).to(self.device)
        elif model_type == "siglip":
            from transformers import SiglipModel
            full_model = SiglipModel.from_pretrained(
                model_name, **load_kwargs
            ).to(self.device)
        else:
            from transformers import AutoModel
            full_model = AutoModel.from_pretrained(
                model_name, **load_kwargs
            ).to(self.device)

        self.model = full_model
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._image_processor = build_image_processor(
            model_name=model_name, overrides=preprocess_overrides,
        )

        # Wire encode functions via closures
        device = self.device
        max_length = self.config.max_length
        tokenizer = self._tokenizer
        image_processor = self._image_processor
        base_dir = self.config.base_dir
        cache_dir = self.config.cache_dir

        def _encode_text(texts: list[str]) -> torch.Tensor:
            encoded = tokenizer(
                list(texts), padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            # Access text_model through self.model (may be wrapped by peft)
            base = self.model
            if hasattr(base, "base_model"):
                base = base.base_model.model
            outputs = base.text_model(**encoded)
            pooled = outputs.pooler_output
            text_proj = getattr(base, "text_projection", None)
            if text_proj is not None:
                return text_proj(pooled)
            return pooled

        def _encode_images(sources: list[str]) -> torch.Tensor:
            images = load_images_batch(
                list(sources), base_dir=base_dir, cache_dir=cache_dir,
            )
            pixel_values = image_processor(images).to(device)
            base = self.model
            if hasattr(base, "base_model"):
                base = base.base_model.model
            outputs = base.vision_model(pixel_values=pixel_values)
            pooled = outputs.pooler_output
            visual_proj = getattr(base, "visual_projection", None)
            if visual_proj is not None:
                return visual_proj(pooled)
            return pooled

        self._encode_text_fn = _encode_text
        self._encode_image_fn = _encode_images

        dtype_str = (
            f" | dtype: {self.config.dtype}"
            if self.config.dtype else ""
        )
        print(
            f"Loaded {model_name} | type: {model_type} | "
            f"device: {self.device}{dtype_str}"
        )

    def _apply_lora(self, adapter_path: str | None = None) -> None:
        """Apply LoRA to the model."""
        if self.config.lora is None:
            return

        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model, adapter_path, is_trainable=True
            )
            trainable = sum(
                p.numel()
                for p in self.model.parameters()
                if p.requires_grad
            )
            total = sum(p.numel() for p in self.model.parameters())
            print(
                f"LoRA warm-start from {adapter_path} | "
                f"trainable: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.2f}%)"
            )
            return

        # Apply fresh LoRA
        from peft import LoraConfig, TaskType, get_peft_model

        target = self.config.lora_target
        base_modules = (
            self.config.lora.target_modules
            or ["q_proj", "k_proj", "v_proj"]
        )

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=base_modules,
        )

        self.model = get_peft_model(self.model, lora_config)

        # Freeze sub-model(s) based on lora_target
        if target == "vision":
            for name, param in self.model.named_parameters():
                if param.requires_grad and "text_model" in name:
                    param.requires_grad = False
        elif target == "text":
            for name, param in self.model.named_parameters():
                if param.requires_grad and "vision_model" in name:
                    param.requires_grad = False

        trainable = sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )
        total = sum(p.numel() for p in self.model.parameters())
        print(
            f"LoRA applied | target: {target} | "
            f"modules: {base_modules} | "
            f"trainable: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    def _encode_text_batch(self, texts):
        """Encode text batch and normalize."""
        emb = self._encode_text_fn(list(texts))
        return torch.nn.functional.normalize(emb, p=2, dim=1)

    def _encode_image_batch(self, sources):
        """Encode image batch and normalize."""
        emb = self._encode_image_fn(list(sources))
        return torch.nn.functional.normalize(emb, p=2, dim=1)

    def train(self, dataset: MultimodalTripletDataset) -> TrainHistory:
        """Run the training loop."""
        history = TrainHistory()

        overfit_samples = None
        if self.config.overfit_batches is not None:
            n = self.config.overfit_batches
            subset = [
                dataset[i]
                for i in range(
                    min(n * self.config.batch_size, len(dataset))
                )
            ]
            from khoji.multimodal_data import MultimodalTriplet
            dataset = MultimodalTripletDataset(
                [MultimodalTriplet(*s) for s in subset]
            )
            overfit_samples = subset
            print(
                f"\n[OVERFIT MODE] Training on {len(dataset)} samples "
                f"for {self.config.epochs} epochs"
            )
            self._overfit_report("BEFORE training", overfit_samples)
        else:
            print(
                f"\nTraining on {len(dataset)} triplets "
                f"for {self.config.epochs} epochs"
            )
            if self.config.sanity_check_samples > 0:
                import random
                n_check = min(
                    self.config.sanity_check_samples, len(dataset)
                )
                indices = random.sample(range(len(dataset)), n_check)
                overfit_samples = [dataset[i] for i in indices]
                self._overfit_report(
                    "BEFORE training", overfit_samples
                )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.overfit_batches is None,
        )

        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        steps_per_epoch = (
            (len(dataloader) + self.config.grad_accum_steps - 1)
            // self.config.grad_accum_steps
        )
        total_opt_steps = steps_per_epoch * self.config.epochs
        total_batches = len(dataloader) * self.config.epochs
        scheduler = self._build_scheduler(optimizer, total_opt_steps)

        print(
            f"Total batches: {total_batches} | "
            f"Optimizer steps: {total_opt_steps} | "
            f"Grad clipping: {self.config.max_grad_norm}\n"
        )

        self.model.train()
        global_opt_step = 0
        accum_loss = 0.0

        pbar = tqdm(total=total_batches, desc="Training", unit="batch")

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            optimizer.zero_grad()

            for batch_idx, (queries, pos_src, neg_src) in enumerate(
                dataloader
            ):
                if self.amp_dtype is not None:
                    with torch.amp.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                    ):
                        q = self._encode_text_batch(queries)
                        p = self._encode_image_batch(pos_src)
                        n = self._encode_image_batch(neg_src)
                        loss = self.config.loss_fn(q, p, n)
                else:
                    q = self._encode_text_batch(queries)
                    p = self._encode_image_batch(pos_src)
                    n = self._encode_image_batch(neg_src)
                    loss = self.config.loss_fn(q, p, n)

                scaled = loss / self.config.grad_accum_steps
                if self.scaler is not None:
                    self.scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                bl = loss.item()
                accum_loss += bl
                epoch_loss += bl
                epoch_batches += 1

                is_accum = (
                    (batch_idx + 1) % self.config.grad_accum_steps == 0
                )
                is_last = (batch_idx + 1) == len(dataloader)

                if is_accum or is_last:
                    if self.scaler is not None:
                        self.scaler.unscale_(optimizer)

                    grad_norm = 0.0
                    if self.config.max_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        ).item()

                    if self.scaler is not None:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    global_opt_step += 1

                    n_accum = (
                        batch_idx % self.config.grad_accum_steps
                    ) + 1
                    history.step_loss.append(accum_loss / n_accum)
                    history.step_lr.append(
                        scheduler.get_last_lr()[0]
                    )
                    history.step_grad_norm.append(grad_norm)
                    accum_loss = 0.0

                    if (
                        self.config.save_every_n_steps is not None
                        and self.config.save_dir is not None
                        and global_opt_step
                        % self.config.save_every_n_steps == 0
                    ):
                        self._save_checkpoint(global_opt_step)

                pbar.update(1)
                pbar.set_postfix(
                    epoch=f"{epoch+1}/{self.config.epochs}",
                    loss=f"{bl:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            avg = epoch_loss / max(epoch_batches, 1)
            history.epoch_loss.append(avg)
            tqdm.write(
                f"  Epoch {epoch+1}/{self.config.epochs} complete | "
                f"Avg Loss: {avg:.4f}"
            )

        pbar.close()

        if overfit_samples is not None:
            self._overfit_report("AFTER training", overfit_samples)

        if self.config.save_dir:
            self.save(self.config.save_dir)

        return history

    def save(self, path: str) -> None:
        """Save model weights."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        if hasattr(self, "_tokenizer") and self._tokenizer is not None:
            self._tokenizer.save_pretrained(save_path)
        label = (
            "LoRA adapter" if self.config.lora is not None
            else "Full model"
        )
        print(f"{label} saved to {save_path}")

    def _save_checkpoint(self, step: int) -> None:
        """Save a checkpoint during training."""
        base = Path(self.config.save_dir)
        if self.config.keep_all_checkpoints:
            ckpt_dir = base / f"checkpoint-{step}"
        else:
            ckpt_dir = base / "checkpoint-latest"
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        tqdm.write(f"  Checkpoint saved to {ckpt_dir}")

    @torch.no_grad()
    def _overfit_report(self, label, samples):
        """Print cosine similarity metrics for sanity check."""
        self.model.eval()

        queries = [s[0] for s in samples]
        positives = [s[1] for s in samples]
        negatives = [s[2] for s in samples]

        q_emb = self._encode_text_batch(queries)
        p_emb = self._encode_image_batch(positives)
        n_emb = self._encode_image_batch(negatives)

        pos_sim = torch.nn.functional.cosine_similarity(q_emb, p_emb)
        neg_sim = torch.nn.functional.cosine_similarity(q_emb, n_emb)
        margin = pos_sim - neg_sim

        print(f"\n  [{label}] Sanity check ({len(samples)} samples):")
        print(
            f"    Avg cos_sim(query, pos):  "
            f"{pos_sim.mean().item():.4f}"
        )
        print(
            f"    Avg cos_sim(query, neg):  "
            f"{neg_sim.mean().item():.4f}"
        )
        print(
            f"    Avg margin (pos - neg):   "
            f"{margin.mean().item():.4f}"
        )
        print(
            f"    Samples where pos > neg:  "
            f"{(margin > 0).sum().item()}/{len(samples)}"
        )

        if len(samples) <= 10:
            print(
                f"    {'Sample':<8} {'pos_sim':>8} {'neg_sim':>8} "
                f"{'margin':>8} {'correct':>8}"
            )
            print(f"    {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
            for i in range(len(samples)):
                c = "yes" if margin[i].item() > 0 else "NO"
                print(
                    f"    {i:<8} {pos_sim[i].item():>8.4f} "
                    f"{neg_sim[i].item():>8.4f} "
                    f"{margin[i].item():>8.4f} {c:>8}"
                )
        print()

        self.model.train()

    def _build_scheduler(self, optimizer, total_steps):
        """Linear warmup then linear decay."""
        warmup = self.config.warmup_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            remaining = total_steps - step
            total_decay = total_steps - warmup
            return max(0.0, remaining / max(total_decay, 1))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
