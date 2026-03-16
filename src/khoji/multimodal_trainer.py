"""Training loop for fine-tuning multimodal models (text-to-image)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from khoji.device import get_device
from khoji.image_utils import build_image_processor, load_images_batch
from khoji.lora import LoRASettings, apply_lora
from khoji.loss import triplet_margin_loss
from khoji.model import _resolve_dtype
from khoji.multimodal_data import MultimodalTripletDataset
from khoji.trainer import TrainHistory


@dataclass
class MultimodalTrainingConfig:
    """Training hyperparameters for multimodal fine-tuning.

    Args:
        lora_target: Which encoder(s) to apply LoRA to: "vision", "text", or "both".
        cache_dir: Directory for caching downloaded images during training.
    """

    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    max_length: int = 77  # CLIP default
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
    base_dir: str | None = None  # for resolving relative image paths


class MultimodalTrainer:
    """Fine-tunes a multimodal model on text-image triplets.

    **HuggingFace CLIP/SigLIP models:**

        MultimodalTrainer("openai/clip-vit-base-patch32", config)

    **Custom models:**

        MultimodalTrainer(
            text_model=my_text_encoder,
            vision_model=my_vision_encoder,
            tokenizer=my_tokenizer,
            image_processor=my_processor,
            config=config,
        )
    """

    def __init__(
        self,
        model_name: str | None = None,
        config: MultimodalTrainingConfig | None = None,
        text_model: torch.nn.Module | None = None,
        vision_model: torch.nn.Module | None = None,
        tokenizer: object | None = None,
        image_processor: Callable | None = None,
        preprocess_overrides: dict | None = None,
    ):
        self.model_name = model_name or "custom"
        self.config = config or MultimodalTrainingConfig()
        self.device = get_device()

        if text_model is not None and vision_model is not None:
            # Custom model path
            if tokenizer is None:
                raise ValueError("Must provide tokenizer with custom models.")
            if image_processor is None:
                raise ValueError("Must provide image_processor with custom models.")
            self.tokenizer = tokenizer
            self.image_processor = image_processor
            self._full_model = None
            self.text_encoder = text_model.to(self.device)
            self.vision_encoder = vision_model.to(self.device)

        elif model_name is not None:
            self._load_hf_model(model_name, preprocess_overrides)
            # Allow custom image_processor to override the auto-detected one
            if image_processor is not None:
                self.image_processor = image_processor

        else:
            raise ValueError("Provide either model_name or (text_model + vision_model).")

        # Apply LoRA
        self._apply_lora()

        # Mixed precision setup
        self.amp_dtype = None
        self.scaler = None
        if self.config.mixed_precision is not None:
            if self.config.mixed_precision == "fp16":
                self.amp_dtype = torch.float16
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
        print(
            f"Effective batch size: {effective_bs} "
            f"(micro={self.config.batch_size} x accum={self.config.grad_accum_steps})"
            f"{amp_str}"
        )

    def _load_hf_model(self, model_name: str, preprocess_overrides: dict | None) -> None:
        """Load a CLIP/SigLIP model from HuggingFace."""
        from transformers import AutoConfig, AutoTokenizer

        model_config = AutoConfig.from_pretrained(model_name)
        model_type = getattr(model_config, "model_type", "unknown")

        load_kwargs = {}
        torch_dtype = _resolve_dtype(self.config.dtype)
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        if model_type == "clip":
            from transformers import CLIPModel

            self._full_model = CLIPModel.from_pretrained(model_name, **load_kwargs).to(self.device)
        elif model_type == "siglip":
            from transformers import SiglipModel

            self._full_model = SiglipModel.from_pretrained(model_name, **load_kwargs).to(self.device)
        else:
            from transformers import AutoModel

            self._full_model = AutoModel.from_pretrained(model_name, **load_kwargs).to(self.device)

        self.text_encoder = self._full_model.text_model
        self.vision_encoder = self._full_model.vision_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.image_processor = build_image_processor(
            model_name=model_name, overrides=preprocess_overrides
        )

        dtype_str = f" | dtype: {self.config.dtype}" if self.config.dtype else ""
        print(f"Loaded {model_name} | type: {model_type} | device: {self.device}{dtype_str}")

    def _apply_lora(self) -> None:
        """Apply LoRA to the full model targeting text/vision/both encoders."""
        if self.config.lora is None:
            return

        if self._full_model is None:
            # Custom models: apply LoRA to sub-models directly
            target = self.config.lora_target
            if target in ("text", "both"):
                self.text_encoder = apply_lora(self.text_encoder, self.config.lora)
            if target in ("vision", "both"):
                self.vision_encoder = apply_lora(self.vision_encoder, self.config.lora)
            return

        # HuggingFace models: apply LoRA to the full model, then freeze
        # the sub-model we don't want to train
        from peft import LoraConfig, TaskType, get_peft_model

        target = self.config.lora_target
        base_modules = self.config.lora.target_modules or ["q_proj", "k_proj", "v_proj"]

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=base_modules,
        )

        self._full_model = get_peft_model(self._full_model, lora_config)

        # Freeze the sub-model(s) we don't want to train LoRA on
        if target == "vision":
            for name, param in self._full_model.named_parameters():
                if param.requires_grad and "text_model" in name:
                    param.requires_grad = False
        elif target == "text":
            for name, param in self._full_model.named_parameters():
                if param.requires_grad and "vision_model" in name:
                    param.requires_grad = False

        self.text_encoder = self._full_model.base_model.model.text_model
        self.vision_encoder = self._full_model.base_model.model.vision_model

        trainable = sum(p.numel() for p in self._full_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._full_model.parameters())
        print(
            f"LoRA applied | target: {target} | modules: {base_modules} | "
            f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
        )

    def train(self, dataset: MultimodalTripletDataset) -> TrainHistory:
        """Run the training loop.

        Args:
            dataset: MultimodalTripletDataset of (query_text, pos_image, neg_image).

        Returns:
            TrainHistory with per-step loss, lr, grad norm, and per-epoch loss.
        """
        history = TrainHistory()

        # Overfit mode
        overfit_samples = None
        if self.config.overfit_batches is not None:
            n = self.config.overfit_batches
            subset = [dataset[i] for i in range(min(n * self.config.batch_size, len(dataset)))]
            from khoji.multimodal_data import MultimodalTriplet

            dataset = MultimodalTripletDataset([MultimodalTriplet(*s) for s in subset])
            overfit_samples = subset
            print(f"\n[OVERFIT MODE] Training on {len(dataset)} samples for {self.config.epochs} epochs")
            self._overfit_report("BEFORE training", overfit_samples)
        else:
            print(f"\nTraining on {len(dataset)} triplets for {self.config.epochs} epochs")
            if self.config.sanity_check_samples > 0:
                import random
                n_check = min(self.config.sanity_check_samples, len(dataset))
                indices = random.sample(range(len(dataset)), n_check)
                overfit_samples = [dataset[i] for i in indices]
                self._overfit_report("BEFORE training", overfit_samples)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.overfit_batches is None,
        )

        # Collect trainable params
        if self._full_model is not None:
            trainable_params = [p for p in self._full_model.parameters() if p.requires_grad]
        else:
            trainable_params = []
            for p in self.text_encoder.parameters():
                if p.requires_grad:
                    trainable_params.append(p)
            for p in self.vision_encoder.parameters():
                if p.requires_grad:
                    trainable_params.append(p)

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        steps_per_epoch = (len(dataloader) + self.config.grad_accum_steps - 1) // self.config.grad_accum_steps
        total_opt_steps = steps_per_epoch * self.config.epochs
        total_batches = len(dataloader) * self.config.epochs
        scheduler = self._build_scheduler(optimizer, total_opt_steps)

        print(
            f"Total batches: {total_batches} | "
            f"Optimizer steps: {total_opt_steps} | "
            f"Grad clipping: {self.config.max_grad_norm}\n"
        )

        # Set to train mode
        self.text_encoder.train()
        self.vision_encoder.train()
        global_batch = 0
        global_opt_step = 0
        accum_loss = 0.0

        pbar = tqdm(total=total_batches, desc="Training", unit="batch")

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            optimizer.zero_grad()

            for batch_idx, (queries, pos_sources, neg_sources) in enumerate(dataloader):
                if self.amp_dtype is not None:
                    with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        query_emb = self._encode_text_batch(queries)
                        pos_emb = self._encode_image_batch(pos_sources)
                        neg_emb = self._encode_image_batch(neg_sources)
                        loss = self.config.loss_fn(query_emb, pos_emb, neg_emb)
                else:
                    query_emb = self._encode_text_batch(queries)
                    pos_emb = self._encode_image_batch(pos_sources)
                    neg_emb = self._encode_image_batch(neg_sources)
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

                is_accum_step = (batch_idx + 1) % self.config.grad_accum_steps == 0
                is_last_batch = (batch_idx + 1) == len(dataloader)

                if is_accum_step or is_last_batch:
                    if self.scaler is not None:
                        self.scaler.unscale_(optimizer)

                    grad_norm = 0.0
                    if self.config.max_grad_norm is not None:
                        if self._full_model is not None:
                            all_params = list(self._full_model.parameters())
                        else:
                            all_params = list(self.text_encoder.parameters()) + list(self.vision_encoder.parameters())
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            all_params, self.config.max_grad_norm
                        ).item()

                    if self.scaler is not None:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    global_opt_step += 1

                    n_accum = (batch_idx % self.config.grad_accum_steps) + 1
                    step_loss = accum_loss / n_accum
                    step_lr = scheduler.get_last_lr()[0]

                    history.step_loss.append(step_loss)
                    history.step_lr.append(step_lr)
                    history.step_grad_norm.append(grad_norm)
                    accum_loss = 0.0

                    if (
                        self.config.save_every_n_steps is not None
                        and self.config.save_dir is not None
                        and global_opt_step % self.config.save_every_n_steps == 0
                    ):
                        self._save_checkpoint(global_opt_step)

                pbar.update(1)
                pbar.set_postfix(
                    epoch=f"{epoch + 1}/{self.config.epochs}",
                    loss=f"{batch_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            epoch_avg = epoch_loss / max(epoch_batches, 1)
            history.epoch_loss.append(epoch_avg)
            tqdm.write(
                f"  Epoch {epoch + 1}/{self.config.epochs} complete | "
                f"Avg Loss: {epoch_avg:.4f}"
            )

        pbar.close()

        # Show after-training metrics
        if overfit_samples is not None:
            self._overfit_report("AFTER training", overfit_samples)

        if self.config.save_dir:
            self.save(self.config.save_dir)

        return history

    def save(self, path: str) -> None:
        """Save the model weights."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self._full_model is not None:
            self._full_model.save_pretrained(save_path)
        else:
            # Custom models: save both encoders
            torch.save(self.text_encoder.state_dict(), save_path / "text_encoder.pt")
            torch.save(self.vision_encoder.state_dict(), save_path / "vision_encoder.pt")

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
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if self._full_model is not None:
            self._full_model.save_pretrained(ckpt_dir)
        else:
            torch.save(self.text_encoder.state_dict(), ckpt_dir / "text_encoder.pt")
            torch.save(self.vision_encoder.state_dict(), ckpt_dir / "vision_encoder.pt")
        self.tokenizer.save_pretrained(ckpt_dir)
        tqdm.write(f"  Checkpoint saved to {ckpt_dir}")

    @torch.no_grad()
    def _overfit_report(self, label: str, samples: list[tuple[str, str, str]]) -> None:
        """Print cosine similarity metrics for sanity check samples.

        Shows per-sample cos_sim(query, positive_image) and cos_sim(query, negative_image),
        plus the margin (pos - neg) and whether the model ranks the positive higher.
        """
        self.text_encoder.eval()
        self.vision_encoder.eval()

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
        print(f"    Avg cos_sim(query, pos_image):   {pos_sim.mean().item():.4f}")
        print(f"    Avg cos_sim(query, neg_image):   {neg_sim.mean().item():.4f}")
        print(f"    Avg margin (pos - neg):          {margin.mean().item():.4f}")
        print(f"    Samples where pos > neg:         {(margin > 0).sum().item()}/{len(samples)}")

        if len(samples) <= 10:
            print(f"    {'Sample':<8} {'pos_sim':>8} {'neg_sim':>8} {'margin':>8} {'correct':>8}")
            print(f"    {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
            for i in range(len(samples)):
                correct = "yes" if margin[i].item() > 0 else "NO"
                print(
                    f"    {i:<8} {pos_sim[i].item():>8.4f} {neg_sim[i].item():>8.4f} "
                    f"{margin[i].item():>8.4f} {correct:>8}"
                )
        print()

        self.text_encoder.train()
        self.vision_encoder.train()

    def _encode_text_batch(self, texts: list[str] | tuple[str, ...]) -> torch.Tensor:
        """Encode a batch of text queries."""
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.device)

        embeddings = self._extract_text_features(encoded)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def _encode_image_batch(self, image_sources: list[str] | tuple[str, ...]) -> torch.Tensor:
        """Load and encode a batch of images."""
        images = load_images_batch(
            list(image_sources),
            base_dir=self.config.base_dir,
            cache_dir=self.config.cache_dir,
        )
        pixel_values = self.image_processor(images).to(self.device)

        embeddings = self._extract_image_features(pixel_values)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def _extract_text_features(self, encoded: dict) -> torch.Tensor:
        """Extract text embeddings from tokenized inputs."""
        if self._full_model is not None:
            # Access the underlying model (unwrap peft if needed)
            base = self._full_model
            if hasattr(base, "base_model"):
                base = base.base_model.model
            outputs = base.text_model(**encoded)
            pooled = outputs.pooler_output
            text_proj = getattr(base, "text_projection", None)
            if text_proj is not None:
                return text_proj(pooled)
            return pooled
        else:
            outputs = self.text_encoder(**encoded)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state[:, 0, :]
            return outputs

    def _extract_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract image embeddings from pixel values."""
        if self._full_model is not None:
            base = self._full_model
            if hasattr(base, "base_model"):
                base = base.base_model.model
            outputs = base.vision_model(pixel_values=pixel_values)
            pooled = outputs.pooler_output
            visual_proj = getattr(base, "visual_projection", None)
            if visual_proj is not None:
                return visual_proj(pooled)
            return pooled
        else:
            outputs = self.vision_encoder(pixel_values)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state[:, 0, :]
            return outputs

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
