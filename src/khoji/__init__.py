"""khoji: Fine-tune embedding models for domain-specific retrieval."""

__version__ = "0.1.1"

from khoji.config import ForgeConfig
from khoji.data import Triplet, TripletDataset, build_random_negatives, mine_hard_negatives
from khoji.dataset import RetrievalDataset, load_beir, load_custom
from khoji.evaluator import EvalResult, Evaluator
from khoji.lora import LoRASettings
from khoji.loss import contrastive_loss, infonce_loss, triplet_margin_loss
from khoji.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from khoji.model import EmbeddingModel
from khoji.run import RunResult, run
from khoji.trainer import TrainHistory, Trainer, TrainingConfig

__all__ = [
    "EmbeddingModel",
    "EvalResult",
    "Evaluator",
    "ForgeConfig",
    "LoRASettings",
    "RetrievalDataset",
    "RunResult",
    "TrainHistory",
    "Trainer",
    "TrainingConfig",
    "Triplet",
    "TripletDataset",
    "build_random_negatives",
    "contrastive_loss",
    "infonce_loss",
    "load_beir",
    "load_custom",
    "mine_hard_negatives",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "run",
    "triplet_margin_loss",
]
