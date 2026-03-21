"""khoji: Fine-tune embedding models for domain-specific retrieval."""

__version__ = "0.2.0"

from khoji.config import ForgeConfig
from khoji.data import (
    Triplet,
    TripletDataset,
    build_mixed_negatives,
    build_random_negatives,
    mine_hard_negatives,
)
from khoji.dataset import RetrievalDataset, load_beir, load_custom
from khoji.evaluator import EvalResult, Evaluator
from khoji.lora import LoRASettings
from khoji.loss import contrastive_loss, infonce_loss, triplet_margin_loss
from khoji.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from khoji.model import EmbeddingModel
from khoji.run import RunResult, run
from khoji.trainer import Trainer, TrainHistory, TrainingConfig

# Multimodal (text-to-image)
from khoji.multimodal_config import MultimodalForgeConfig
from khoji.multimodal_data import (
    MultimodalTriplet,
    MultimodalTripletDataset,
    build_mixed_negatives_multimodal,
    build_random_negatives_multimodal,
    mine_hard_negatives_multimodal,
)
from khoji.multimodal_dataset import MultimodalRetrievalDataset, load_flickr30k, load_rsicd, load_custom_multimodal
from khoji.multimodal_evaluator import MultimodalEvaluator
from khoji.multimodal_model import JointEmbeddingModel, MultimodalEmbeddingModel
from khoji.multimodal_run import run_multimodal
from khoji.multimodal_trainer import MultimodalTrainer, MultimodalTrainingConfig
from khoji.image_utils import load_image, load_images_batch

# Composed (image+text → image)
from khoji.composed_config import ComposedForgeConfig
from khoji.composed_data import (
    ComposedTriplet,
    ComposedTripletDataset,
    build_mixed_negatives_composed,
    build_random_negatives_composed,
    mine_hard_negatives_composed,
)
from khoji.composed_dataset import ComposedRetrievalDataset, load_custom_composed
from khoji.composed_evaluator import ComposedEvaluator
from khoji.composed_run import run_composed
from khoji.composed_trainer import ComposedTrainer, ComposedTrainingConfig

__all__ = [
    # Text-text
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
    "build_mixed_negatives",
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
    # Multimodal (text-to-image)
    "JointEmbeddingModel",
    "MultimodalEmbeddingModel",
    "MultimodalEvaluator",
    "MultimodalForgeConfig",
    "MultimodalRetrievalDataset",
    "MultimodalTrainer",
    "MultimodalTrainingConfig",
    "MultimodalTriplet",
    "MultimodalTripletDataset",
    "build_mixed_negatives_multimodal",
    "build_random_negatives_multimodal",
    "load_custom_multimodal",
    "load_flickr30k",
    "load_rsicd",
    "load_image",
    "load_images_batch",
    "mine_hard_negatives_multimodal",
    "run_multimodal",
    # Composed (image+text → image)
    "ComposedEvaluator",
    "ComposedForgeConfig",
    "ComposedRetrievalDataset",
    "ComposedTrainer",
    "ComposedTrainingConfig",
    "ComposedTriplet",
    "ComposedTripletDataset",
    "build_mixed_negatives_composed",
    "build_random_negatives_composed",
    "load_custom_composed",
    "mine_hard_negatives_composed",
    "run_composed",
]
