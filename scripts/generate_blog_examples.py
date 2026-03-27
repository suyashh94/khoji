"""Generate before/after retrieval examples for blog posts.

Creates professional visualizations showing:
1. Multimodal (RSICD): text query → top-5 satellite images, baseline vs fine-tuned
2. Composed (FashionIQ): reference image + caption → top-5 gallery images, baseline vs fine-tuned
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from PIL import Image

import khoji
from khoji.image_utils import load_image
from khoji.multimodal_dataset import load_rsicd

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.bbox': 'tight', 'savefig.dpi': 150, 'savefig.pad_inches': 0.3,
})

BLUE = '#2563EB'
GREEN = '#16A34A'
ORANGE = '#EA580C'
GRAY = '#6B7280'
RED = '#DC2626'
OUTDIR = Path("blog/khoji-technical-guide/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# MULTIMODAL: Text → Image (RSICD Satellite)
# ═══════════════════════════════════════════════════════════

def generate_multimodal_examples():
    """Generate before/after satellite image retrieval examples."""
    print("=" * 60)
    print("MULTIMODAL EXAMPLES (RSICD)")
    print("=" * 60)

    test_ds = load_rsicd(split="test")
    corpus_ids = list(test_ds.corpus.keys())
    corpus_sources = list(test_ds.corpus.values())

    # Load baseline and fine-tuned models
    print("Loading baseline CLIP...")
    model_base = khoji.MultimodalEmbeddingModel("openai/clip-vit-base-patch32")
    print("Loading fine-tuned CLIP...")
    model_ft = khoji.MultimodalEmbeddingModel(
        "openai/clip-vit-base-patch32",
        adapter_path="output/multimodal-retrieval/config-approach/adapter"
    )

    # Encode all corpus images once per model
    print("Encoding corpus with baseline...")
    img_emb_base = model_ft.encode_image_sources(
        corpus_sources, base_dir=test_ds.base_dir, batch_size=64
    )
    # Re-encode with fine-tuned
    print("Encoding corpus with fine-tuned...")
    img_emb_ft = model_ft.encode_image_sources(
        corpus_sources, base_dir=test_ds.base_dir, batch_size=64
    )
    # Baseline corpus embeddings
    img_emb_base = model_base.encode_image_sources(
        corpus_sources, base_dir=test_ds.base_dir, batch_size=64
    )

    # Pick interesting queries
    queries = [
        "an airport with multiple runways and terminals",
        "a dense residential area with many buildings",
        "a river flowing through green farmland",
    ]

    for qi, query in enumerate(queries):
        print(f"\nQuery {qi+1}: '{query}'")

        # Encode query with both models
        q_base = model_base.encode_text([query])
        q_ft = model_ft.encode_text([query])

        # Rank
        scores_base = torch.mm(q_base, img_emb_base.t()).squeeze(0)
        scores_ft = torch.mm(q_ft, img_emb_ft.t()).squeeze(0)

        top5_base = torch.topk(scores_base, 5).indices.tolist()
        top5_ft = torch.topk(scores_ft, 5).indices.tolist()

        # Load images (skip None on failure)
        imgs_base = []
        for idx in top5_base:
            img = load_image(corpus_sources[idx], base_dir=test_ds.base_dir)
            if img is not None:
                imgs_base.append(img)

        imgs_ft = []
        for idx in top5_ft:
            img = load_image(corpus_sources[idx], base_dir=test_ds.base_dir)
            if img is not None:
                imgs_ft.append(img)

        # Create figure
        fig = plt.figure(figsize=(16, 7))
        gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.1,
                               width_ratios=[0.15, 1, 1, 1, 1, 1])

        # Title
        fig.suptitle(f'Query: "{query}"', fontsize=14, fontweight='bold', y=0.98)

        # Row labels
        ax_label1 = fig.add_subplot(gs[0, 0])
        ax_label1.axis('off')
        ax_label1.text(0.5, 0.5, 'Before\nFine-Tuning', ha='center', va='center',
                      fontsize=12, fontweight='bold', color=GRAY,
                      transform=ax_label1.transAxes)

        ax_label2 = fig.add_subplot(gs[1, 0])
        ax_label2.axis('off')
        ax_label2.text(0.5, 0.5, 'After\nFine-Tuning', ha='center', va='center',
                      fontsize=12, fontweight='bold', color=GREEN,
                      transform=ax_label2.transAxes)

        # Baseline row
        for i, (img, idx) in enumerate(zip(imgs_base, top5_base)):
            ax = fig.add_subplot(gs[0, i + 1])
            ax.imshow(img)
            ax.set_title(f'#{i+1}  (score: {scores_base[idx]:.3f})',
                        fontsize=9, color=GRAY)
            ax.axis('off')
            # Gray border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#D1D5DB')
                spine.set_linewidth(2)

        # Fine-tuned row
        for i, (img, idx) in enumerate(zip(imgs_ft, top5_ft)):
            ax = fig.add_subplot(gs[1, i + 1])
            ax.imshow(img)
            ax.set_title(f'#{i+1}  (score: {scores_ft[idx]:.3f})',
                        fontsize=9, color=GREEN, fontweight='bold')
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(GREEN)
                spine.set_linewidth(2)

        plt.savefig(OUTDIR / f"multimodal_example_{qi+1}.png")
        plt.close()
        print(f"  Saved multimodal_example_{qi+1}.png")

    del model_base, model_ft, img_emb_base, img_emb_ft
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory freed.")


# ═══════════════════════════════════════════════════════════
# COMPOSED: (Image + Text) → Image (FashionIQ)
# ═══════════════════════════════════════════════════════════

def generate_composed_examples():
    """Generate before/after composed retrieval examples."""
    print("\n" + "=" * 60)
    print("COMPOSED EXAMPLES (FashionIQ)")
    print("=" * 60)

    MODEL = "Salesforce/blip2-itm-vit-g"
    CATEGORY = "dress"
    DATA_DIR = "./data/fashioniq"
    CACHE_DIR = "./data/fashioniq/image_cache"
    URL_MAP_BASE = (
        "https://raw.githubusercontent.com/"
        "hongwang600/fashion-iq-metadata/master/image_url"
    )

    # Load data
    data_path = Path(DATA_DIR)
    with open(data_path / "captions" / f"cap.{CATEGORY}.val.json") as f:
        annotations = json.load(f)
    with open(data_path / "image_splits" / f"split.{CATEGORY}.val.json") as f:
        gallery_ids = json.load(f)

    # Load URL mapping
    cache_file = data_path / "image_url" / f"asin2url.{CATEGORY}.txt"
    url_mapping = {}
    with open(cache_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                url_mapping[parts[0].strip()] = parts[1].strip()

    def load_img(image_id):
        for ext in [".jpg", ".jpeg", ".png"]:
            local = Path(CACHE_DIR) / f"{image_id}{ext}"
            if local.exists():
                try:
                    return Image.open(local).convert("RGB")
                except Exception:
                    pass
        url = url_mapping.get(image_id)
        if url:
            try:
                return load_image(url, cache_dir=CACHE_DIR)
            except Exception:
                return None
        return None

    # Load gallery images
    print("Loading gallery images...")
    gallery_imgs = []
    valid_gallery_ids = []
    for gid in gallery_ids[:300]:  # Use first 300 for speed + memory
        img = load_img(gid)
        if img is not None:
            gallery_imgs.append(img)
            valid_gallery_ids.append(gid)
    print(f"Loaded {len(valid_gallery_ids)} gallery images")

    import gc

    # Load baseline model, encode, then free it
    print("Loading baseline BLIP-2...")
    model_base = khoji.JointEmbeddingModel(MODEL)
    print("Encoding gallery with baseline...")
    gallery_emb_base = model_base.encode(images=gallery_imgs, batch_size=16)

    # Pre-encode queries with baseline before freeing
    # (we'll collect them in the loop below, so store model ref)
    base_model_ref = model_base

    # Now load fine-tuned model — but first free baseline to make room
    # We need baseline for query encoding too, so do it sequentially per example
    # Strategy: encode all gallery + queries with baseline, free it, then do fine-tuned

    # Collect annotations we'll use
    examples = []
    for ann in annotations:
        if len(examples) >= 3:
            break
        cand_img = load_img(ann["candidate"])
        target_img = load_img(ann["target"])
        if cand_img is None or target_img is None:
            continue
        if ann["target"] not in valid_gallery_ids:
            continue
        caption = ann["captions"][0]
        if len(caption) < 10 or len(caption) > 80:
            continue
        examples.append((ann["candidate"], caption, ann["target"], cand_img, target_img))

    # Encode all baseline queries
    base_query_embs = []
    for cand_id, caption, target_id, cand_img, target_img in examples:
        q = model_base.encode(images=[cand_img], texts=[caption], show_progress=False)
        base_query_embs.append(q)

    del model_base, base_model_ref
    gc.collect()
    torch.cuda.empty_cache()

    # Now load fine-tuned model
    print("Loading fine-tuned BLIP-2...")
    model_ft = khoji.JointEmbeddingModel(
        MODEL, adapter_path="output/composed-retrieval/dress/adapter"
    )
    print("Encoding gallery with fine-tuned...")
    gallery_emb_ft = model_ft.encode(images=gallery_imgs, batch_size=16)

    for ei, (cand_id, caption, target_id, cand_img, target_img) in enumerate(examples):
        print(f"\nExample {ei+1}: '{caption}'")

        # Use pre-computed baseline query embedding
        q_base = base_query_embs[ei]
        q_ft = model_ft.encode(images=[cand_img], texts=[caption], show_progress=False)

        # Rank
        scores_base = torch.mm(q_base, gallery_emb_base.t()).squeeze(0)
        scores_ft = torch.mm(q_ft, gallery_emb_ft.t()).squeeze(0)

        top5_base = torch.topk(scores_base, 5).indices.tolist()
        top5_ft = torch.topk(scores_ft, 5).indices.tolist()

        # Check target rank
        target_idx = valid_gallery_ids.index(target_id) if target_id in valid_gallery_ids else -1
        rank_base = (scores_base.argsort(descending=True) == target_idx).nonzero().item() + 1 if target_idx >= 0 else "N/A"
        rank_ft = (scores_ft.argsort(descending=True) == target_idx).nonzero().item() + 1 if target_idx >= 0 else "N/A"

        # Create figure
        fig = plt.figure(figsize=(18, 8.5))
        gs = gridspec.GridSpec(3, 7, figure=fig, hspace=0.3, wspace=0.12,
                               height_ratios=[0.05, 1, 1],
                               width_ratios=[1.2, 0.1, 1, 1, 1, 1, 1])

        # Query section
        ax_ref = fig.add_subplot(gs[1:, 0])
        ax_ref.imshow(cand_img)
        ax_ref.set_title('Reference Image', fontsize=11, fontweight='bold')
        ax_ref.axis('off')
        for spine in ax_ref.spines.values():
            spine.set_visible(True)
            spine.set_color(ORANGE)
            spine.set_linewidth(3)

        # Caption below reference
        ax_ref.text(0.5, -0.08, f'"{caption}"', ha='center', va='top',
                   fontsize=11, style='italic', color=ORANGE, fontweight='bold',
                   transform=ax_ref.transAxes, wrap=True)

        # Plus sign
        ax_plus = fig.add_subplot(gs[1:, 1])
        ax_plus.axis('off')

        # Row titles
        fig.text(0.55, 0.95, 'Top-5 Retrieved Results', ha='center',
                fontsize=13, fontweight='bold')

        # Baseline row
        fig.text(0.22, 0.88, f'Before Fine-Tuning  (target rank: {rank_base})',
                fontsize=10, fontweight='bold', color=GRAY)
        for i, idx in enumerate(top5_base):
            ax = fig.add_subplot(gs[1, i + 2])
            result_img = gallery_imgs[idx]
            ax.imshow(result_img)
            is_target = valid_gallery_ids[idx] == target_id
            border_color = GREEN if is_target else '#D1D5DB'
            title_suffix = ' (TARGET)' if is_target else ''
            ax.set_title(f'#{i+1}{title_suffix}', fontsize=9,
                        color=GREEN if is_target else GRAY,
                        fontweight='bold' if is_target else 'normal')
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(border_color)
                spine.set_linewidth(2 if not is_target else 3)

        # Fine-tuned row
        fig.text(0.22, 0.44, f'After Fine-Tuning  (target rank: {rank_ft})',
                fontsize=10, fontweight='bold', color=GREEN)
        for i, idx in enumerate(top5_ft):
            ax = fig.add_subplot(gs[2, i + 2])
            result_img = gallery_imgs[idx]
            ax.imshow(result_img)
            is_target = valid_gallery_ids[idx] == target_id
            border_color = GREEN if is_target else '#D1D5DB'
            title_suffix = ' (TARGET)' if is_target else ''
            ax.set_title(f'#{i+1}{title_suffix}', fontsize=9,
                        color=GREEN if is_target else GRAY,
                        fontweight='bold' if is_target else 'normal')
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(border_color)
                spine.set_linewidth(2 if not is_target else 3)

        plt.savefig(OUTDIR / f"composed_example_{ei+1}.png")
        plt.close()
        print(f"  Target rank: {rank_base} → {rank_ft}")
        print(f"  Saved composed_example_{ei+1}.png")

    del model_base, model_ft
    torch.cuda.empty_cache()


if __name__ == "__main__":
    generate_multimodal_examples()
    generate_composed_examples()
    print("\nAll examples generated!")
