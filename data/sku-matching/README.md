# SKU Matching Dataset

AI-generated grocery product images for cross-brand SKU matching using mixed-mode (img+txt → img+txt) retrieval.

## Overview

**Task:** Given a product from one supermarket brand (photo + description), find the equivalent product in another brand's catalog.

**Why mixed-mode?** Neither image nor text alone is sufficient:
- **Image alone fails:** Same product across brands has completely different packaging
- **Text alone fails:** Brand names differ, descriptions use different wording
- **Image + text together:** The model learns that a blue-striped milk carton labeled "Whole Milk 2L" matches an orange carton labeled "Whole Milk 2L" — combining visual product cues with textual attribute matching

## Dataset Structure

```
data/sku-matching/
  images/
    sainsburys/    # 40 product images (orange brand packaging)
    tesco/         # 40 product images (blue/white striped packaging)
    aldi/          # 40 product images (red/white budget packaging)
    lidl/          # 40 product images (blue/yellow bold packaging)
    waitrose/      # 40 product images (green/cream premium packaging)
  metadata/
    catalog.json   # Full catalog: 200 SKUs with descriptions, prices, attributes
  generate_dataset.py  # Script to regenerate images via Azure OpenAI GPT-Image-1
  README.md
```

## Products

**5 brands × 8 families × 5 variants = 200 SKUs**

| Family | Variants |
|--------|----------|
| Porridge Oats | Regular 1kg, Organic 1kg, Regular 500g, Gluten-Free 1kg, Chocolate 1kg |
| Milk | Whole 2L, Semi-Skimmed 2L, Organic Whole 2L, Skimmed 1L, Lactose-Free 2L |
| Bread | White Sliced, Wholemeal Sliced, Sourdough Loaf, Seeded Loaf, GF White |
| Yoghurt | Natural 500g, Greek Style 500g, Strawberry 500g, Vanilla 150g, Organic Natural 500g |
| Pasta | Spaghetti 500g, Penne 500g, Organic Penne 500g, GF Fusilli 500g, Wholewheat Spaghetti 500g |
| Orange Juice | Smooth 1L, With Bits 1L, Organic 1L, From Concentrate 1L, Freshly Squeezed 1L |
| Crisps | Ready Salted 150g, Cheese & Onion 150g, Salt & Vinegar 150g, Organic 100g, Sharing Bag 300g |
| Peanut Butter | Smooth 340g, Crunchy 340g, Organic Smooth 340g, No Added Sugar 340g, Chocolate 340g |

Each product has a structured text description:
```
"Sainsbury's Porridge Oats Regular, 1kg, £1.80, wholegrain, high fibre"
"Tesco Porridge Oats Regular, 1kg, £1.76, wholegrain, high fibre"
```

## Train/Test Split

One model is trained and evaluated on **two dimensions** of generalization:

**Brand generalization:** Waitrose is held out entirely — the model never sees any Waitrose product image during training. At test time, 30 Waitrose products query against the 4-brand corpus (160 items).

**Product generalization:** Crisps and Peanut Butter families are never used as queries or positives during training. At test time, Sainsbury's Crisps/PB products query against a corpus of 150 items (the 4-brand corpus minus Sainsbury's own Crisps/PB, to prevent self-matching).

```
                    Train families               Held-out families
                    (Oats, Milk, Bread,          (Crisps, Peanut Butter)
                     Yoghurt, Pasta, OJ)
                    ──────────────────────       ─────────────────────
Train brands        ██ TRAINING ██               In corpus as negatives,
(Sainsbury's,       120 queries, 160 corpus      never as queries or
 Tesco, Aldi,                                    positives
 Lidl)

Held-out brand      █ TEST: Brand █
(Waitrose)          30 queries → 160 corpus
                    (unseen packaging)

Sainsbury's only    (trained on other families)   █ TEST: Product █
                                                  10 queries → 150 corpus
                                                  (unseen categories)
```

**Corpus** = all 160 products from the 4 train brands (all families). This represents "our catalog."

**Note on product test:** Crisps and Peanut Butter items ARE in the training corpus as potential negatives — the model may see their images/text during training when they are sampled as negatives for other queries. However, the model never learns Crisps↔Crisps or PB↔PB cross-brand matching (no such query-positive pairs exist in training). This mirrors the real-world scenario where your catalog contains all products but you haven't annotated cross-brand matches for every category yet.

## Regenerating Images

The images were generated using Azure OpenAI GPT-Image-1. To regenerate:

1. Set up Azure OpenAI with a `gpt-image-1` deployment
2. Update the credentials in `generate_dataset.py`
3. Run: `python data/sku-matching/generate_dataset.py`

Rate limited to ~15 images/minute. Full generation takes ~35 minutes.

## Results

See the [training script](../../scripts/train_sku_matching.py) for the full experiment, or the [report](../../output/sku-matching/report.html) for visual before/after comparisons.

| Eval Dimension | Metric | Baseline | Fine-tuned | Delta |
|---|---|---|---|---|
| **Brand** (Waitrose) | nDCG@1 | 0.7667 | **0.9333** | +0.1666 |
| **Brand** (Waitrose) | Recall@5 | 0.7750 | **0.9667** | +0.1917 |
| **Brand** (Waitrose) | Recall@10 | 0.9250 | **1.0000** | +0.0750 |
| **Product** (Crisps, PB) | nDCG@1 | 1.0000 | **1.0000** | +0.0000 |
| **Product** (Crisps, PB) | Recall@3 | 0.8333 | **1.0000** | +0.1667 |
| **Product** (Crisps, PB) | Recall@5 | 0.9667 | **1.0000** | +0.0333 |
