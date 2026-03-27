"""Generate synthetic SKU matching dataset using GPT-Image-1.

Creates product images for 5 brands × 8 families × 5 variants = 200 SKUs.
Rate-limited to 16 images/minute per Azure OpenAI limits.

Usage:
    python data/sku-matching/generate_dataset.py
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import requests
from PIL import Image
from io import BytesIO

# ── Azure OpenAI config ──────────────────────────────────

ENDPOINT = os.environ.get(
    "AZURE_OPENAI_ENDPOINT",
    "https://your-resource.cognitiveservices.azure.com",
)
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-image-1")
API_VERSION = "2025-04-01-preview"

IMAGE_DIR = Path(__file__).parent / "images"
METADATA_DIR = Path(__file__).parent / "metadata"

# ── Product definitions ──────────────────────────────────

BRANDS = ["Sainsburys", "Tesco", "Aldi", "Lidl", "Waitrose"]

BRAND_STYLES = {
    "Sainsburys": "Sainsbury's own-brand orange packaging with clean modern design",
    "Tesco": "Tesco own-brand blue and white striped packaging with simple layout",
    "Aldi": "Aldi budget-style simple white and red packaging with minimal design",
    "Lidl": "Lidl own-brand blue and yellow packaging with bold text",
    "Waitrose": "Waitrose premium green and cream elegant packaging with refined typography",
}

BRAND_PRICE_MULTIPLIER = {
    "Sainsburys": 1.0,
    "Tesco": 0.98,
    "Aldi": 0.75,
    "Lidl": 0.77,
    "Waitrose": 1.25,
}

# (family, variant, base_size, base_price_gbp, attributes)
PRODUCTS = [
    # Porridge Oats
    ("Porridge Oats", "Regular", "1kg", 1.80, "wholegrain, high fibre"),
    ("Porridge Oats", "Organic", "1kg", 2.60, "organic, USDA certified, wholegrain"),
    ("Porridge Oats", "Regular Small", "500g", 1.10, "wholegrain, high fibre"),
    ("Porridge Oats", "Gluten Free", "1kg", 3.20, "gluten-free, certified, wholegrain"),
    ("Porridge Oats", "Chocolate", "1kg", 2.40, "chocolate flavour, wholegrain"),
    # Milk
    ("Milk", "Whole", "2L", 1.45, "whole milk, 3.6% fat, pasteurised"),
    ("Milk", "Semi-Skimmed", "2L", 1.45, "semi-skimmed, 1.8% fat, pasteurised"),
    ("Milk", "Organic Whole", "2L", 1.95, "organic, whole milk, free-range, pasteurised"),
    ("Milk", "Skimmed", "1L", 0.85, "skimmed, 0.1% fat, pasteurised"),
    ("Milk", "Lactose Free", "2L", 1.80, "lactose-free, semi-skimmed, filtered"),
    # Bread
    ("Bread", "White Sliced", "800g", 1.10, "white, medium sliced, soft"),
    ("Bread", "Wholemeal Sliced", "800g", 1.20, "wholemeal, medium sliced, high fibre"),
    ("Bread", "Sourdough Loaf", "400g", 2.20, "sourdough, artisan, naturally leavened"),
    ("Bread", "Seeded Loaf", "800g", 1.50, "seeded, multiseed, sliced"),
    ("Bread", "Gluten Free White", "550g", 2.80, "gluten-free, white, sliced"),
    # Yoghurt
    ("Yoghurt", "Natural", "500g", 0.90, "natural, plain, live cultures"),
    ("Yoghurt", "Greek Style", "500g", 1.40, "Greek style, thick, high protein"),
    ("Yoghurt", "Strawberry", "500g", 1.10, "strawberry, fruit pieces, creamy"),
    ("Yoghurt", "Vanilla", "150g", 0.60, "vanilla, single pot, creamy"),
    ("Yoghurt", "Organic Natural", "500g", 1.50, "organic, natural, live cultures"),
    # Pasta
    ("Pasta", "Spaghetti", "500g", 0.75, "spaghetti, durum wheat, Italian style"),
    ("Pasta", "Penne", "500g", 0.75, "penne, durum wheat, Italian style"),
    ("Pasta", "Organic Penne", "500g", 1.30, "organic, penne, bronze die cut"),
    ("Pasta", "Gluten Free Fusilli", "500g", 1.60, "gluten-free, fusilli, rice and maize"),
    ("Pasta", "Wholewheat Spaghetti", "500g", 0.95, "wholewheat, spaghetti, high fibre"),
    # Orange Juice
    ("Orange Juice", "Smooth", "1L", 1.50, "smooth, not from concentrate, chilled"),
    ("Orange Juice", "With Bits", "1L", 1.50, "with bits, not from concentrate, chilled"),
    ("Orange Juice", "Organic", "1L", 2.20, "organic, smooth, not from concentrate"),
    ("Orange Juice", "From Concentrate", "1L", 0.85, "from concentrate, ambient, added vitamin C"),
    ("Orange Juice", "Freshly Squeezed", "1L", 2.80, "freshly squeezed, chilled, 3-day shelf life"),
    # Crisps
    ("Crisps", "Ready Salted", "150g", 1.25, "ready salted, classic, potato crisps"),
    ("Crisps", "Cheese and Onion", "150g", 1.25, "cheese and onion, flavoured, potato crisps"),
    ("Crisps", "Salt and Vinegar", "150g", 1.25, "salt and vinegar, tangy, potato crisps"),
    ("Crisps", "Lightly Salted Organic", "100g", 1.80, "organic, lightly salted, hand cooked"),
    ("Crisps", "Sharing Bag Ready Salted", "300g", 2.00, "ready salted, sharing bag, party size"),
    # Peanut Butter
    ("Peanut Butter", "Smooth", "340g", 1.90, "smooth, no palm oil, high protein"),
    ("Peanut Butter", "Crunchy", "340g", 1.90, "crunchy, no palm oil, high protein"),
    ("Peanut Butter", "Organic Smooth", "340g", 3.20, "organic, smooth, single origin"),
    ("Peanut Butter", "No Added Sugar", "340g", 2.40, "no added sugar, smooth, 100% peanuts"),
    ("Peanut Butter", "Chocolate", "340g", 2.60, "chocolate, smooth, cocoa blended"),
]


def make_sku_id(brand: str, family: str, variant: str) -> str:
    """Create a filesystem-safe SKU ID."""
    return f"{brand}_{family}_{variant}".replace(" ", "-").replace("'", "").lower()


def make_description(brand: str, family: str, variant: str, size: str,
                     base_price: float, attributes: str) -> str:
    """Create a structured product description."""
    price = round(base_price * BRAND_PRICE_MULTIPLIER[brand], 2)
    brand_display = "Sainsbury's" if brand == "Sainsburys" else brand
    return f"{brand_display} {family} {variant}, {size}, £{price:.2f}, {attributes}"


def make_image_prompt(brand: str, family: str, variant: str, size: str) -> str:
    """Create an image generation prompt."""
    brand_display = "Sainsbury's" if brand == "Sainsburys" else brand
    style = BRAND_STYLES[brand]
    return (
        f"A product photograph of a {brand_display} supermarket own-brand "
        f"{family.lower()} product, {variant.lower()} variant, {size} size. "
        f"The packaging is in {style}. "
        f"Single product centered on a plain white background, "
        f"studio product photography, clean and professional, "
        f"showing the front of the package clearly with product name visible. "
        f"No hands, no other objects, no text overlays."
    )


def _get_token():
    """Get Azure AD bearer token for Cognitive Services."""
    from azure.identity import DefaultAzureCredential
    cred = DefaultAzureCredential()
    token = cred.get_token("https://cognitiveservices.azure.com/.default")
    return token.token


def generate_image(prompt: str, save_path: Path) -> bool:
    """Generate an image using Azure OpenAI GPT-Image-1."""
    url = f"{ENDPOINT}/openai/deployments/{DEPLOYMENT}/images/generations?api-version={API_VERSION}"

    token = _get_token()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    body = {
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "quality": "low",
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        # GPT-Image-1 returns base64
        if "data" in data and len(data["data"]) > 0:
            item = data["data"][0]
            if "b64_json" in item:
                img_bytes = base64.b64decode(item["b64_json"])
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                img = img.resize((512, 512), Image.LANCZOS)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(save_path, "JPEG", quality=90)
                return True
            elif "url" in item:
                img_resp = requests.get(item["url"], timeout=60)
                img_resp.raise_for_status()
                img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                img = img.resize((512, 512), Image.LANCZOS)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(save_path, "JPEG", quality=90)
                return True

        print(f"  Unexpected response format: {list(data.keys())}")
        return False

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Build full product catalog
    catalog = []
    for brand in BRANDS:
        for family, variant, size, base_price, attributes in PRODUCTS:
            sku_id = make_sku_id(brand, family, variant)
            description = make_description(brand, family, variant, size, base_price, attributes)
            prompt = make_image_prompt(brand, family, variant, size)
            img_path = IMAGE_DIR / brand.lower() / f"{sku_id}.jpg"

            catalog.append({
                "sku_id": sku_id,
                "brand": brand,
                "family": family,
                "variant": variant,
                "size": size,
                "base_price": base_price,
                "price": round(base_price * BRAND_PRICE_MULTIPLIER[brand], 2),
                "attributes": attributes,
                "description": description,
                "image_path": str(img_path.relative_to(Path(__file__).parent)),
                "prompt": prompt,
            })

    # Save catalog metadata
    with open(METADATA_DIR / "catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"Catalog: {len(catalog)} SKUs ({len(BRANDS)} brands × {len(PRODUCTS)} products)")

    # Generate images (rate-limited: 16/min)
    to_generate = [item for item in catalog
                   if not (Path(__file__).parent / item["image_path"]).exists()]
    print(f"Images to generate: {len(to_generate)} (skipping {len(catalog) - len(to_generate)} existing)")

    generated = 0
    failed = 0
    batch_start = time.time()
    batch_count = 0

    for i, item in enumerate(to_generate):
        # Rate limiting: 16 per minute
        if batch_count >= 15:
            elapsed = time.time() - batch_start
            if elapsed < 62:
                wait = 62 - elapsed
                print(f"  Rate limit: waiting {wait:.0f}s...")
                time.sleep(wait)
            batch_start = time.time()
            batch_count = 0

        img_path = Path(__file__).parent / item["image_path"]
        print(f"[{i+1}/{len(to_generate)}] {item['sku_id']}...", end=" ", flush=True)

        ok = generate_image(item["prompt"], img_path)
        if ok:
            generated += 1
            print("OK")
        else:
            failed += 1
            print("FAILED")

        batch_count += 1

    print(f"\nDone: {generated} generated, {failed} failed, {len(catalog) - len(to_generate)} skipped")

    # Verify all images exist
    missing = [item for item in catalog
               if not (Path(__file__).parent / item["image_path"]).exists()]
    if missing:
        print(f"\nWARNING: {len(missing)} images missing:")
        for item in missing[:10]:
            print(f"  {item['sku_id']}")
    else:
        print(f"\nAll {len(catalog)} images present.")


if __name__ == "__main__":
    main()
