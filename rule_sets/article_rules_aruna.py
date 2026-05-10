"""
rule_sets/article_rules_aruna.py
==================================
Detection + OCR rules for Aruna (අරුණ) – Sinhala daily newspaper.

Layout characteristics:
  • Moderate column density, clean gutters
  • Strong headline hierarchy with larger fonts
  • Mix of photos and text articles
  • Relatively clean white backgrounds between articles
"""

RULES = {
    "newspaper_name": "Aruna",

    # ── Thresholding ─────────────────────────────────────────────────────────
    "white_threshold": 232,

    # ── Morphological cleanup ─────────────────────────────────────────────────
    "morph_kernel_size": (2, 2),
    "morph_iterations":  2,

    # ── Area filters ──────────────────────────────────────────────────────────
    "min_area":       2500,
    "max_area_ratio": 0.48,

    # ── Dimension filters ─────────────────────────────────────────────────────
    "min_width":       50,
    "min_height":      50,
    "max_width_ratio":  0.95,
    "max_height_ratio": 0.95,

    # ── Aspect-ratio guard ────────────────────────────────────────────────────
    "min_aspect_ratio": 0.05,
    "max_aspect_ratio": 20.0,

    # ── Content-density filter ────────────────────────────────────────────────
    "min_density": 0.02,

    # ── Overlap removal ───────────────────────────────────────────────────────
    "overlap_threshold": 0.40,

    # ── Crop margin ───────────────────────────────────────────────────────────
    "save_margin": 5,

    # ── OCR settings ──────────────────────────────────────────────────────────
    "ocr_lang":       "si",
    "ocr_extra_lang": "en",
    "ocr_min_chars":  10,
}