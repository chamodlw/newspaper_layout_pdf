"""
rule_sets/article_rules_divaina.py
====================================
Detection + OCR rules for Divaina – popular Sinhala tabloid/broadsheet.

Layout characteristics:
  • Busier layout with more photos and graphical headlines
  • Coloured section headers (red/blue bars common)
  • Dense ad sections, especially bottom half of pages
  • Some pages use tinted background panels behind article blocks
"""

RULES = {
    "newspaper_name": "Divaina",

    # ── Thresholding ─────────────────────────────────────────────────────────
    # Lower threshold to capture content on slightly tinted backgrounds.
    "white_threshold": 220,

    # ── Morphological cleanup ─────────────────────────────────────────────────
    "morph_kernel_size": (2, 2),
    "morph_iterations":  3,        # more iterations – connects through colour panels

    # ── Area filters ──────────────────────────────────────────────────────────
    "min_area":       2000,        # smaller min – picks up photo captions too
    "max_area_ratio": 0.50,

    # ── Dimension filters ─────────────────────────────────────────────────────
    "min_width":       45,
    "min_height":      45,
    "max_width_ratio":  0.96,
    "max_height_ratio": 0.96,

    # ── Aspect-ratio guard ────────────────────────────────────────────────────
    "min_aspect_ratio": 0.05,
    "max_aspect_ratio": 22.0,

    # ── Content-density filter ────────────────────────────────────────────────
    # Slightly lower density – coloured backgrounds count as non-white content.
    "min_density": 0.015,

    # ── Overlap removal ───────────────────────────────────────────────────────
    "overlap_threshold": 0.40,

    # ── Crop margin ───────────────────────────────────────────────────────────
    "save_margin": 5,

    # ── OCR settings ──────────────────────────────────────────────────────────
    "ocr_lang":       "si",
    "ocr_extra_lang": "en",
    "ocr_min_chars":  10,
}