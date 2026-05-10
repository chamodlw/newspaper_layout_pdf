"""
rule_sets/article_rules_mawbima.py
=====================================
Detection + OCR rules for Mawbima (මව්බිම) – Sinhala daily newspaper.

Layout characteristics:
  • Modern layout with bold section colour bars (red/blue accents)
  • Well-defined article borders, clear column separators
  • Heavy use of photos alongside articles
  • Some pages with tinted background panels
  • Slightly more whitespace than Lankadeepa
"""

RULES = {
    "newspaper_name": "Mawbima",

    # ── Thresholding ─────────────────────────────────────────────────────────
    # Lower to capture content under light colour panel backgrounds
    "white_threshold": 225,

    # ── Morphological cleanup ─────────────────────────────────────────────────
    "morph_kernel_size": (2, 2),
    "morph_iterations":  3,

    # ── Area filters ──────────────────────────────────────────────────────────
    "min_area":       2000,
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
    "min_density": 0.015,

    # ── Overlap removal ───────────────────────────────────────────────────────
    "overlap_threshold": 0.40,

    # ── Crop margin ───────────────────────────────────────────────────────────
    "save_margin": 6,

    # ── OCR settings ──────────────────────────────────────────────────────────
    "ocr_lang":       "si",
    "ocr_extra_lang": "en",
    "ocr_min_chars":  10,
}