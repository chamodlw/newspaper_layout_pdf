"""
rule_sets/article_rules_dinamina.py
=====================================
Detection + OCR rules for Dinamina – government / state broadsheet.

Layout characteristics:
  • Slightly wider column gutters than Lankadeepa
  • More whitespace around headlines; larger headline fonts
  • Occasional full-width banner articles at top of page
  • Moderate ad density; ads tend to be boxed with visible border lines
"""

RULES = {
    "newspaper_name": "Dinamina",

    # ── Thresholding ─────────────────────────────────────────────────────────
    "white_threshold": 230,        # slightly lower – gutters can be off-white

    # ── Morphological cleanup ─────────────────────────────────────────────────
    "morph_kernel_size": (3, 3),   # slightly larger kernel – wider gutters
    "morph_iterations":  2,

    # ── Area filters ──────────────────────────────────────────────────────────
    "min_area":       3000,        # bigger minimum – fewer tiny fragments
    "max_area_ratio": 0.50,        # allow wider banner articles

    # ── Dimension filters ─────────────────────────────────────────────────────
    "min_width":       60,
    "min_height":      50,
    "max_width_ratio":  0.97,      # full-width banners near page edge
    "max_height_ratio": 0.95,

    # ── Aspect-ratio guard ────────────────────────────────────────────────────
    "min_aspect_ratio": 0.04,
    "max_aspect_ratio": 25.0,      # wider banners have higher aspect ratio

    # ── Content-density filter ────────────────────────────────────────────────
    "min_density": 0.02,

    # ── Overlap removal ───────────────────────────────────────────────────────
    "overlap_threshold": 0.35,

    # ── Crop margin ───────────────────────────────────────────────────────────
    "save_margin": 6,

    # ── OCR settings ──────────────────────────────────────────────────────────
    "ocr_lang":       "si",
    "ocr_extra_lang": "en",
    "ocr_min_chars":  10,
}