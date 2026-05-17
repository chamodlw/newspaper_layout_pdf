"""
rule_sets/article_rules_lankadeepa.py
======================================
Detection + OCR rules for Lankadeepa – dense Sinhala broadsheet.

Layout characteristics:
  • 6-column broadsheet, heavy Sinhala text density
  • White gutters (>=235 brightness) separate columns / articles
  • Mix of headlines, body text, photos, and small ads
  • Occasional coloured banners (dark backgrounds with light text)
  • Page 1: Front page - large headline blocks, feature photos, right-side briefs
  • Page 2: Classified ads page (ලුහුඩු දැන්වීම්) - ENTIRE page excluded
  • Page 3+: Standard inner news pages

CALIBRATION HISTORY
-------------------
v1 (initial):
  - min_area=2500, min_width=50, min_height=50
  - Page 1 excluded masthead (Zone B: y=0-0.10) and bottom ad (Zone A: y=0.82-1.0)
  - Page 2 excluded y=0.20-1.0

v2 (green-border fix):
  - Removed Zone B from page 1 (masthead IS a capture target)
  - Moved Zone A bottom ad start from y=0.82 to y=0.975
  - Expanded page 2 exclusion to full page (0.0-1.0)
  Result: 623 articles detected (should be <100) - still massively over-segmenting

v3 (layout-image fix - current):
  ROOT CAUSE: min_area=2500 is ~0.006% of the rendered page area, allowing
  individual words, single text lines, and gutter fragments to each become
  "articles". Layout visualisations confirmed hundreds of tiny coloured
  boxes per page.

  Fixes applied:
  1. BASE min_area:         2500 -> 50000   (filter word/line fragments)
  2. BASE min_width:          50 -> 150     (minimum meaningful column width)
  3. BASE min_height:         50 -> 120     (minimum meaningful article height)
  4. BASE morph_kernel:     (2,2) -> (3,3)  (better gap-bridging within articles)
  5. BASE morph_iter:          2 -> 3       (merge text blocks into one blob)
  6. BASE overlap_thresh:    0.4 -> 0.30   (stricter - avoid wrong merges)
  7. PAGE 1 min_area:       8000 -> 15000
  8. PAGE 1 min_width:        80 -> 100
  9. PAGE 1 min_height:       60 -> 80
  10. PAGE 1 morph_iter:       3 -> 4
  11. PAGE 3+ exclusion zones added:
       Top  (logo + lottery-results strip): y = 0.000 - 0.115
       Bottom (bank ad + footer):           y = 0.790 - 1.000
  12. PAGE 1 Zone A kept at y=0.975 (S-lon Polycon ad at very bottom)
  13. PAGE 2 full-page exclusion retained (0.0, 0.0, 1.0, 1.0)
"""

# ---------------------------------------------------------------------------
# Shared base  (inner news pages, page 3 onwards)
# ---------------------------------------------------------------------------

_BASE = {
    "newspaper_name": "Lankadeepa",

    # Thresholding
    "white_threshold": 235,

    # Morphological cleanup
    # Increased kernel + iterations to merge individual text glyphs into a
    # coherent article blob, preventing over-segmentation into word fragments.
    "morph_kernel_size": (3, 3),   # was (2, 2)
    "morph_iterations":  3,        # was 2

    # Area filters
    # 2500 px was ~0.006% of a 150-DPI rendered page - catching every word.
    # 50 000 px equates roughly to a 1-column x 3-cm block at 150 DPI, which
    # is the practical minimum for an identifiable news item.
    "min_area":       50_000,      # was 2500
    "max_area_ratio": 0.45,

    # Dimension filters
    # Raised to reject narrow gutters, single text lines, and caption slivers.
    "min_width":       210,
    "min_height":      210,
    "max_width_ratio":  0.95,
    "max_height_ratio": 0.95,

    # Aspect-ratio guard
    "min_aspect_ratio": 0.05,
    "max_aspect_ratio": 20.0,

    # Content-density filter
    "min_density": 0.02,

    # Overlap removal
    # Tightened from 0.4 to 0.30 to reduce accidental merging of adjacent
    # ad or story blocks into a single giant candidate.
    "overlap_threshold": 0.20,     # was 0.4

    # Crop margin
    "save_margin": 5,

    # Gutter reinforcement — zero out thin white vertical/horizontal strips
    # in the pre-dilation content mask so dilation cannot bridge across column
    # gutters and merge adjacent articles.
    #   gutter_dark_fraction : columns/rows with fewer than this fraction of
    #                          dark pixels are treated as a gutter strip.
    #   min_vertical_gutter_px   : minimum width (px) of a vertical gutter to reinforce.
    #   min_horizontal_gutter_px : minimum height (px) of a horizontal gutter to reinforce.
    "reinforce_gutters":         True,
    "gutter_dark_fraction":      0.04,
    "min_vertical_gutter_px":    8,
    "min_horizontal_gutter_px":  10,

    # OCR settings
    "ocr_lang":       "si",
    "ocr_extra_lang": "en",
    "ocr_min_chars":  10,

    # Per-page exclusion zones (overridden per-page below)
    "excluded_zones_by_page": {},
}


# ---------------------------------------------------------------------------
# Page 1: Front page
# ---------------------------------------------------------------------------
#
# Changes from v2:
#   - min_area lowered to 15000 (small front-page briefs must still pass)
#   - min_width/height relaxed vs base (front-page briefs are narrow)
#   - morph_iterations raised to 4 (large display fonts need more gap-bridging)
#   - Zone B (masthead exclusion) remains ABSENT - masthead is a capture target
#   - Zone A (bottom S-lon ad) kept at y=0.975 - only absolute bottom 2.5%

PAGE_1_RULES = {
    **_BASE,
    "newspaper_name": "Lankadeepa - Page 1 (Front)",

    # Front-page brief items are smaller than inner-page articles.
    # 15000 px ~ a single-column headline at 150 DPI.
    "min_area":       15_000,      # was 8000

    # Up to 60% of the page for the dominant headline block
    "max_area_ratio": 0.60,

    # Multi-column headline blobs can be very wide relative to their height
    "max_aspect_ratio": 25.0,

    # Four iterations to bridge the wide spacing inside large display fonts
    # Strongly asymmetric kernel: thin width (3) avoids crossing vertical white 
    # gutters between separate articles, while massive height (45) reaches far 
    # down to grab floating "See Page 5/6" tags.
    "morph_kernel_size": (3, 18),
    "morph_iterations": 3,

    "min_width":  100,             # was 80
    "min_height":  80,             # was 60

    "excluded_zones_by_page": {
        1: [
            # Zone A - S-lon Polycon full-width ad at the very bottom of page.
            # Only the last 2.5% of the page height.
            # The editorial bottom-brief row ends at ~y=0.973, so starting
            # the exclusion at y=0.975 avoids clipping it.
            (0.0, 0.975, 1.0, 0.025),
        ]
    },
}


# ---------------------------------------------------------------------------
# Page 2: Classified ads page (ලුහුඩු දැන්වීම්)
# ---------------------------------------------------------------------------
#
# The entire page is advertising content.
# Full-page exclusion ensures zero articles are captured regardless of what
# the morphological detector finds in the dense text columns.

PAGE_2_RULES = {
    **_BASE,
    "newspaper_name": "Lankadeepa - Page 2 (Classifieds)",

    "excluded_zones_by_page": {
        2: [
            (0.0, 0.0, 1.0, 1.0),   # exclude entire page
        ]
    },
}


# ---------------------------------------------------------------------------
# Page 3+: Standard inner news pages
# ---------------------------------------------------------------------------
#
# Changes from v2:
#   - All base parameter increases apply (see _BASE above).
#   - Two new exclusion zones added:
#       Top    - logo row + lottery / sports-results strip (non-editorial header)
#       Bottom - full-width bank advertisement banner + People's Bank footer
#
# These zones are measured from the page_03_layout.jpg visualisation:
#   Top strip:    y = 0.000 to 0.115  (Lankadeepa logo + lottery numbers row)
#   Bottom strip: y = 0.790 to 1.000  (People's Bank ad banner + QR footer)
#
# NOTE: get_rules_for_page() dynamically injects the actual page number as
# the dict key so the separator can match it correctly.

PAGE_3_PLUS_RULES = {
    **_BASE,
    "newspaper_name": "Lankadeepa - Inner Pages",
    
    # ── Multi-column / Layout Adjustments ──────────────────────────────────────
    # - Taller kernel (y > x) aggressively bridges the large vertical white space 
    #   between top titles and their associated body text/images.
    # - Width expansion bridges the column gutters for multi-column grouping.
    "morph_kernel_size": (8, 12),
    "morph_iterations":  4,

    # Allow larger article blocks since multi-column news spans wide areas 
    "max_area_ratio": 0.65,
    "max_width_ratio": 0.98,

    # excluded_zones_by_page is populated dynamically in get_rules_for_page()
}

# Exclusion zones reused for every inner page (constant fractions)
_INNER_PAGE_EXCLUSIONS = [
    # Top: Lankadeepa logo header + lottery / match-results strip.
    # Visible in layout as two thin horizontal bands from y=0.0 to y~0.112.
    # Using 0.115 to include any single-line overhang.
    (0.0, 0.000, 1.0, 0.115),

    # Bottom: full-width advertisement banner (People's Bank / CMG / etc.)
    # + "Pride of the Nation" footer strip.
    # Editorial content ends at ~y=0.790; the ad + footer fill y=0.790-1.000.
    (0.0, 0.790, 1.0, 0.210),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# RULES is kept for backward-compatibility with callers that import a single
# rule dict.  It carries ``page_rules_factory`` so that process_pdf() can
# dispatch the correct per-page parameters automatically.
RULES = {
    **PAGE_3_PLUS_RULES,
    "page_rules_factory": None,   # filled in below after get_rules_for_page is defined
}


def get_rules_for_page(page_number: int) -> dict:
    """
    Return the appropriate rule dict for the given 1-based page number.

    For inner pages (>=3) the returned dict has ``excluded_zones_by_page``
    populated with the actual page number so the separator can match it.

    Parameters
    ----------
    page_number : int
        1-based page index (1 = front page, 2 = classified ads, etc.)

    Returns
    -------
    dict
        Rule dictionary ready to pass to ``separate_page()``.
    """
    if page_number == 1:
        return PAGE_1_RULES

    if page_number == 2:
        return PAGE_2_RULES

    # Inner pages: clone and inject the correct page-number key so the
    # separator's exclusion-zone lookup finds the right entry.
    rules = dict(PAGE_3_PLUS_RULES)
    rules["excluded_zones_by_page"] = {
        page_number: list(_INNER_PAGE_EXCLUSIONS),
    }
    return rules


# Wire the factory into RULES now that the function is defined.
RULES["page_rules_factory"] = get_rules_for_page