"""
rule_sets/__init__.py
=====================
Central registry for per-newspaper rule sets.

To add a new newspaper:
  1. Create  rule_sets/article_rules_<name>.py  with a RULES dict.
  2. Import it here and add an entry to RULE_SETS + RULE_SET_LABELS.
  3. Add the Sinhala name and any filename keywords to _KEYWORD_MAP.

Keys that every RULES dict must contain
  (see any existing rule file for full documentation of each key):

  newspaper_name, white_threshold, morph_kernel_size, morph_iterations,
  min_area, max_area_ratio, min_width, min_height, max_width_ratio,
  max_height_ratio, min_aspect_ratio, max_aspect_ratio, min_density,
  overlap_threshold, save_margin,
  ocr_lang, ocr_extra_lang, ocr_min_chars

Supported newspapers
---------------------
  1  අරුණ      Aruna
  2  ලංකාදීප   Lankadeepa
  3  දිනමිණ    Dinamina
  4  දිවයින    Divaina
  5  මව්බිම    Mawbima
"""

from rule_sets.article_rules_aruna       import RULES as _ARUNA
from rule_sets.article_rules_lankadeepa  import RULES as _LANKADEEPA
from rule_sets.article_rules_dinamina    import RULES as _DINAMINA
from rule_sets.article_rules_divaina     import RULES as _DIVAINA
from rule_sets.article_rules_mawbima     import RULES as _MAWBIMA

# ── Integer key → rule dict ──────────────────────────────────────────────────
RULE_SETS: dict[int, dict] = {
    1: _ARUNA,
    2: _LANKADEEPA,
    3: _DINAMINA,
    4: _DIVAINA,
    5: _MAWBIMA,
}

# ── Human-readable labels (shown in the selection menu) ──────────────────────
RULE_SET_LABELS: dict[int, str] = {
    1: "අරුණ      (Aruna)     ",
    2: "ලංකාදීප  (Lankadeepa)  ",
    3: "දිනමිණ   (Dinamina)    ",
    4: "දිවයින   (Divaina)     ",
    5: "මව්බිම   (Mawbima)     ",
}

# ── Auto-detect from PDF filename ────────────────────────────────────────────
# Add both romanised spellings and Sinhala Unicode keywords so either works.
_KEYWORD_MAP: dict[str, int] = {
    # Aruna – අරුණ
    "aruna":       1,
    "අරුණ":       1,

    # Lankadeepa – ලංකාදීප
    "lankadeepa":  2,
    "ලංකාදීප":   2,

    # Dinamina – දිනමිණ
    "dinamina":    3,
    "dinaminа":    3,   # common alternate romanisation
    "දිනමිණ":    3,

    # Divaina – දිවයින
    "divaina":     4,
    "diwaina":     4,
    "දිවයින":    4,

    # Mawbima – මව්බිම
    "mawbima":     5,
    "maubima":     5,
    "මව්බිම":    5,
}


def detect_rule_set(pdf_name: str) -> dict | None:
    """
    Try to infer the rule set from the PDF filename.
    Returns the matching RULES dict, or None if no match found.

    Checks both romanised and Sinhala Unicode keywords so filenames
    in either script are recognised automatically.

    Examples:
        detect_rule_set("lankadeepa_2024_05_10.pdf")  →  LANKADEEPA rules
        detect_rule_set("මව්බිම_2024_05_10.pdf")     →  MAWBIMA rules
        detect_rule_set("2024-11-11.pdf")              →  None  (ask user)
    """
    lower = pdf_name.lower()
    for keyword, key in _KEYWORD_MAP.items():
        if keyword.lower() in lower:
            return RULE_SETS[key]
    return None