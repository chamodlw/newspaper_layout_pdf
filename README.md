# Newspaper PDF Pipeline
## Sinhala Newspaper Article Separator + Surya OCR Extractor

Mirrors the image-based pipeline exactly — same folder structure, same
article crops, same Surya OCR — but the input is a **PDF** instead of a
scanned image.

---

## Folder structure

```
newspaper_pdf_pipeline/
│
├── newspaper_pdfs/                   ← PUT YOUR PDF FILES HERE
│   ├── lankadeepa_2024_05_10.pdf
│   ├── dinamina_2024_05_11.pdf
│   └── divaina_2024_05_12.pdf
│
├── rule_sets/                        ← per-newspaper rule sets
│   ├── __init__.py                   (registry + auto-detect)
│   ├── article_rules_lankadeepa.py
│   ├── article_rules_dinamina.py
│   └── article_rules_divaina.py
│
├── pdf_separator.py                  ← STEP 1 – PDF → article crops
├── pdf_news_extract.py               ← STEP 2 – article crops → text
├── pdf_to_images.py                  ← helper (render PDF pages only)
│
├── newspaper_pages/                  ← intermediate: rendered page PNGs
│   └── lankadeepa_2024_05_10/
│       ├── page_01.png
│       └── page_02.png
│
├── separated_articles/               ← article crop images (lossless PNG)
│   └── lankadeepa_2024_05_10/
│       ├── page_01/
│       │   ├── article_01.png
│       │   └── article_02.png
│       └── page_02/
│           └── article_01.png
│
├── separated_layout/                 ← visualisation with bounding boxes
│   └── lankadeepa_2024_05_10/
│       ├── page_01_layout.jpg
│       └── page_02_layout.jpg
│
└── extract_news/                     ← OCR text output
    └── lankadeepa_2024_05_10/
        ├── page_01_extracted.txt
        ├── page_02_extracted.txt
        └── all_pages_extracted.txt   ← combined whole-newspaper text
```

---

## Installation

```bash
pip install pymupdf opencv-python numpy pillow surya-ocr
```

---

## Usage

### Step 1 — Separate articles

```bash
python pdf_separator.py
```

- Scans `newspaper_pdfs/` for PDFs.
- Lets you pick which PDFs to process (or press Enter for all).
- **Auto-detects** the rule set from the filename
  (e.g. `lankadeepa_*.pdf` → Lankadeepa rules).
- If the newspaper can't be identified from the filename, asks you to
  select a rule set from the menu.
- Renders each page at 200 DPI → saves to `newspaper_pages/`.
- Runs contour-based article detection on each page.
- Saves article crops (lossless PNG) to `separated_articles/`.
- Saves bounding-box visualisations to `separated_layout/`.

### Step 2 — Extract text

```bash
python pdf_news_extract.py
```

- Automatically reads all article crops from `separated_articles/`.
- Runs Surya OCR on each crop.
- Writes per-page text files and a combined `all_pages_extracted.txt`
  to `extract_news/`.

---

## Adding a new newspaper

1. Create `rule_sets/article_rules_<name>.py` following the same
   key structure as the existing files.
2. Add it to the registry in `rule_sets/__init__.py`:
   ```python
   from rule_sets.article_rules_<name> import RULES as _NAME
   RULE_SETS[4]      = _NAME
   RULE_SET_LABELS[4] = "Newspaper Name  –  short description"
   _KEYWORD_MAP["keyword"] = 4
   ```

---

## Tuning tips

| Problem                        | Try                                         |
|-------------------------------|---------------------------------------------|
| Too many tiny fragments        | Increase `min_area`, `min_width`, `min_height` |
| Articles merged together       | Decrease `morph_iterations` or `morph_kernel_size` |
| Missing articles near edges    | Increase `max_width_ratio` / `max_height_ratio` |
| Poor OCR on small text         | Increase `DEFAULT_DPI` in `pdf_separator.py` (try 250–300) |
| Coloured backgrounds not split | Lower `white_threshold` (try 200–220) |
| Duplicate boxes                | Lower `overlap_threshold` (try 0.25–0.35) |