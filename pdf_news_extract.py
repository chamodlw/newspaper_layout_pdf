#!/usr/bin/env python3
"""
pdf_news_extract.py
===================
OCR extraction pipeline for PDF-separated articles using Surya OCR.

This is the PDF counterpart of news_extract_surya.py.

Reads from:
    separated_articles/
        <pdf_stem>/
            page_001/
                article_01.png
                article_02.png
                ...
            page_002/
                ...

Writes to:
    extract_news/
        <pdf_stem>/
            page_001_extracted.txt
            page_002_extracted.txt
            ...
            all_pages_extracted.txt   ← combined output for the whole PDF

No user input required. Just run it after pdf_separator.py.

Requirements:
    pip install surya-ocr pymupdf opencv-python pillow
"""

import os
import sys

import cv2
import numpy as np
from PIL import Image

# ── Folder paths ──────────────────────────────────────────────────────────────
ARTICLES_BASE = "separated_articles"
EXTRACT_BASE  = "extract_news"

# ── Minimum characters to count as "has text" ─────────────────────────────────
OCR_MIN_CHARS = 10

# ── Supported image extensions ────────────────────────────────────────────────
ARTICLE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ── Surya language codes ──────────────────────────────────────────────────────
OCR_LANGS = ["si", "en"]


# ────────────────────────────────────────────────────────────────────────────
# Load Surya models  (identical to news_extract_surya.py – reuses same logic)
# ────────────────────────────────────────────────────────────────────────────

def load_surya_models():
    """
    Load Surya detection and recognition models.
    Supports both the new API (0.10+) and older API automatically.
    """
    print("  Loading Surya models (first run downloads ~1 GB) ...")

    try:
        from surya.recognition import RecognitionPredictor
        from surya.detection   import DetectionPredictor
        try:
            from surya.foundation import FoundationPredictor
            fp  = FoundationPredictor()
            rec = RecognitionPredictor(fp)
        except ImportError:
            rec = RecognitionPredictor()
        det = DetectionPredictor()
        print("  Surya models loaded (new API).\n")
        return ("new", rec, det)

    except ImportError:
        pass

    try:
        from surya.ocr import run_ocr
        from surya.model.detection.segformer import (
            load_model as load_det_model,
            load_processor as load_det_processor,
        )
        from surya.model.recognition.model     import load_model     as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor

        det_processor, det_model = load_det_processor(), load_det_model()
        rec_model, rec_processor = load_rec_model(), load_rec_processor()
        print("  Surya models loaded (legacy API).\n")
        return ("old", run_ocr, det_model, det_processor, rec_model, rec_processor)

    except ImportError:
        print("\nError: surya-ocr is not installed.")
        print("  Install it with:  pip install surya-ocr")
        print("  Then re-run this script.\n")
        sys.exit(1)


# ────────────────────────────────────────────────────────────────────────────
# Preprocessing  (identical to news_extract_surya.py)
# ────────────────────────────────────────────────────────────────────────────

def preprocess_for_surya(crop_bgr: np.ndarray) -> Image.Image:
    denoised = cv2.fastNlMeansDenoisingColored(crop_bgr, h=7, hColor=7)
    h, w = denoised.shape[:2]
    if w < 100 or h < 100:
        scale    = max(100 / w, 100 / h, 1.0)
        denoised = cv2.resize(denoised, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))


# ────────────────────────────────────────────────────────────────────────────
# OCR dispatch  (identical to news_extract_surya.py)
# ────────────────────────────────────────────────────────────────────────────

def ocr_image(pil_img: Image.Image, models: tuple) -> str:
    if models[0] == "new":
        _, rec, det = models
        results = rec([pil_img], [OCR_LANGS], det)
        lines   = [line.text for line in results[0].text_lines]
    else:
        _, run_ocr, det_model, det_processor, rec_model, rec_processor = models
        results = run_ocr(
            [pil_img], [OCR_LANGS],
            det_model, det_processor,
            rec_model, rec_processor,
        )
        lines = [line.text for line in results[0].text_lines]
    return "\n".join(lines).strip()


# ────────────────────────────────────────────────────────────────────────────
# Per-page extraction
# ────────────────────────────────────────────────────────────────────────────

def extract_page(page_articles_dir: str, output_dir: str,
                 page_label: str, pdf_stem: str,
                 models: tuple) -> tuple[int, list]:
    """
    Run Surya OCR on all article images in *page_articles_dir*.

    Returns (article_count, ocr_entries)
    """
    article_files = sorted([
        f for f in os.listdir(page_articles_dir)
        if f.startswith("article_")
        and os.path.splitext(f)[1].lower() in ARTICLE_EXTENSIONS
    ])

    if not article_files:
        print(f"    Warning: no article images in '{page_articles_dir}' – skipped")
        return 0, []

    print(f"    {page_label}: {len(article_files)} article(s) ... ", end="", flush=True)

    ocr_entries = []
    word_total  = 0

    for idx, fname in enumerate(article_files, start=1):
        fpath = os.path.join(page_articles_dir, fname)
        crop  = cv2.imread(fpath)
        if crop is None:
            ocr_entries.append((idx, fname, "(could not read image)"))
            continue

        pil_img   = preprocess_for_surya(crop)
        extracted = ocr_image(pil_img, models)

        if len(extracted.replace(" ", "").replace("\n", "")) < OCR_MIN_CHARS:
            extracted = "(no text detected)"
        else:
            word_total += len(extracted.split())

        ocr_entries.append((idx, fname, extracted))

    print(f"{word_total} words total")

    # ── Write per-page text file ──────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    text_file_path = os.path.join(output_dir, f"{page_label}_extracted.txt")

    with open(text_file_path, "w", encoding="utf-8") as tf:
        tf.write(f"EXTRACTED TEXT  –  {pdf_stem}  /  {page_label}  [Surya OCR]\n")
        tf.write("=" * 70 + "\n")
        tf.write(f"OCR engine    : Surya  (langs: {', '.join(OCR_LANGS)})\n")
        tf.write(f"Source folder : {page_articles_dir}\n")
        tf.write(f"Articles      : {len(article_files)}\n")
        tf.write("=" * 70 + "\n\n")

        for num, fname, text in ocr_entries:
            tf.write(f"{num}.\n")
            tf.write(f"[{os.path.join(page_articles_dir, fname)}]\n")
            tf.write("-" * 50 + "\n")
            tf.write(text + "\n\n")

    return len(article_files), ocr_entries


# ────────────────────────────────────────────────────────────────────────────
# Per-PDF extraction
# ────────────────────────────────────────────────────────────────────────────

def extract_pdf(pdf_stem: str, models: tuple) -> int:
    """
    Process all pages under separated_articles/<pdf_stem>/ and write
    per-page text files + a combined all_pages_extracted.txt.

    Returns total article count.
    """
    articles_dir = os.path.join(ARTICLES_BASE, pdf_stem)
    output_dir   = os.path.join(EXTRACT_BASE,  pdf_stem)

    page_dirs = sorted([
        d for d in os.listdir(articles_dir)
        if os.path.isdir(os.path.join(articles_dir, d))
    ])

    if not page_dirs:
        print(f"  Warning: no page subfolders in '{articles_dir}' – skipped")
        return 0

    print(f"\n  [{pdf_stem}]  –  {len(page_dirs)} page folder(s)")

    all_entries: list[tuple[str, int, str, str]] = []  # (page_label, idx, fname, text)
    total_articles = 0

    for page_label in page_dirs:
        page_articles_dir = os.path.join(articles_dir, page_label)
        n, entries        = extract_page(page_articles_dir, output_dir,
                                          page_label, pdf_stem, models)
        total_articles += n
        for num, fname, text in entries:
            all_entries.append((page_label, num, fname, text))

    # ── Combined output file ──────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, "all_pages_extracted.txt")

    with open(combined_path, "w", encoding="utf-8") as cf:
        cf.write(f"FULL NEWSPAPER EXTRACTION  –  {pdf_stem}  [Surya OCR]\n")
        cf.write("=" * 70 + "\n")
        cf.write(f"OCR engine    : Surya  (langs: {', '.join(OCR_LANGS)})\n")
        cf.write(f"Pages         : {len(page_dirs)}\n")
        cf.write(f"Total articles: {total_articles}\n")
        cf.write("=" * 70 + "\n\n")

        current_page = None
        art_num      = 0
        for page_label, idx, fname, text in all_entries:
            if page_label != current_page:
                current_page = page_label
                art_num      = 0
                cf.write(f"\n{'─' * 70}\n")
                cf.write(f"  {page_label.upper()}\n")
                cf.write(f"{'─' * 70}\n\n")
            art_num += 1
            cf.write(f"{art_num}.\n")
            cf.write(f"[{os.path.join(articles_dir, page_label, fname)}]\n")
            cf.write("-" * 50 + "\n")
            cf.write(text + "\n\n")

    print(f"\n  Per-page text files  ->  {output_dir}/")
    print(f"  Combined text file   ->  {combined_path}")
    print(f"  Total articles       :   {total_articles}")

    return total_articles


# ────────────────────────────────────────────────────────────────────────────
# Entry point  –  fully automatic, no user input
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if not os.path.isdir(ARTICLES_BASE):
        print(f"\nError: '{ARTICLES_BASE}' folder not found.")
        print(f"  Run pdf_separator.py first.\n")
        sys.exit(1)

    pdf_stems = sorted([
        d for d in os.listdir(ARTICLES_BASE)
        if os.path.isdir(os.path.join(ARTICLES_BASE, d))
    ])

    if not pdf_stems:
        print(f"\nError: No subfolders found in '{ARTICLES_BASE}'.")
        print(f"  Run pdf_separator.py first.\n")
        sys.exit(1)

    print()
    print("=" * 70)
    print(f"  PDF NEWS EXTRACTOR  –  Surya OCR")
    print(f"  Found {len(pdf_stems)} newspaper PDF folder(s) to process")
    print(f"  Languages : {', '.join(OCR_LANGS)}")
    print("=" * 70)
    print()

    models = load_surya_models()

    grand_total = 0
    for stem in pdf_stems:
        try:
            n = extract_pdf(stem, models)
            grand_total += n
        except Exception as exc:
            print(f"\n  Error processing '{stem}': {exc}")
            import traceback; traceback.print_exc()

    print()
    print("=" * 70)
    print(f"  ALL DONE")
    print(f"  Newspapers processed : {len(pdf_stems)}")
    print(f"  Total articles       : {grand_total}")
    print(f"  Text files           -> {EXTRACT_BASE}/")
    print("=" * 70)
    print()