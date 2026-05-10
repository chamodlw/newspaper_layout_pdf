#!/usr/bin/env python3
"""
pdf_to_images.py
================
Converts every page of a newspaper PDF into a high-resolution PNG image.

Output folder structure (mirrors the image-based pipeline):
    newspaper_pages/
        <pdf_stem>/
            page_001.png
            page_002.png
            ...

Uses PyMuPDF (fitz) for rendering – fast, no external binaries needed,
and produces lossless PNG at configurable DPI (default 200 DPI which
gives ~1650 × 2330 px for an A2 broadsheet – good balance of quality
vs. file size for Surya OCR).

Requirements:
    pip install pymupdf
"""

import os
import sys

try:
    import fitz          # PyMuPDF
except ImportError:
    print("\nError: PyMuPDF is not installed.")
    print("  Install it with:  pip install pymupdf\n")
    sys.exit(1)


# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_DPI    = 200      # rendering resolution (higher = better OCR, larger files)
PAGES_BASE_DIR = "newspaper_pages"
SUPPORTED_EXT  = {".pdf"}


# ────────────────────────────────────────────────────────────────────────────
# Core converter
# ────────────────────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: str,
                  output_dir: str | None = None,
                  dpi: int = DEFAULT_DPI) -> list[str]:
    """
    Render every page of *pdf_path* to a PNG and save under *output_dir*.

    Parameters
    ----------
    pdf_path   : path to the input PDF file
    output_dir : directory where page PNGs are saved
                 (defaults to  newspaper_pages/<pdf_stem>/)
    dpi        : rendering resolution in dots-per-inch

    Returns
    -------
    List of absolute paths to the generated PNG files (sorted by page).
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: '{pdf_path}'")

    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]

    if output_dir is None:
        output_dir = os.path.join(PAGES_BASE_DIR, pdf_stem)
    os.makedirs(output_dir, exist_ok=True)

    doc        = fitz.open(pdf_path)
    num_pages  = len(doc)
    zoom       = dpi / 72.0          # PyMuPDF default unit is 72 DPI
    mat        = fitz.Matrix(zoom, zoom)
    pad_width  = len(str(num_pages)) + 1   # zero-padding width

    print(f"\n  PDF      : {os.path.basename(pdf_path)}")
    print(f"  Pages    : {num_pages}")
    print(f"  DPI      : {dpi}")
    print(f"  Output   : {output_dir}\n")

    saved_paths: list[str] = []

    for page_num in range(num_pages):
        page     = doc[page_num]
        pix      = page.get_pixmap(matrix=mat, alpha=False)
        fname    = f"page_{str(page_num + 1).zfill(pad_width)}.png"
        fpath    = os.path.join(output_dir, fname)
        pix.save(fpath)
        saved_paths.append(fpath)
        print(f"  [{page_num + 1:>{pad_width}}/{num_pages}]  {fname}  "
              f"({pix.width} × {pix.height} px)")

    doc.close()

    print(f"\n  Done – {len(saved_paths)} page(s) saved to '{output_dir}'\n")
    return saved_paths


# ────────────────────────────────────────────────────────────────────────────
# CLI – scan newspaper_pdfs/ folder
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    PDF_FOLDER = "newspaper_pdfs"

    if not os.path.isdir(PDF_FOLDER):
        print(f"\nError: Folder '{PDF_FOLDER}' not found.")
        print(f"  Create a '{PDF_FOLDER}/' folder and place your PDF files inside it.\n")
        sys.exit(1)

    pdfs = sorted([
        os.path.join(PDF_FOLDER, f)
        for f in os.listdir(PDF_FOLDER)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ])

    if not pdfs:
        print(f"\nError: No PDF files found in '{PDF_FOLDER}/'.\n")
        sys.exit(1)

    print()
    print("=" * 70)
    print(f"  PDF → IMAGES CONVERTER")
    print(f"  Found {len(pdfs)} PDF(s) in '{PDF_FOLDER}/'")
    print("=" * 70)

    total_pages = 0
    for pdf_path in pdfs:
        print(f"\n{'─' * 70}")
        try:
            pages = pdf_to_images(pdf_path)
            total_pages += len(pages)
        except Exception as exc:
            print(f"  Error converting '{os.path.basename(pdf_path)}': {exc}")
            import traceback; traceback.print_exc()

    print()
    print("=" * 70)
    print(f"  ALL DONE  |  PDFs: {len(pdfs)}  |  Total pages: {total_pages}")
    print(f"  Page images  ->  {PAGES_BASE_DIR}/")
    print("=" * 70)
    print()