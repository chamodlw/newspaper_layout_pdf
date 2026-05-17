#!/usr/bin/env python3
"""
pdf_separator.py
================
Full pipeline:  PDF  →  page images  →  separated article crops

This is the PDF counterpart of newspaper_separator.py.

Folder structure produced (mirrors the image-based pipeline exactly):
    newspaper_pdfs/
        lankadeepa_2024_05_10.pdf   ← input PDFs go here

    newspaper_pages/
        lankadeepa_2024_05_10/
            page_001.png            ← rendered PDF pages (intermediate)
            page_002.png
            ...

    separated_articles/
        lankadeepa_2024_05_10/
            page_001/
                article_01.png      ← cropped article images (lossless PNG)
                article_02.png
            page_002/
                ...

    separated_layout/
        lankadeepa_2024_05_10/
            page_001_layout.jpg     ← visualisation with bounding boxes
            page_002_layout.jpg
            ...

Usage
-----
    python pdf_separator.py

The script will:
  1. Scan newspaper_pdfs/ for PDFs.
  2. Let you choose which PDF(s) to process.
  3. Try to auto-detect the rule set from the filename, then ask if unknown.
  4. Render each page to a PNG.
  5. Run the same contour-detection logic as newspaper_separator.py on each page.
  6. Save article crops + layout visualisations.

Requirements
------------
    pip install pymupdf opencv-python numpy
"""

import os
import sys

# pyrefly: ignore [missing-import]
import cv2
import numpy as np

# ── PDF rendering ─────────────────────────────────────────────────────────────
try:
    # pyrefly: ignore [missing-import]
    import fitz
except ImportError:
    print("\nError: PyMuPDF is not installed.  pip install pymupdf\n")
    sys.exit(1)

# ── Rule-set registry ─────────────────────────────────────────────────────────
try:
    from rule_sets import RULE_SETS, RULE_SET_LABELS, detect_rule_set
except ImportError as e:
    print(f"\nError loading rule sets: {e}")
    print("  Make sure the rule_sets/ package is next to this script.\n")
    sys.exit(1)


# ── Paths ─────────────────────────────────────────────────────────────────────
PDF_FOLDER      = "newspaper_pdfs"
PAGES_BASE      = "newspaper_pages"
ARTICLES_BASE   = "separated_articles"
LAYOUT_BASE     = "separated_layout"
DEFAULT_DPI     = 200
SUPPORTED_EXT   = {".pdf"}


# ────────────────────────────────────────────────────────────────────────────
# Overlap helper  (identical to newspaper_separator.py)
# ────────────────────────────────────────────────────────────────────────────

def boxes_overlap(box1, box2, threshold: float) -> bool:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_left   = max(x1, x2);  y_top    = max(y1, y2)
    x_right  = min(x1+w1, x2+w2);  y_bottom = min(y1+h1, y2+h2)
    if x_right <= x_left or y_bottom <= y_top:
        return False
    intersection  = (x_right - x_left) * (y_bottom - y_top)
    overlap_ratio = intersection / min(w1*h1, w2*h2)
    return overlap_ratio > threshold


# ────────────────────────────────────────────────────────────────────────────
# Article separation  (mirrors newspaper_separator.py logic)
# ────────────────────────────────────────────────────────────────────────────

def separate_page(image_path: str, rules: dict,
                  output_folder: str, layout_folder: str,
                  layout_name: str) -> tuple[int, list, list]:
    """
    Detect and crop articles from a single rendered page image.

    Returns (num_articles, final_boxes, saved_crops)
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(layout_folder, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from '{image_path}'")

    original                  = image.copy()
    img_height, img_width     = image.shape[:2]
    total_area                = img_width * img_height

    # ── Unpack rules ──────────────────────────────────────────────────────────
    white_threshold = rules["white_threshold"]
    kernel_size     = rules["morph_kernel_size"]
    morph_iter      = rules["morph_iterations"]
    min_area        = rules["min_area"]
    max_area        = total_area * rules["max_area_ratio"]
    min_width       = rules["min_width"]
    min_height      = rules["min_height"]
    max_width       = img_width  * rules["max_width_ratio"]
    max_height      = img_height * rules["max_height_ratio"]
    min_ar          = rules["min_aspect_ratio"]
    max_ar          = rules["max_aspect_ratio"]
    min_density     = rules["min_density"]
    overlap_thresh  = rules["overlap_threshold"]
    save_margin     = rules["save_margin"]

    # Gutter-reinforcement parameters (optional – default off)
    reinforce_gutters      = rules.get("reinforce_gutters", False)
    gutter_dark_fraction   = rules.get("gutter_dark_fraction", 0.10)
    min_vert_gutter_px     = rules.get("min_vertical_gutter_px", 8)
    min_horiz_gutter_px    = rules.get("min_horizontal_gutter_px", 8)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    gray         = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
    content_mask  = cv2.bitwise_not(white_mask)

    # ── Gutter reinforcement ───────────────────────────────────────────────────
    # Problem: Lankadeepa (and similar papers) use thin black rule lines between
    # side-by-side articles.  After thresholding those hairlines leave a few
    # dark pixels inside what should be a clean white gutter, giving the
    # subsequent dilation a foothold to bridge the gap and merge two articles
    # into one detected blob.
    #
    # Fix: before dilating, detect nearly-empty column/row bands in content_mask
    # (where <gutter_dark_fraction of pixels are content) and zero them out so
    # no dilation can cross the boundary.
    if reinforce_gutters:
        # ── Vertical gutters (separate side-by-side articles) ──────────────
        col_dark_frac = content_mask.sum(axis=0) / (255.0 * img_height)  # (w,)
        gutter_cols   = (col_dark_frac < gutter_dark_fraction).astype(np.uint8)

        # Find contiguous runs of gutter columns
        padded = np.concatenate(([0], gutter_cols, [0]))
        diff   = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]   # exclusive
        for xs, xe in zip(starts, ends):
            if xe - xs >= min_vert_gutter_px:
                content_mask[:, xs:xe] = 0  # reinforce: blank the gutter

        # ── Horizontal gutters (separate stacked articles) ─────────────────
        row_dark_frac = content_mask.sum(axis=1) / (255.0 * img_width)   # (h,)
        gutter_rows   = (row_dark_frac < gutter_dark_fraction).astype(np.uint8)

        padded = np.concatenate(([0], gutter_rows, [0]))
        diff   = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]
        for ys, ye in zip(starts, ends):
            if ye - ys >= min_horiz_gutter_px:
                content_mask[ys:ye, :] = 0  # reinforce: blank the gutter

    kernel        = np.ones(kernel_size, np.uint8)
    content_mask  = cv2.dilate(content_mask, kernel, iterations=morph_iter)

    # ── Contour detection ─────────────────────────────────────────────────────
    contours, _ = cv2.findContours(content_mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # ── Filtering ─────────────────────────────────────────────────────────────
    valid_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if not (min_area <= area <= max_area):       continue
        if w < min_width  or h < min_height:         continue
        if w > max_width  or h > max_height:         continue
        ar = w / h if h > 0 else 0
        if not (min_ar <= ar <= max_ar):             continue
        roi     = content_mask[y:y+h, x:x+w]
        density = np.sum(roi == 255) / area if area > 0 else 0
        if density < min_density:                    continue
        valid_boxes.append({"box": [x, y, w, h], "area": area})

    # ── Overlap removal ───────────────────────────────────────────────────────
    valid_boxes.sort(key=lambda b: b["area"], reverse=True)
    final_boxes: list = []
    used = [False] * len(valid_boxes)
    for i, bi in enumerate(valid_boxes):
        if used[i]: continue
        if any(boxes_overlap(bi["box"], fb, overlap_thresh) for fb in final_boxes):
            continue
        final_boxes.append(bi["box"])
        used[i] = True

    # Sort top-to-bottom, left-to-right
    final_boxes.sort(key=lambda b: (b[1], b[0]))

    # ── Save article crops ────────────────────────────────────────────────────
    saved_crops: list = []
    for idx, box in enumerate(final_boxes, start=1):
        x, y, w, h = box
        xc = max(0, x - save_margin)
        yc = max(0, y - save_margin)
        wc = min(img_width  - xc, w + 2 * save_margin)
        hc = min(img_height - yc, h + 2 * save_margin)

        article_crop = original[yc:yc+hc, xc:xc+wc]
        fname        = f"article_{idx:02d}.png"
        fpath        = os.path.join(output_folder, fname)
        cv2.imwrite(fpath, article_crop)
        saved_crops.append((fname, article_crop))

        # Draw bounding box on visualisation
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(image, (x, y - 22), (x + 55, y), (0, 200, 0), -1)
        cv2.putText(image, f"#{idx}", (x + 3, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # ── Save layout visualisation ─────────────────────────────────────────────
    viz_path = os.path.join(layout_folder, f"{layout_name}_layout.jpg")
    cv2.imwrite(viz_path, image)

    return len(final_boxes), final_boxes, saved_crops


# ────────────────────────────────────────────────────────────────────────────
# PDF rendering  (one page at a time to keep RAM usage low)
# ────────────────────────────────────────────────────────────────────────────

def render_page(doc: fitz.Document, page_idx: int,
                output_dir: str, dpi: int = DEFAULT_DPI) -> str:
    """Render a single PDF page to PNG and return its path."""
    zoom  = dpi / 72.0
    mat   = fitz.Matrix(zoom, zoom)
    page  = doc[page_idx]
    pix   = page.get_pixmap(matrix=mat, alpha=False)
    pad   = len(str(len(doc))) + 1
    fname = f"page_{str(page_idx + 1).zfill(pad)}.png"
    fpath = os.path.join(output_dir, fname)
    os.makedirs(output_dir, exist_ok=True)
    pix.save(fpath)
    return fpath


# ────────────────────────────────────────────────────────────────────────────
# Full PDF pipeline
# ────────────────────────────────────────────────────────────────────────────

def process_pdf(pdf_path: str, rules: dict, dpi: int = DEFAULT_DPI) -> int:
    """
    Render each page of the PDF and separate articles from each page.

    Returns total number of articles found across all pages.
    """
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]

    pages_dir    = os.path.join(PAGES_BASE,   pdf_stem)
    articles_dir = os.path.join(ARTICLES_BASE, pdf_stem)
    layout_dir   = os.path.join(LAYOUT_BASE,   pdf_stem)

    doc       = fitz.open(pdf_path)
    num_pages = len(doc)
    pad       = len(str(num_pages)) + 1

    print(f"\n  PDF      : {os.path.basename(pdf_path)}")
    print(f"  Pages    : {num_pages}")
    print(f"  Rules    : {rules['newspaper_name']}")
    print(f"  DPI      : {dpi}")
    print()

    total_articles = 0

    for page_idx in range(num_pages):
        page_label = f"page_{str(page_idx + 1).zfill(pad)}"
        page_articles_dir = os.path.join(articles_dir, page_label)
        print(f"  [{page_idx + 1:>{pad}}/{num_pages}]  Rendering {page_label} ... ", end="", flush=True)

        # Render page to PNG
        page_img_path = render_page(doc, page_idx, pages_dir, dpi)
        img = cv2.imread(page_img_path)
        print(f"{img.shape[1]}×{img.shape[0]}px  |  Separating ... ", end="", flush=True)

        # Separate articles from this page
        try:
            n, _, _ = separate_page(
                image_path    = page_img_path,
                rules         = rules,
                output_folder = page_articles_dir,
                layout_folder = layout_dir,
                layout_name   = page_label,
            )
            total_articles += n
            print(f"{n} article(s) found")
        except Exception as exc:
            print(f"ERROR – {exc}")

    doc.close()

    print()
    print(f"  Article images  ->  {articles_dir}/")
    print(f"  Layout images   ->  {layout_dir}/")
    print(f"  Page images     ->  {pages_dir}/")
    print(f"  Total articles  :   {total_articles}")

    return total_articles


# ────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ────────────────────────────────────────────────────────────────────────────

def scan_pdf_folder(folder: str = PDF_FOLDER) -> list[str]:
    if not os.path.isdir(folder):
        print(f"\nError: Folder '{folder}' not found.")
        print(f"  Create '{folder}/' and place your newspaper PDFs inside it.\n")
        sys.exit(1)
    pdfs = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ])
    if not pdfs:
        print(f"\nError: No PDF files found in '{folder}/'.\n")
        sys.exit(1)
    return pdfs


def select_pdfs(pdfs: list[str]) -> list[str]:
    print()
    print("+=============================================================+")
    print("|            PDFs found in 'newspaper_pdfs' folder            |")
    print("+=============================================================+")
    for i, path in enumerate(pdfs, start=1):
        fname = os.path.basename(path)
        print(f"|  [{i:2}]  {fname:<52}|")
    print("|                                                             |")
    print("|  Enter number(s) to process, or press Enter for all        |")
    print("|  Examples:  1        1,3,5        (Enter) = all            |")
    print("+=============================================================+")
    print()
    while True:
        raw = input("  Your selection: ").strip()
        if raw == "":
            print(f"\n  Processing all {len(pdfs)} PDF(s).\n")
            return pdfs
        try:
            indices = [int(x.strip()) for x in raw.split(",")]
            invalid = [i for i in indices if i < 1 or i > len(pdfs)]
            if invalid:
                print(f"  Invalid number(s): {invalid}. Try again.")
                continue
            selected = [pdfs[i - 1] for i in indices]
            print(f"\n  Selected: {', '.join(os.path.basename(p) for p in selected)}\n")
            return selected
        except ValueError:
            print("  Please enter numbers separated by commas.")


def select_rule_set_interactive() -> dict:
    print()
    print("+=============================================================+")
    print("|              Newspaper Rule-Set Selector                    |")
    print("+=============================================================+")
    for key, label in RULE_SET_LABELS.items():
        print(f"|  [{key}]  {label:<50}|")
    print("+=============================================================+")
    print()
    available = list(RULE_SETS.keys())
    while True:
        try:
            choice = int(input(f"  Enter rule-set number ({available[0]}-{available[-1]}): ").strip())
            if choice in RULE_SETS:
                print(f"\n  Selected: {RULE_SET_LABELS[choice]}\n")
                return RULE_SETS[choice]
            print(f"  Invalid choice. Please enter one of: {available}")
        except ValueError:
            print("  Please enter a valid integer.")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_pdfs      = scan_pdf_folder()
    selected_pdfs = select_pdfs(all_pdfs)

    grand_total = 0
    for pdf_path in selected_pdfs:
        print()
        print("=" * 70)
        print(f"  Processing : {os.path.basename(pdf_path)}")
        print("=" * 70)

        # Try auto-detection first
        rules = detect_rule_set(os.path.basename(pdf_path))
        if rules:
            print(f"\n  Auto-detected rule set: {rules['newspaper_name']}")
        else:
            print(f"\n  Could not auto-detect newspaper from filename.")
            rules = select_rule_set_interactive()

        try:
            n = process_pdf(pdf_path, rules)
            grand_total += n
        except Exception as exc:
            print(f"\n  Error: {exc}")
            import traceback; traceback.print_exc()

    print()
    print("=" * 70)
    print(f"  ALL DONE  |  PDFs: {len(selected_pdfs)}  |  Total articles: {grand_total}")
    print(f"  Articles  ->  {ARTICLES_BASE}/")
    print(f"  Layouts   ->  {LAYOUT_BASE}/")
    print("=" * 70)
    print()