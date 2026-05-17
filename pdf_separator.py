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

import cv2
import numpy as np

# ── PDF rendering ─────────────────────────────────────────────────────────────
try:
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
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _apply_exclusion_zones(mask: np.ndarray, zones: list,
                            img_height: int, img_width: int) -> np.ndarray:
    """
    Zero out rectangular regions specified as fractional (x, y, w, h) tuples.
    Applied to the content mask so no contours are detected in excluded areas.
    """
    for (x_f, y_f, w_f, h_f) in zones:
        x1 = int(x_f * img_width)
        y1 = int(y_f * img_height)
        x2 = min(img_width,  x1 + int(w_f * img_width))
        y2 = min(img_height, y1 + int(h_f * img_height))
        mask[y1:y2, x1:x2] = 0
    return mask


def _reinforce_gutters(mask: np.ndarray,
                       dark_frac: float,
                       min_vert_px: int,
                       min_horiz_px: int) -> np.ndarray:
    """
    Zero out thin white vertical and horizontal strips (column/row gutters) so
    that subsequent dilation cannot bridge across them and merge adjacent articles.

    Only strips where the fraction of dark pixels is below ``dark_frac`` AND
    the run of such columns/rows is at least ``min_vert_px`` / ``min_horiz_px``
    pixels wide are cleared.
    """
    h, w = mask.shape

    # --- Vertical gutters (between columns) ---
    col_dark = np.count_nonzero(mask, axis=0) / h   # dark-pixel fraction per column
    gutter_col = col_dark < dark_frac

    padded = np.concatenate([[False], gutter_col, [False]])
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends   = np.where( padded[:-1] & ~padded[1:])[0]
    for s, e in zip(starts, ends):
        if (e - s) >= min_vert_px:
            mask[:, s:e] = 0

    # --- Horizontal gutters (between rows) ---
    row_dark = np.count_nonzero(mask, axis=1) / w   # dark-pixel fraction per row
    gutter_row = row_dark < dark_frac

    padded = np.concatenate([[False], gutter_row, [False]])
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends   = np.where( padded[:-1] & ~padded[1:])[0]
    for s, e in zip(starts, ends):
        if (e - s) >= min_horiz_px:
            mask[s:e, :] = 0

    return mask


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
                  layout_name: str,
                  page_number: int = 1) -> tuple[int, list, list]:
    """
    Detect and crop articles from a single rendered page image.

    Parameters
    ----------
    image_path   : path to the rendered page PNG.
    rules        : rule dict for this specific page (may be page-specific).
    output_folder: where to save article crops.
    layout_folder: where to save the layout visualisation.
    layout_name  : base name for the layout visualisation file.
    page_number  : 1-based page index — used to look up exclusion zones.

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

    # ── Preprocessing ─────────────────────────────────────────────────────────
    gray          = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
    content_mask  = cv2.bitwise_not(white_mask)

    # ── Exclusion zones (applied BEFORE gutter reinforcement and dilation) ────
    # Zones are zero'd out so no content from headers/footers/ads feeds into
    # morphology or contour detection.
    zones = rules.get("excluded_zones_by_page", {}).get(page_number, [])
    if zones:
        content_mask = _apply_exclusion_zones(
            content_mask, zones, img_height, img_width)

    # ── Gutter reinforcement (applied BEFORE dilation) ────────────────────────
    # Zero out thin white column/row strips so dilation cannot bridge across
    # them and merge articles from adjacent columns.
    if rules.get("reinforce_gutters", False):
        content_mask = _reinforce_gutters(
            content_mask,
            dark_frac    = rules.get("gutter_dark_fraction",      0.05),
            min_vert_px  = rules.get("min_vertical_gutter_px",    10),
            min_horiz_px = rules.get("min_horizontal_gutter_px",  8),
        )

    # ── Morphological dilation (bridges gaps within a single article) ─────────
    kernel       = np.ones(kernel_size, np.uint8)
    content_mask = cv2.dilate(content_mask, kernel, iterations=morph_iter)

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

    # Some newspapers (e.g. Lankadeepa) supply a per-page rule factory so that
    # page 1 (front), page 2 (classifieds), and inner pages each get their own
    # thresholds, kernel sizes, and exclusion zones.
    page_rules_factory = rules.get("page_rules_factory")

    for page_idx in range(num_pages):
        page_number       = page_idx + 1
        page_label        = f"page_{str(page_number).zfill(pad)}"
        page_articles_dir = os.path.join(articles_dir, page_label)
        print(f"  [{page_number:>{pad}}/{num_pages}]  Rendering {page_label} ... ", end="", flush=True)

        # Render page to PNG
        page_img_path = render_page(doc, page_idx, pages_dir, dpi)
        img = cv2.imread(page_img_path)
        print(f"{img.shape[1]}×{img.shape[0]}px  |  Separating ... ", end="", flush=True)

        # Resolve per-page rules (falls back to the shared dict if no factory)
        page_rules = page_rules_factory(page_number) if page_rules_factory else rules

        # Separate articles from this page
        try:
            n, _, _ = separate_page(
                image_path    = page_img_path,
                rules         = page_rules,
                output_folder = page_articles_dir,
                layout_folder = layout_dir,
                layout_name   = page_label,
                page_number   = page_number,
            )
            total_articles += n
            print(f"{n} article(s) found  [{page_rules['newspaper_name']}]")
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