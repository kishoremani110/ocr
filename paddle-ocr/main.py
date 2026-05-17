#!/usr/bin/env python3
"""
pdf_to_markdown.py — Convert a PDF to Markdown using PaddleOCR.

IMPORTANT: Do NOT rename this script to paddle.py — that name shadows
the paddlepaddle package and causes 'paddle is not a package' errors.

All dependencies are MIT or Apache 2.0 licensed:
  - paddlepaddle      Apache 2.0
  - paddleocr         Apache 2.0
  - pdfplumber        MIT
  - Pillow            MIT (HPND, OSI approved)

Usage:
    python pdf_to_markdown.py input.pdf
    python pdf_to_markdown.py input.pdf -o output.md
    python pdf_to_markdown.py input.pdf --pages 1-5
    python pdf_to_markdown.py input.pdf --dpi 200 --lang en
"""

import os

# ── oneDNN / PIR workaround ──────────────────────────────────────────────────
# On Windows, PaddlePaddle's oneDNN (MKL-DNN) backend can raise:
#   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
#     [pir::ArrayAttribute<pir::DoubleAttribute>]
# Setting these env-vars before any paddle import is the first line of defence.
# A second, stronger fix (paddle.set_flags) is applied inside convert() after
# paddle is imported, because flag names changed across paddle 2.x / 3.x.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_enable_new_ir_in_executor"] = "0"   # paddle 3.x flag name

import argparse
import re
import sys
import tempfile  # still used for the temp JPEG written per-page
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_page_range(spec: str, total: int) -> list[int]:
    """Parse a page range like '1-5,7,9-11' into a sorted list of 0-based indices."""
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = max(1, int(start.strip()))
            end = min(total, int(end.strip()))
            pages.update(range(start, end + 1))
        else:
            p = int(part.strip())
            if 1 <= p <= total:
                pages.add(p)
    return sorted(pages)


def _poly_to_item(poly, text: str):
    """Convert a polygon + text into a (y_center, height, x_left, text) tuple."""
    ys = [pt[1] for pt in poly]
    xs = [pt[0] for pt in poly]
    return (min(ys) + max(ys)) / 2, max(ys) - min(ys), min(xs), text.strip()


def _extract_boxes_and_texts(result) -> list[tuple]:
    """
    Normalise PaddleOCR predict() output into (y_center, height, x_left, text) tuples.

    PaddleOCR >= 3 / PaddleX returns one result object per image.  That object
    can surface its data in three different ways depending on the exact build:

      A) Attribute access  – result[0].rec_texts / result[0].dt_polys
      B) Dict access       – result[0]["rec_texts"] / result[0]["dt_polys"]
      C) Legacy list       – result[0] is a list of [box, (text, conf)] pairs
                             (PaddleOCR < 3, ocr() API)

    We try A → B → C in order and fall back gracefully.  If none work we
    print the raw result so the caller can report the exact structure.
    """
    if not result:
        return []

    res = result[0]

    # ── Format A: attribute access (PaddleX result objects) ──────────────────
    if hasattr(res, "rec_texts") and hasattr(res, "dt_polys"):
        return [
            _poly_to_item(poly, text)
            for poly, text in zip(res.dt_polys, res.rec_texts)
            if text and text.strip()
        ]

    # ── Format B: dict access ─────────────────────────────────────────────────
    if isinstance(res, dict) and "rec_texts" in res and "dt_polys" in res:
        return [
            _poly_to_item(poly, text)
            for poly, text in zip(res["dt_polys"], res["rec_texts"])
            if text and text.strip()
        ]

    # ── Format C: legacy list of [box, (text, conf)] pairs ───────────────────
    if isinstance(res, list):
        items = []
        for entry in res:
            try:
                box, (text, _conf) = entry
                if text and text.strip():
                    items.append(_poly_to_item(box, text))
            except (TypeError, ValueError):
                pass  # entry has unexpected shape — skip it
        return items

    # ── Unknown format — print for debugging ─────────────────────────────────
    print(
        f"[WARN] Unrecognised PaddleOCR result format — type={type(res).__name__}\n"
        f"       dir={[a for a in dir(res) if not a.startswith('_')]}\n"
        f"       repr={repr(res)[:300]}"
    )
    return []


def boxes_to_lines(result, line_gap_ratio: float = 0.6) -> list[str]:
    """
    Group OCR word-boxes into visual lines by proximity, then join them.
    Accepts both legacy ocr() and new predict() result formats.
    Returns a list of line strings in top-to-bottom order.
    """
    items = _extract_boxes_and_texts(result)
    if not items:
        return []

    items.sort(key=lambda t: t[0])  # sort by vertical position

    # Group into lines: a new line starts when y gap > line_gap_ratio * avg_height
    lines: list[list[tuple]] = []
    current_line = [items[0]]

    for item in items[1:]:
        prev_y, prev_h = current_line[-1][0], current_line[-1][1]
        this_y, this_h = item[0], item[1]
        avg_h = (prev_h + this_h) / 2 or 1
        if this_y - prev_y < line_gap_ratio * avg_h:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    lines.append(current_line)

    # Within each line sort left-to-right, then join words
    text_lines = []
    for line in lines:
        line.sort(key=lambda t: t[2])
        text_lines.append(" ".join(t[3] for t in line))

    return text_lines


def heuristic_markdown(lines: list[str]) -> str:
    """
    Apply simple heuristics to promote short, capitalised lines to headings
    and wrap everything else as normal paragraphs.
    """
    md_parts: list[str] = []
    prev_blank = True

    for line in lines:
        stripped = line.strip()
        if not stripped:
            md_parts.append("")
            prev_blank = True
            continue

        words = stripped.split()
        char_count = len(stripped)

        # Heading heuristic: ≤ 60 chars, ends without period,
        # mostly title-cased or ALL-CAPS
        is_short = char_count <= 60
        no_period = not stripped.endswith(".")
        title_cased = sum(1 for w in words if w[0].isupper()) >= len(words) * 0.6
        all_caps = stripped.isupper() and len(words) <= 8

        if is_short and no_period and (title_cased or all_caps) and len(words) >= 1:
            # Guess heading level by character count
            if char_count <= 25 and len(words) <= 4:
                md_parts.append(f"## {stripped}")
            else:
                md_parts.append(f"### {stripped}")
        else:
            # Normal paragraph text — add blank line between paragraphs
            if not prev_blank:
                md_parts.append("")
            md_parts.append(stripped)

        prev_blank = False

    return "\n".join(md_parts)


def page_image_to_markdown(ocr, image_path: str) -> str:
    """Run OCR on one page image and return its Markdown representation."""
    # PaddleOCR >= 3 uses predict() (a generator); consume it into a list.
    # PaddleOCR < 3 used ocr(); fall back gracefully if predict() is absent.
    if hasattr(ocr, "predict"):
        result = list(ocr.predict(image_path))
    else:
        result = ocr.ocr(image_path, cls=True)
    lines = boxes_to_lines(result)
    return heuristic_markdown(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def convert(
    pdf_path: str,
    output_path: str | None,
    page_spec: str | None,
    dpi: int,
    lang: str,
) -> None:
    try:
        import paddle
        from paddleocr import PaddleOCR
    except ImportError:
        sys.exit(
            "paddleocr is not installed.\n"
            "Install with:\n"
            "  pip install paddlepaddle paddleocr"
        )

    # ── oneDNN / PIR crash workaround (Windows) ─────────────────────────────
    # Config.enable_mkldnn is a C extension method — Python cannot override it.
    # Instead we wrap paddle.inference.create_predictor (a Python-level function)
    # so we can call config.disable_mkldnn() on the Config object right before
    # the Predictor is built, regardless of what PaddleX set on it beforehand.
    try:
        import paddle.inference as _pi

        _orig_create_predictor = _pi.create_predictor

        def _create_predictor_no_mkldnn(config):
            try:
                config.disable_mkldnn()
            except Exception:
                pass
            return _orig_create_predictor(config)

        _pi.create_predictor = _create_predictor_no_mkldnn
    except Exception:
        pass

    try:
        import pdfplumber
    except ImportError:
        sys.exit(
            "pdfplumber is not installed.\n"
            "Install with:\n"
            "  pip install pdfplumber"
        )

    pdf = Path(pdf_path)
    if not pdf.exists():
        sys.exit(f"File not found: {pdf_path}")

    print(f"[1/4] Initialising PaddleOCR (lang={lang}) …")
    # device='cpu' bypasses PaddleOCR's GPU auto-detection.
    try:
        ocr = PaddleOCR(use_textline_orientation=True, lang=lang, device="cpu")
    except TypeError:
        # Older PaddleOCR builds don't accept use_textline_orientation or device.
        try:
            ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        except Exception as exc:
            sys.exit(
                f"Failed to initialise PaddleOCR: {exc}\n\n"
                "This is usually a paddlepaddle version mismatch. Try:\n"
                "  pip install -U paddlepaddle paddleocr"
            )
    except Exception as exc:
        sys.exit(
            f"Failed to initialise PaddleOCR: {exc}\n\n"
            "This is usually a paddlepaddle version mismatch. Try:\n"
            "  pip install -U paddlepaddle paddleocr"
        )

    print(f"[2/4] Opening PDF with pdfplumber …")
    with pdfplumber.open(str(pdf)) as doc, tempfile.TemporaryDirectory() as tmpdir:
        total = len(doc.pages)
        print(f"      {total} page(s) found.")

        # Resolve which pages to process
        if page_spec:
            page_numbers = parse_page_range(page_spec, total)
        else:
            page_numbers = list(range(1, total + 1))

        print(f"[3/4] Rasterising & running OCR on {len(page_numbers)} page(s) …")
        md_sections: list[str] = []

        for page_num in page_numbers:
            print(f"      OCR page {page_num}/{total} …", end="\r", flush=True)

            # pdfplumber renders the page to a PIL Image via .to_image()
            # resolution= controls DPI; .original is the underlying PIL Image
            page_img = doc.pages[page_num - 1].to_image(resolution=dpi)
            img_path = str(Path(tmpdir) / f"page_{page_num:04d}.jpg")
            page_img.original.convert("RGB").save(img_path, "JPEG")

            page_md = page_image_to_markdown(ocr, img_path)
            md_sections.append(f"<!-- Page {page_num} -->\n\n{page_md}")

        print()  # newline after \r progress

    # Build the final Markdown document
    doc_title = re.sub(r"[_-]", " ", pdf.stem).title()
    header = f"# {doc_title}\n\n---\n"
    markdown = header + "\n\n---\n\n".join(md_sections)

    # Write output
    out_path = output_path or pdf.with_suffix(".md")
    Path(out_path).write_text(markdown, encoding="utf-8")
    print(f"[4/4] Markdown saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown using PaddleOCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdf", help="Input PDF file path")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output .md file path (default: same name as PDF)",
    )
    parser.add_argument(
        "--pages",
        default=None,
        metavar="RANGE",
        help="Pages to process, e.g. '1-5' or '1,3,5-8' (default: all)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Rasterisation DPI — higher = better quality, slower (default: 150)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help=(
            "PaddleOCR language code (default: en). "
            "Examples: ch, fr, de, ja, ko, es, ar. "
            "See https://paddlepaddle.github.io/PaddleOCR/latest/en/ppocr/blog/multi_languages.html"
        ),
    )

    args = parser.parse_args()
    convert(
        pdf_path=args.pdf,
        output_path=args.output,
        page_spec=args.pages,
        dpi=args.dpi,
        lang=args.lang,
    )


if __name__ == "__main__":
    main()