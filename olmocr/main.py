#!/usr/bin/env python3
"""
pdf_to_markdown.py — Convert a PDF to Markdown using olmOCR2 via Ollama.

Dependencies (all MIT / Apache-2.0 / BSD licensed — no system tools required):
    pip install ollama pypdfium2 pillow

Usage:
    python pdf_to_markdown.py document.pdf             # GPU (default)
    python pdf_to_markdown.py document.pdf --gpu       # explicit GPU
    python pdf_to_markdown.py document.pdf --cpu       # force CPU
    python pdf_to_markdown.py document.pdf -o out.md   # custom output path
    python pdf_to_markdown.py document.pdf --pages 2-6 # page range
    python pdf_to_markdown.py document.pdf --dpi 200   # higher quality
"""

import argparse
import io
import os
import sys
from pathlib import Path

import pypdfium2 as pdfium  # Apache-2.0 / BSD-3-Clause — no poppler needed
import ollama               # MIT

MODEL = "richardyoung/olmocr2:7b-q8"
DEFAULT_DPI = 150           # 72 pt/inch × scale → pixel density sent to model

OCR_PROMPT = (
    "Convert this document page to Markdown. "
    "Preserve all text content, headings, lists, tables, "
    "and mathematical expressions (use LaTeX for equations). "
    "Output only the Markdown — no commentary, no code fences."
)


# ---------------------------------------------------------------------------
# PDF → images using pypdfium2  (pure Python, Apache-2.0)
# ---------------------------------------------------------------------------

def pdf_page_count(pdf_path: Path) -> int:
    doc = pdfium.PdfDocument(str(pdf_path))
    n = len(doc)
    doc.close()
    return n


def render_page_to_bytes(pdf_path: Path, page_index: int, dpi: int) -> bytes:
    """
    Render a single PDF page to JPEG bytes using pypdfium2.
    page_index is 0-based.
    """
    scale = dpi / 72  # pdfium works in 72 dpi units
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page = doc[page_index]
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()          # PIL.Image (Pillow)
        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Ollama OCR
# ---------------------------------------------------------------------------

def check_ollama_model() -> None:
    """Abort if the model isn't available locally."""
    try:
        models = [m.model for m in ollama.list().models]
    except Exception as exc:
        sys.exit(f"Error: cannot connect to Ollama — {exc}")

    base = MODEL.split(":")[0]
    if not any(MODEL in m or base in m for m in models):
        sys.exit(
            f"Error: model '{MODEL}' not found locally.\n"
            f"Pull it first:  ollama pull {MODEL}"
        )


def ocr_page(jpeg_bytes: bytes, use_gpu: bool) -> str:
    """Send one page (as JPEG bytes) to olmOCR2 and return Markdown text."""
    import base64
    b64 = base64.b64encode(jpeg_bytes).decode()

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": OCR_PROMPT,
                "images": [b64],   # ollama-python accepts base64 strings
            }
        ],
        options={
            "num_gpu": 99 if use_gpu else 0,
            "num_ctx": 8192,
        },
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown using olmOCR2 via Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdf", help="Path to the input PDF file.")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Output Markdown file. "
            "Defaults to <pdf_name>.md in the same folder as the PDF."
        ),
    )

    accel = parser.add_mutually_exclusive_group()
    accel.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU acceleration (default behaviour).",
    )
    accel.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Force CPU-only inference (slower).",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Rendering DPI for each page (default: {DEFAULT_DPI}).",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        metavar="START-END",
        help="1-based page range, e.g. '1-5'. Defaults to all pages.",
    )

    args = parser.parse_args()
    args.use_gpu = not args.cpu   # GPU is the default unless --cpu is passed
    return args


def resolve_page_range(total: int, page_range: str | None) -> range:
    if page_range is None:
        return range(total)
    try:
        s, e = page_range.split("-")
        start, end = int(s) - 1, int(e) - 1   # convert to 0-based
    except ValueError:
        sys.exit(f"Error: --pages must be START-END (e.g. '1-5'). Got: '{page_range}'")
    if start < 0 or end >= total or start > end:
        sys.exit(
            f"Error: page range {page_range} is invalid "
            f"(document has {total} pages)."
        )
    return range(start, end + 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        sys.exit(f"Error: file not found — {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        sys.exit(f"Error: expected a .pdf file, got '{pdf_path.suffix}'")

    output_path = (
        Path(args.output) if args.output
        else pdf_path.parent / (pdf_path.stem + ".md")
    )

    print(f"PDF       : {pdf_path}")
    print(f"Output    : {output_path}")
    print(f"Mode      : {'GPU' if args.use_gpu else 'CPU'}")
    print(f"Raster DPI: {args.dpi}")
    print()

    # Pre-flight
    check_ollama_model()

    total_pages = pdf_page_count(pdf_path)
    page_indices = resolve_page_range(total_pages, args.pages)
    count = len(page_indices)
    print(f"Pages to process: {count} of {total_pages}\n")

    markdown_parts: list[str] = []

    for pos, idx in enumerate(page_indices, start=1):
        page_num = idx + 1   # human-readable (1-based)
        print(f"  [{pos}/{count}] Rendering page {page_num} …", end=" ", flush=True)
        jpeg_bytes = render_page_to_bytes(pdf_path, idx, args.dpi)
        print("OCR …", end=" ", flush=True)
        try:
            md = ocr_page(jpeg_bytes, args.use_gpu)
            print("done")
        except Exception as exc:
            print(f"FAILED — {exc}")
            md = f"<!-- OCR failed for page {page_num}: {exc} -->"

        markdown_parts.append(f"<!-- Page {page_num} -->\n\n{md}")

    final_md = "\n\n---\n\n".join(markdown_parts)
    output_path.write_text(final_md, encoding="utf-8")
    print(f"\nDone. Markdown saved to: {output_path}")


if __name__ == "__main__":
    main()