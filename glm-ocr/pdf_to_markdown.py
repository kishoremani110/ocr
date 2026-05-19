#!/usr/bin/env python3
"""
PDF to Markdown converter using GLM-OCR model via Ollama.

Usage:
    python pdf_to_markdown.py input.pdf
    python pdf_to_markdown.py input.pdf --output output.md
    python pdf_to_markdown.py input.pdf --dpi 200 --model glm-ocr
    python pdf_to_markdown.py input.pdf --host http://172.17.0.209:11434

Environment variables (loaded from .env):
    GLM_OCR_HOST  - Ollama server URL (default: http://localhost:11434)
    GLM_OCR_MODEL - Model name (default: glm-ocr)

Requirements:
    pip install ollama pdfplumber python-dotenv
"""

import argparse
import base64
import io
import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    import pdfplumber
except ImportError:
    print("Error: pip install pdfplumber"); sys.exit(1)

try:
    import ollama
except ImportError:
    print("Error: pip install ollama"); sys.exit(1)


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = os.environ.get("GLM_OCR_MODEL", "glm-ocr")
DEFAULT_DPI   = 150
DEFAULT_HOST  = os.environ.get("GLM_OCR_HOST", "http://localhost:11434")
if not DEFAULT_HOST.startswith("http"):
    DEFAULT_HOST = f"http://{DEFAULT_HOST}:{os.environ.get('GLM_OCR_PORT', '11434')}"

OCR_PROMPT = (
    "You are an OCR assistant. Extract all text from this document page and "
    "return it as clean, well-structured Markdown.\n\n"
    "CRITICAL TABLE RULES:\n"
    "- Count the exact number of columns from the table header.\n"
    "- Every row MUST have the same number of pipe (|) separators as the header.\n"
    "- If a cell is empty/blank, write it as '| |' (pipe space pipe).\n"
    "- NEVER skip or collapse empty cells.\n"
    "- Example: '| Data | | Value | |' has 4 columns, 2 of which are empty.\n\n"
    "- If a cell appears misaligned or spans awkardly, still extract its content into "
    "  the correct column position.\n"
    "- Extract all visible text in every cell, even if the cell borders look broken. \n\n"
    "SUBSCRIPT AND SUPERSCRIPT RULES:\n"
    "- Use HTML tags for superscripts: x<sup>2</sup>, 10<sup>3</sup>\n"
    "- Use HTML tags for subscripts: H<sub>2</sub>O, CO<sub>2</sub>\n"
    "- NEVER use LaTeX notation like ^{} or _{}\n\n"
    "Preserve headings, bullet lists, numbered lists, and emphasis where visible. "
    "Do not add commentary — output only the Markdown content."
)


# ── Elapsed-time ticker ────────────────────────────────────────────────────────

class Ticker:
    FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(self, label: str):
        self.label   = label
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        t0  = time.time()
        idx = 0
        while not self._stop.is_set():
            elapsed = int(time.time() - t0)
            frame   = self.FRAMES[idx % len(self.FRAMES)]
            line    = f"\r  {frame}  {self.label} {elapsed}s"
            sys.stdout.write(line)
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1

    def start(self) -> "Ticker":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()


# ── PDF/Image rasterisation ───────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def rasterise_image(img_path: str) -> list:
    from PIL import Image
    img = Image.open(img_path).convert("RGB")

    max_edge = 1024
    w, h     = img.size
    scale    = min(max_edge / w, max_edge / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    kb = len(buf.getvalue()) / 1024
    print(f"  Image: {img.size[0]}×{img.size[1]}px  {kb:.0f} KB")
    return [buf.getvalue()]


def rasterise_pdf(pdf_path: str, dpi: int) -> list:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            img = page.to_image(resolution=dpi).original

            max_edge = 1024
            w, h     = img.size
            scale    = min(max_edge / w, max_edge / h, 1.0)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h))

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            kb = len(buf.getvalue()) / 1024
            print(f"  Page {i}: {img.size[0]}×{img.size[1]}px  {kb:.0f} KB")
            pages.append(buf.getvalue())
    return pages


def rasterise_input(file_path: str, dpi: int) -> list:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return rasterise_pdf(file_path, dpi)
    elif ext in IMAGE_EXTENSIONS:
        return rasterise_image(file_path)
    else:
        print(f"Error: Unsupported file type '{ext}'. Supported: .pdf, .png, .jpg, .jpeg, .bmp, .tiff, .webp")
        sys.exit(1)


# ── OCR via Ollama with streaming ─────────────────────────────────────────────

def ocr_page(jpeg_bytes: bytes, host: str, model: str, page_num: int) -> str:
    b64    = base64.b64encode(jpeg_bytes).decode("utf-8")
    client = ollama.Client(host=host, timeout=300)

    ticker = Ticker("waiting for first token ...").start()
    first  = True
    parts  = []

    try:
        stream = client.chat(
            model=model,
            messages=[{"role": "user", "content": OCR_PROMPT, "images": [b64]}],
            stream=True,
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if first:
                ticker.stop()
                first = False
            sys.stdout.write(token)
            sys.stdout.flush()
            parts.append(token)

    except ollama.ResponseError as exc:
        ticker.stop()
        print(f"\n  [!] Ollama error on page {page_num}: {exc}")
        return f"<!-- OCR failed for page {page_num}: {exc} -->"
    except Exception as exc:
        ticker.stop()
        print(f"\n  [!] Unexpected error on page {page_num}: {exc}")
        return f"<!-- OCR failed for page {page_num}: {exc} -->"

    print("\n")
    return "".join(parts).strip()


# ── Orchestration ─────────────────────────────────────────────────────────────

def convert(file_path: str, output_path: str, host: str, model: str, dpi: int) -> None:
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        print(f"Error: File not found — {file_path}"); sys.exit(1)

    print(f"Input   : {file_path}")
    print(f"Output  : {output_path}")
    print(f"Host    : {host}")
    print(f"Model   : {model}")
    print(f"DPI     : {dpi}\n")

    ticker = Ticker("Rasterising input ...").start()
    pages  = rasterise_input(file_path, dpi)
    ticker.stop()
    total  = len(pages)
    print(f"  Rasterised {total} page(s)\n")

    pages_md = []
    for idx, jpeg_bytes in enumerate(pages, start=1):
        print(f"── Page {idx}/{total}")
        md = ocr_page(jpeg_bytes, host, model, idx)
        pages_md.append(md)

    header    = f"<!-- source: {os.path.basename(file_path)} -->\n\n"
    separator = "\n\n---\n\n"
    full_md   = header + separator.join(pages_md)

    output_path = os.path.abspath(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(full_md, encoding="utf-8")
    print(f"Done! Markdown saved to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a PDF or image to Markdown using GLM-OCR via Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="PDF or image file (png, jpg, bmp, tiff, webp)")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--host",  default=DEFAULT_HOST)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dpi",   type=int, default=DEFAULT_DPI)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.output is None:
        inp         = Path(args.input)
        args.output = str(inp.parent / (inp.stem + ".md"))
    convert(args.input, args.output, args.host, args.model, args.dpi)


if __name__ == "__main__":
    main()
