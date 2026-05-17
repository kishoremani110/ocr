#!/usr/bin/env python3
"""
PDF to Markdown converter using GLM-OCR model via Ollama.

Usage:
    python pdf_to_markdown.py input.pdf
    python pdf_to_markdown.py input.pdf --output output.md
    python pdf_to_markdown.py input.pdf --dpi 200 --model glm-ocr
    python pdf_to_markdown.py input.pdf --host http://localhost:11434

Requirements:
    pip install ollama pdfplumber

License compliance (MIT / Apache-2.0 only):
    ollama, pdfplumber, pdfminer.six — MIT
    pypdfium2 — Apache-2.0
    Pillow — MIT-CMU
    charset-normalizer — MIT
    cryptography — Apache-2.0 OR BSD-2
"""

import argparse
import base64
import io
import os
import sys
import threading
import time
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("Error: pip install pdfplumber"); sys.exit(1)

try:
    import ollama
except ImportError:
    print("Error: pip install ollama"); sys.exit(1)


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "glm-ocr"
DEFAULT_DPI   = 150
DEFAULT_HOST  = "http://localhost:11434"

OCR_PROMPT = (
    "You are an OCR assistant. Extract all text from this document page and "
    "return it as clean, well-structured Markdown. Preserve headings, bullet "
    "lists, numbered lists, tables, and emphasis where visible. Do not add "
    "commentary — output only the Markdown content."
)


# ── Elapsed-time ticker ────────────────────────────────────────────────────────

class Ticker:
    """
    Prints an updating elapsed-time line in place:
        waiting for first token ... 12s
    Stops (and clears the line) as soon as .stop() is called.
    """
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


# ── PDF rasterisation (pdfplumber — MIT) ──────────────────────────────────────

def rasterise_pdf(pdf_path: str, dpi: int) -> list:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            img = page.to_image(resolution=dpi).original

            # Resize so the longest edge is at most 1024px — matches what
            # the Ollama GUI sends when you drag-and-drop a screenshot.
            max_edge = 1024
            w, h     = img.size
            scale    = min(max_edge / w, max_edge / h, 1.0)  # never upscale
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h))

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            kb = len(buf.getvalue()) / 1024
            print(f"  Page {i}: {img.size[0]}×{img.size[1]}px  {kb:.0f} KB")
            pages.append(buf.getvalue())
    return pages


# ── OCR via Ollama with streaming ─────────────────────────────────────────────

def ocr_page(jpeg_bytes: bytes, host: str, model: str, page_num: int) -> str:
    b64    = base64.b64encode(jpeg_bytes).decode("utf-8")
    client = ollama.Client(host=host)

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

def convert(pdf_path: str, output_path: str, host: str, model: str, dpi: int) -> None:
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.isfile(pdf_path):
        print(f"Error: File not found — {pdf_path}"); sys.exit(1)

    print(f"PDF     : {pdf_path}")
    print(f"Output  : {output_path}")
    print(f"Host    : {host}")
    print(f"Model   : {model}")
    print(f"DPI     : {dpi}\n")

    ticker = Ticker("Rasterising PDF ...").start()
    pages  = rasterise_pdf(pdf_path, dpi)
    ticker.stop()
    total  = len(pages)
    print(f"  Rasterised {total} page(s)\n")

    pages_md = []
    for idx, jpeg_bytes in enumerate(pages, start=1):
        print(f"── Page {idx}/{total}")
        md = ocr_page(jpeg_bytes, host, model, idx)
        pages_md.append(md)

    header    = f"<!-- source: {os.path.basename(pdf_path)} -->\n\n"
    separator = "\n\n---\n\n"
    full_md   = header + separator.join(pages_md)

    output_path = os.path.abspath(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(full_md, encoding="utf-8")
    print(f"Done! Markdown saved to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a PDF to Markdown using GLM-OCR via Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("pdf")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--host",  default=DEFAULT_HOST)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dpi",   type=int, default=DEFAULT_DPI)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.output is None:
        pdf        = Path(args.pdf)
        args.output = str(pdf.parent / (pdf.stem + ".md"))
    convert(args.pdf, args.output, args.host, args.model, args.dpi)


if __name__ == "__main__":
    main()
