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

    Ollama running locally with GLM-OCR pulled:
        ollama pull glm-ocr

License compliance (MIT / Apache-2.0 only):
    ollama             — MIT
    pdfplumber         — MIT
    pdfminer.six       — MIT       (pdfplumber runtime dep)
    pypdfium2          — Apache-2.0 (pdfplumber runtime dep, used for rendering)
    Pillow             — MIT-CMU   (pdfplumber runtime dep)
    charset-normalizer — MIT       (pdfminer.six runtime dep)
    cryptography       — Apache-2.0 OR BSD-2 (pdfminer.six runtime dep)
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
    print("Error: 'pdfplumber' not found.  Install it with: pip install pdfplumber")
    sys.exit(1)

try:
    import ollama
except ImportError:
    print("Error: 'ollama' not found.  Install it with: pip install ollama")
    sys.exit(1)


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


# ── Spinner ───────────────────────────────────────────────────────────────────

class Spinner:
    """Simple CLI spinner that runs in a background thread."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = ""):
        self.message  = message
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        idx = 0
        while not self._stop.is_set():
            frame = self.FRAMES[idx % len(self.FRAMES)]
            sys.stdout.write(f"\r  {frame}  {self.message}")
            sys.stdout.flush()
            time.sleep(0.08)
            idx += 1

    def start(self) -> "Spinner":
        self._thread.start()
        return self

    def stop(self, final_message: str = "") -> None:
        self._stop.set()
        self._thread.join()
        # Clear the spinner line
        sys.stdout.write(f"\r{' ' * (len(self.message) + 6)}\r")
        if final_message:
            sys.stdout.write(final_message)
        sys.stdout.flush()


# ── PDF rasterisation (pdfplumber — MIT) ──────────────────────────────────────

def rasterise_pdf(pdf_path: str, dpi: int) -> list:
    """
    Render every page of *pdf_path* to JPEG bytes at *dpi* resolution.
    Returns a list of raw JPEG bytes, one per page.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            img = page.to_image(resolution=dpi).original  # PIL Image
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            pages.append(buf.getvalue())
    return pages


# ── OCR via Ollama with streaming (ollama — MIT) ───────────────────────────────

def ocr_page(jpeg_bytes: bytes, host: str, model: str, page_num: int) -> str:
    """
    Stream the GLM-OCR response token-by-token, printing each chunk as it
    arrives so the terminal stays live. Returns the full Markdown string.
    """
    b64    = base64.b64encode(jpeg_bytes).decode("utf-8")
    client = ollama.Client(host=host)

    print()  # newline before streamed output begins

    try:
        chunks = client.chat(
            model=model,
            messages=[
                {
                    "role":    "user",
                    "content": OCR_PROMPT,
                    "images":  [b64],
                }
            ],
            stream=True,
        )

        parts = []
        for chunk in chunks:
            token = chunk["message"]["content"]
            sys.stdout.write(token)
            sys.stdout.flush()
            parts.append(token)

        print()  # newline after streamed output ends
        return "".join(parts).strip()

    except ollama.ResponseError as exc:
        msg = f"<!-- OCR failed for page {page_num}: {exc} -->"
        print(f"\n  [!] Ollama error on page {page_num}: {exc}")
        return msg
    except Exception as exc:  # noqa: BLE001
        msg = f"<!-- OCR failed for page {page_num}: {exc} -->"
        print(f"\n  [!] Unexpected error on page {page_num}: {exc}")
        return msg


# ── Orchestration ─────────────────────────────────────────────────────────────

def convert(pdf_path: str, output_path: str, host: str, model: str, dpi: int) -> None:
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.isfile(pdf_path):
        print(f"Error: File not found — {pdf_path}")
        sys.exit(1)

    print(f"PDF     : {pdf_path}")
    print(f"Output  : {output_path}")
    print(f"Host    : {host}")
    print(f"Model   : {model}")
    print(f"DPI     : {dpi}")
    print()

    # Rasterise with a spinner (can be slow for large PDFs)
    spinner = Spinner("Rasterising PDF pages ...").start()
    pages   = rasterise_pdf(pdf_path, dpi)
    total   = len(pages)
    spinner.stop(f"  Rasterised {total} page(s)\n\n")

    pages_md = []
    for idx, jpeg_bytes in enumerate(pages, start=1):
        print(f"── Page {idx}/{total} ", end="", flush=True)

        # Spinner while waiting for the first token
        spinner = Spinner(f"waiting for model response ...").start()

        # We need to get the first chunk before stopping the spinner.
        # Wrap the streaming so the spinner stops on first token.
        b64    = base64.b64encode(jpeg_bytes).decode("utf-8")
        client = ollama.Client(host=host)

        try:
            stream = client.chat(
                model=model,
                messages=[
                    {
                        "role":    "user",
                        "content": OCR_PROMPT,
                        "images":  [b64],
                    }
                ],
                stream=True,
            )

            parts        = []
            first_chunk  = True

            for chunk in stream:
                token = chunk["message"]["content"]
                if first_chunk:
                    spinner.stop()
                    print()          # newline; streamed text starts here
                    first_chunk = False
                sys.stdout.write(token)
                sys.stdout.flush()
                parts.append(token)

            print()                  # newline after page output
            md = "".join(parts).strip()

        except ollama.ResponseError as exc:
            spinner.stop()
            print(f"\n  [!] Ollama error on page {idx}: {exc}")
            md = f"<!-- OCR failed for page {idx}: {exc} -->"
        except Exception as exc:     # noqa: BLE001
            spinner.stop()
            print(f"\n  [!] Unexpected error on page {idx}: {exc}")
            md = f"<!-- OCR failed for page {idx}: {exc} -->"

        pages_md.append(md)
        print()

    header    = f"<!-- Generated by pdf_to_markdown.py | source: {os.path.basename(pdf_path)} -->\n\n"
    separator = "\n\n---\n\n"
    full_md   = header + separator.join(pages_md)

    output_path = os.path.abspath(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(full_md)

    print(f"Done! Markdown saved to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown using GLM-OCR via Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdf", help="Path to the input PDF file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output Markdown file (default: <pdf_name>.md beside the PDF)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Ollama server URL (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Render DPI — higher = better quality but slower (default: {DEFAULT_DPI})",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.output is None:
        pdf         = Path(args.pdf)
        args.output = str(pdf.parent / (pdf.stem + ".md"))

    convert(
        pdf_path    = args.pdf,
        output_path = args.output,
        host        = args.host,
        model       = args.model,
        dpi         = args.dpi,
    )


if __name__ == "__main__":
    main()